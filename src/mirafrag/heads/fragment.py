from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mirafrag.config import MiraFragConfig
from mirafrag.fragments import FRAGMENT_EDGE_FEATURE_DIM


class FragmentSpectrumHead(nn.Module):
    """
    Sparse candidate-based spectrum head.

    The head pools foundation encoder atom features over fragment formulas, appends fragment features and precursor metadata, optionally runs message passing over the fragment graph, scores formulas, can add recursive parent-to-child path evidence, expands scores to isotope/adduct peak candidates, and predicts an OOS logit.
    """

    def __init__(self, config: MiraFragConfig) -> None:
        """
        Create fragment encoders, optional graph message layers, candidate scorer, and OOS scorer.
        """
        super().__init__()
        self.num_bins = int(config.num_bins)
        self.fragment_path_layers = max(0, int(config.fragment_path_layers))
        self.fragment_gnn_layers = nn.ModuleList(
            [
                FragmentGraphMessageLayer(
                    hidden_dim=config.hidden_dim,
                    edge_dim=FRAGMENT_EDGE_FEATURE_DIM,
                    dropout=config.dropout,
                )
                for _ in range(max(0, int(config.fragment_gnn_layers)))
            ]
        )
        self.fragment_encoder = nn.Sequential(
            nn.LazyLinear(config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.LazyLinear(config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.collision_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        layers: list[nn.Module] = [
            nn.Linear(5 * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(max(0, config.num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.scorer = nn.Sequential(*layers)
        if self.fragment_path_layers > 0:
            # Preserve the retained scorer's lazy initialization sequence when
            # the optional path branch is toggled for an architecture experiment.
            with torch.random.fork_rng(devices=[]):
                self.fragment_path_edge_scorer = self._make_fragment_path_edge_scorer(
                    config
                )
                self.fragment_path_root_scorer = self._make_fragment_path_root_scorer(
                    config
                )
                self.fragment_path_score_encoder = (
                    self._make_fragment_path_score_encoder(config)
                )
                self.fragment_path_residual = self._make_fragment_path_residual(config)
                self._reset_fragment_path_residual()
        else:
            self.fragment_path_edge_scorer = None
            self.fragment_path_root_scorer = None
            self.fragment_path_score_encoder = None
            self.fragment_path_residual = None
        self.oos_scorer = self._make_oos_scorer(config)

    @staticmethod
    def _make_fragment_path_edge_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build the parent-to-child edge scorer used by recursive path propagation.

        The scorer receives parent state, child state, parent-child interactions,
        child precursor context, child collision-energy context, and the existing
        typed fragment-graph edge features.
        """
        hidden_dim = int(config.hidden_dim)
        return nn.Sequential(
            nn.Linear(5 * hidden_dim + FRAGMENT_EDGE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_fragment_path_root_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build the scorer for direct root evidence of each retained fragment.

        Root evidence allows formulas with no retained parent edge to still get
        a path feature, which is important after fragment pruning.
        """
        hidden_dim = int(config.hidden_dim)
        return nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_fragment_path_score_encoder(config: MiraFragConfig) -> nn.Module:
        """
        Encode the normalized scalar path log-score into the model hidden space.
        """
        return nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _make_fragment_path_residual(config: MiraFragConfig) -> nn.Module:
        """
        Build the zero-initialized residual that turns path evidence into logits.

        Keeping this branch residual and initially zero preserves the retained
        formula scorer at initialization while still allowing gradients to add
        recursive fragmentation-path corrections during training.
        """
        hidden_dim = int(config.hidden_dim)
        return nn.Sequential(
            nn.Linear(8 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _reset_fragment_path_residual(self) -> None:
        """
        Initialize the path residual's output layer to make the branch a no-op.
        """
        if self.fragment_path_residual is None:
            return
        final_layer = self.fragment_path_residual[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _make_oos_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build the small network that predicts out-of-support probability per spectrum.
        """
        metadata_feature_dim = 2 + 2 * int(config.metadata_dim)
        return nn.Sequential(
            nn.Linear(metadata_feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Score sparse fragment peak candidates for a batch.

        The returned dictionary contains candidate logits, OOS logits, m/z values, bins, formula indices, graph edges, batch indices, and shape metadata consumed by losses and evaluation.
        """
        fragment_batch = fragments['batch'].to(device=node_feats.device).long()
        batch_size = int(metadata_features.shape[0])
        if fragment_batch.numel() == 0:
            return {
                'kind': 'sparse',
                'logits': node_feats.new_empty(0),
                'oos_logits': self.oos_scorer(metadata_features).squeeze(-1),
                'mzs': node_feats.new_empty(0),
                'bins': torch.empty(0, dtype=torch.long, device=node_feats.device),
                'log_prior': node_feats.new_empty(0),
                'formula_index': torch.empty(
                    0,
                    dtype=torch.long,
                    device=node_feats.device,
                ),
                'edge_index': torch.empty(
                    2, 0, dtype=torch.long, device=node_feats.device
                ),
                'edge_attr': node_feats.new_empty(0, FRAGMENT_EDGE_FEATURE_DIM),
                'formula_batch': torch.empty(
                    0, dtype=torch.long, device=node_feats.device
                ),
                'batch': torch.empty(0, dtype=torch.long, device=node_feats.device),
                'batch_size': batch_size,
                'num_bins': self.num_bins,
            }

        formula_batch = fragments['formula_batch'].to(device=node_feats.device).long()
        fragment_atom_features = self._pool_fragment_atoms(
            node_feats,
            fragments['atom_index'].to(device=node_feats.device).long(),
            fragments['atom_ptr'].to(device=node_feats.device).long(),
        )
        molecule_atom_features = self._pool_molecule_atoms(
            node_feats,
            graph_batch.to(device=node_feats.device).long()
            if graph_batch is not None
            else None,
            batch_size=batch_size,
        )
        fragment_inputs = torch.cat(
            [
                fragment_atom_features,
                fragments['features'].to(
                    device=node_feats.device,
                    dtype=node_feats.dtype,
                ),
            ],
            dim=-1,
        )
        context_inputs = torch.cat(
            [
                molecule_atom_features[formula_batch],
                metadata_features[formula_batch],
            ],
            dim=-1,
        )
        formula_features = self.fragment_encoder(fragment_inputs)
        context_features = self.context_encoder(context_inputs)
        collision_features = self.collision_encoder(
            self._collision_energy_feature(metadata_features)[formula_batch]
        )
        edge_index = fragments['edge_index'].to(device=node_feats.device)
        edge_attr = fragments['edge_attr'].to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        for layer in self.fragment_gnn_layers:
            formula_features = layer(
                formula_features,
                edge_index,
                edge_attr,
                collision_features,
            )
        formula_index = fragments['formula_index'].to(device=node_feats.device).long()
        scorer_features = torch.cat(
            [
                formula_features,
                context_features,
                formula_features * context_features,
                collision_features,
                formula_features * collision_features,
            ],
            dim=-1,
        )
        formula_logits = self.scorer(scorer_features).squeeze(-1)
        formula_logits = formula_logits + self._fragment_path_delta(
            formula_features,
            context_features,
            collision_features,
            edge_index,
            edge_attr,
            formula_batch,
            batch_size=batch_size,
        )
        peak_mzs = fragments['mz'].to(device=node_feats.device, dtype=node_feats.dtype)
        peak_bins = (
            fragments['bin']
            .to(device=node_feats.device)
            .long()
            .clamp(
                0,
                self.num_bins - 1,
            )
        )
        log_prior = fragments['log_prior'].to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        logits = formula_logits[formula_index] + log_prior
        return {
            'kind': 'sparse',
            'logits': logits,
            'oos_logits': self.oos_scorer(metadata_features).squeeze(-1),
            'mzs': peak_mzs,
            'bins': peak_bins,
            'log_prior': log_prior,
            'formula_index': formula_index,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'formula_batch': formula_batch,
            'batch': fragment_batch,
            'batch_size': batch_size,
            'num_bins': self.num_bins,
        }

    def _fragment_path_delta(
        self,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Return recursive parent-to-child path corrections for formula logits.

        Parent-to-child edges define a directed fragment tree or DAG. The method
        scores root formulas and directed transitions, propagates log-space path
        evidence for a fixed number of steps, normalizes it per spectrum, and
        maps it through a zero-initialized residual. A disabled path branch
        returns exact zeros.
        """
        if self.fragment_path_layers <= 0 or formula_features.numel() == 0:
            return formula_features.new_zeros(formula_features.shape[0])
        if (
            self.fragment_path_edge_scorer is None
            or self.fragment_path_root_scorer is None
            or self.fragment_path_score_encoder is None
            or self.fragment_path_residual is None
        ):
            return formula_features.new_zeros(formula_features.shape[0])

        root_inputs = torch.cat(
            [formula_features, context_features, collision_features], dim=-1
        )
        root_scores = self.fragment_path_root_scorer(root_inputs).squeeze(-1)
        path_scores = root_scores
        if edge_index.numel() > 0 and edge_attr.numel() > 0:
            parent_mask = edge_attr[:, 0] > 0.5
            src = edge_index[0, parent_mask].long()
            dst = edge_index[1, parent_mask].long()
            parent_features = formula_features[src]
            child_features = formula_features[dst]
            edge_inputs = torch.cat(
                [
                    parent_features,
                    child_features,
                    parent_features * child_features,
                    context_features[dst],
                    collision_features[dst],
                    edge_attr[parent_mask],
                ],
                dim=-1,
            )
            edge_scores = self.fragment_path_edge_scorer(edge_inputs).squeeze(-1)
            root_frontier = self._fragment_path_root_frontier(root_scores, dst)
            path_scores = self._propagate_fragment_path_scores(
                root_frontier,
                edge_scores,
                src,
                dst,
                num_nodes=formula_features.shape[0],
            )
            path_scores = torch.where(
                self._fragment_path_reachable(path_scores), path_scores, root_scores
            )

        normalized_scores = self._standardize_by_batch(
            path_scores,
            formula_batch,
            batch_size=batch_size,
        )
        path_features = self.fragment_path_score_encoder(
            normalized_scores.unsqueeze(-1)
        )
        residual_inputs = torch.cat(
            [
                formula_features,
                context_features,
                formula_features * context_features,
                collision_features,
                formula_features * collision_features,
                path_features,
                formula_features * path_features,
                context_features * path_features,
            ],
            dim=-1,
        )
        return self.fragment_path_residual(residual_inputs).squeeze(-1)

    @staticmethod
    def _fragment_path_root_frontier(
        root_scores: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Keep direct root evidence only for formulas without a retained parent.

        Fragment trees can be pruned before scoring. A formula with no retained
        incoming parent edge is therefore treated as a root of the retained
        fragment graph, while children must earn their path evidence through
        parent-to-child propagation.
        """
        has_parent = torch.zeros(
            root_scores.shape[0], dtype=torch.bool, device=root_scores.device
        )
        has_parent.index_fill_(0, dst, True)
        return root_scores.masked_fill(
            has_parent, FragmentSpectrumHead._fragment_path_neg_inf(root_scores)
        )

    @staticmethod
    def _fragment_path_neg_inf(values: torch.Tensor) -> float:
        """
        Return a finite log-space sentinel that behaves like negative infinity.

        A finite sentinel avoids ``inf - inf`` and ``0 * NaN`` patterns in
        autograd while remaining far outside the range of plausible learned path
        scores.
        """
        if values.dtype in (torch.float16, torch.bfloat16):
            return -1.0e4
        return -1.0e9

    @staticmethod
    def _fragment_path_reachable(values: torch.Tensor) -> torch.Tensor:
        """
        Return which propagated path scores represent reachable nodes.
        """
        return values > (FragmentSpectrumHead._fragment_path_neg_inf(values) * 0.5)

    def _propagate_fragment_path_scores(
        self,
        root_scores: torch.Tensor,
        edge_scores: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Accumulate root-to-fragment path log-scores up to the configured depth.

        A frontier is propagated at each step, then merged into the total with
        ``logaddexp``. This avoids repeatedly counting the same one-step path
        while still supporting multiple retained parents for one child.
        """
        total_scores = root_scores
        frontier_scores = root_scores
        for _ in range(self.fragment_path_layers):
            candidate_scores = frontier_scores[src] + edge_scores
            propagated_scores = self._scatter_logsumexp(
                candidate_scores, dst, size=num_nodes
            )
            total_scores = torch.logaddexp(total_scores, propagated_scores)
            frontier_scores = propagated_scores
        return total_scores

    @staticmethod
    def _scatter_logsumexp(
        values: torch.Tensor,
        index: torch.Tensor,
        *,
        size: int,
    ) -> torch.Tensor:
        """
        Compute ``logsumexp(values)`` grouped by destination index.
        """
        grouped_max = values.new_full(
            (size,), FragmentSpectrumHead._fragment_path_neg_inf(values)
        )
        grouped_max.scatter_reduce_(0, index, values, reduce='amax', include_self=True)
        finite_groups = torch.isfinite(grouped_max)
        value_groups = grouped_max[index]
        finite_values = torch.isfinite(values) & torch.isfinite(value_groups)
        safe_values = torch.where(finite_values, values, torch.zeros_like(values))
        safe_groups = torch.where(
            finite_values, value_groups, torch.zeros_like(value_groups)
        )
        shifted = torch.where(
            finite_values,
            torch.exp(safe_values - safe_groups),
            torch.zeros_like(values),
        )
        grouped_sum = values.new_zeros(size)
        grouped_sum.index_add_(0, index, shifted)
        grouped_logsum = torch.log(grouped_sum.clamp_min(1e-12)) + grouped_max
        return torch.where(finite_groups, grouped_logsum, grouped_max)

    @staticmethod
    def _standardize_by_batch(
        values: torch.Tensor,
        batch_index: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Standardize scalar formula scores independently for each spectrum.
        """
        if values.numel() == 0:
            return values
        counts = values.new_zeros(batch_size)
        counts.index_add_(0, batch_index, values.new_ones(values.shape[0]))
        sums = values.new_zeros(batch_size)
        sums.index_add_(0, batch_index, values)
        means = sums / counts.clamp_min(1.0)
        centered = values - means[batch_index]
        squared_sums = values.new_zeros(batch_size)
        squared_sums.index_add_(0, batch_index, centered.square())
        variances = squared_sums / counts.clamp_min(1.0)
        scales = torch.sqrt(variances + 1e-6)
        return centered / scales[batch_index].clamp_min(1e-3)

    @staticmethod
    def _collision_energy_feature(metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Return the normalized collision-energy channel from metadata features.

        Metadata is built as ``[precursor_mz, collision_energy, adduct_embedding,
        instrument_embedding]``. A zero fallback keeps tests or external callers
        with reduced metadata tensors well-defined.
        """
        if metadata_features.shape[-1] <= 1:
            return metadata_features.new_zeros(metadata_features.shape[0], 1)
        return metadata_features[:, 1:2]

    @staticmethod
    def _pool_fragment_atoms(
        node_feats: torch.Tensor,
        atom_index: torch.Tensor,
        atom_ptr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool atom features for each fragment formula using atom pointer ranges.
        """
        num_fragments = max(int(atom_ptr.numel()) - 1, 0)
        pooled = node_feats.new_zeros(num_fragments, node_feats.shape[-1])
        if num_fragments == 0:
            return pooled

        counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(1)
        fragment_idx = torch.repeat_interleave(
            torch.arange(num_fragments, device=node_feats.device),
            counts.to(device=node_feats.device),
        )
        pooled.index_add_(0, fragment_idx, node_feats[atom_index])
        return pooled / counts.to(
            dtype=node_feats.dtype, device=node_feats.device
        ).unsqueeze(-1)

    @staticmethod
    def _pool_molecule_atoms(
        node_feats: torch.Tensor,
        graph_batch: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Mean-pool atom features for each precursor molecule in the batch.
        """
        pooled = node_feats.new_zeros(batch_size, node_feats.shape[-1])
        if graph_batch is None or graph_batch.numel() == 0:
            return pooled
        graph_batch = graph_batch.clamp(0, batch_size - 1)
        pooled.index_add_(0, graph_batch, node_feats)
        counts = node_feats.new_zeros(batch_size, 1)
        counts.index_add_(
            0,
            graph_batch,
            node_feats.new_ones(graph_batch.shape[0], 1),
        )
        return pooled / counts.clamp_min(1.0)


class FragmentGraphMessageLayer(nn.Module):
    """
    One message-passing layer over the fragment relationship graph.

    Messages are gated by edge features and collision-energy conditioning, averaged at each destination formula, and combined with residual MLP updates and layer normalization.
    """

    def __init__(self, *, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        """
        Initialize source projection, edge gate, residual update, norms, and dropout.
        """
        super().__init__()
        self.source = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_gate = nn.Linear(hidden_dim, hidden_dim)
        self.condition_film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        conditioning_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one round of edge- and collision-conditioned message passing.
        """
        hidden = self._apply_conditioning(node_features, conditioning_features)
        if edge_index.numel() == 0:
            return hidden

        src = edge_index[0].long()
        dst = edge_index[1].long()
        gate_logits = self.edge_gate(edge_attr) + self.condition_gate(
            conditioning_features[dst]
        )
        messages = self.source(hidden[src]) * torch.sigmoid(gate_logits)
        aggregated = hidden.new_zeros(hidden.shape)
        aggregated.index_add_(0, dst, messages)
        degree = hidden.new_zeros(hidden.shape[0], 1)
        degree.index_add_(0, dst, hidden.new_ones(dst.shape[0], 1))
        aggregated = aggregated / degree.clamp_min(1.0)

        hidden = self.message_norm(hidden + self.dropout(aggregated))
        return self.update_norm(hidden + self.dropout(self.update(hidden)))

    def _apply_conditioning(
        self,
        node_features: torch.Tensor,
        conditioning_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply feature-wise collision-energy modulation to fragment states.
        """
        scale, shift = self.condition_film(conditioning_features).chunk(2, dim=-1)
        return node_features * (1.0 + torch.tanh(scale)) + shift
