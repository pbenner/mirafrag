from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.parameter import is_lazy

from mirafrag.config import MiraFragConfig
from mirafrag.fragments import (
    BOND_BREAK_FEATURE_DIM,
    FRAGMENT_EDGE_FEATURE_DIM,
    FRAGMENT_FORMULA_DIM,
)

FRAGMENT_ACTION_GEOMETRY_FEATURE_DIM = 8


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
        self.config = config
        self.num_bins = int(config.num_bins)
        self.fragment_path_layers = max(0, int(config.fragment_path_layers))
        self.fragment_path_primary = bool(
            getattr(config, 'fragment_path_primary', False)
        )
        self.fragment_action_primary_layers = max(
            0, int(getattr(config, 'fragment_action_primary_layers', 0))
        )
        self.bond_break_geometry_features = bool(
            getattr(config, 'bond_break_geometry_features', False)
        )
        self.collision_feature_dim = 1
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
        self.fragment_input_dropout = nn.Dropout(config.dropout)
        self.context_input_dropout = nn.Dropout(config.dropout)
        self.collision_input_dropout = nn.Dropout(config.dropout)
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
            nn.Linear(self.collision_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.formula_count_input_dropout = nn.Dropout(config.dropout)
        self.formula_count_encoder = nn.Sequential(
            nn.Linear(FRAGMENT_FORMULA_DIM, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self._reset_formula_count_encoder()
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
        if self.fragment_action_primary_layers > 0:
            with torch.random.fork_rng(devices=[]):
                self.fragment_action_primary_input_dropout = nn.Dropout(config.dropout)
                self.fragment_action_primary_pair_scorer = (
                    self._make_fragment_action_primary_pair_scorer(config)
                )
                self.fragment_action_primary_residual = (
                    self._make_fragment_action_primary_residual(config)
                )
                self._reset_fragment_action_primary_residual()
        else:
            self.fragment_action_primary_input_dropout = None
            self.fragment_action_primary_pair_scorer = None
            self.fragment_action_primary_residual = None
        self.oos_input_dropout = nn.Dropout(config.dropout)
        self.oos_hidden_dropout = nn.Dropout(config.dropout)
        self.oos_scorer = self._make_oos_scorer(config)

    def _reset_formula_count_encoder(self) -> None:
        """
        Initialize formula-composition residual as an exact no-op.
        """
        final_layer = self.formula_count_encoder[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _formula_count_delta(
        self,
        fragments: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode normalized fragment formula composition.

        Older fragment-cache entries did not store ``formula_counts``; those are
        treated as zeros so old checkpoints and cache files remain readable.
        """
        count_features = fragments.get('formula_counts')
        num_formulas = int(fragments['features'].shape[0])
        if count_features is None:
            count_features = torch.zeros(
                num_formulas,
                FRAGMENT_FORMULA_DIM,
                dtype=dtype,
                device=device,
            )
        else:
            count_features = count_features.to(device=device, dtype=dtype)
            if count_features.shape[-1] != FRAGMENT_FORMULA_DIM:
                raise ValueError(
                    'fragment formula_counts dim mismatch: '
                    f'expected {FRAGMENT_FORMULA_DIM}, got {count_features.shape[-1]}'
                )
        return self.formula_count_encoder(
            self.formula_count_input_dropout(count_features)
        )

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
    def _fragment_action_hidden_dim(config: MiraFragConfig) -> int:
        """
        Return the small hidden width used for constrained action scorers.
        """
        return max(32, min(128, int(config.hidden_dim) // 4))

    @classmethod
    def _make_fragment_action_mlp(
        cls,
        input_dim: int,
        config: MiraFragConfig,
        *,
        output_zero: bool = False,
    ) -> nn.Module:
        """
        Build a compact MLP for local action-path scoring.
        """
        hidden_dim = cls._fragment_action_hidden_dim(config)
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(
            max(0, int(getattr(config, 'fragment_action_primary_layers', 1)) - 1)
        ):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        module = nn.Sequential(*layers)
        if output_zero and isinstance(module[-1], nn.Linear):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        return module

    @staticmethod
    def _make_fragment_action_primary_pair_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build the optional scorer for explicit local fragmentation actions.

        The first layer is lazy because node feature dimensions differ between
        encoders. Each row scores one oriented broken bond in one candidate
        fragment; multi-break candidates aggregate these scores before the
        residual calibrator maps them back into formula logits.
        """
        hidden_dim = int(config.hidden_dim)
        layers: list[nn.Module] = [
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(
            max(0, int(getattr(config, 'fragment_action_primary_layers', 1)) - 1)
        ):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    @classmethod
    def _make_fragment_action_primary_residual(
        cls, config: MiraFragConfig
    ) -> nn.Module:
        """
        Calibrate explicit bond-action evidence as a zero-initialized residual.
        """
        return cls._make_fragment_action_mlp(4, config, output_zero=True)

    def _reset_fragment_action_primary_residual(self) -> None:
        """
        Initialize the action-primary residual's output layer as an exact no-op.
        """
        if self.fragment_action_primary_residual is None:
            return
        final_layer = self.fragment_action_primary_residual[-1]
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

    def _oos_logits(self, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Score out-of-support probability with checkpoint-compatible layers.

        The linear layers stay in ``self.oos_scorer`` at the original state-dict
        keys so checkpoints trained before OOS dropout remain loadable. Dropout
        is applied around those layers without changing their parameter names.
        """
        hidden = self.oos_scorer[0](self.oos_input_dropout(metadata_features))
        hidden = self.oos_scorer[1](hidden)
        hidden = self.oos_hidden_dropout(hidden)
        logits = self.oos_scorer[2](hidden).squeeze(-1)
        return logits

    def forward(
        self,
        node_feats: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
        graph: dict[str, torch.Tensor] | None = None,
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
                'oos_logits': self._oos_logits(metadata_features),
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
                'bond_break_logits': node_feats.new_empty(0),
                'bond_break_formula_index': torch.empty(
                    0,
                    dtype=torch.long,
                    device=node_feats.device,
                ),
                'bond_break_batch': torch.empty(
                    0, dtype=torch.long, device=node_feats.device
                ),
            }

        formula_batch = fragments['formula_batch'].to(device=node_feats.device).long()
        fragment_atom_index = (
            fragments['atom_index'].to(device=node_feats.device).long()
        )
        fragment_atom_ptr = fragments['atom_ptr'].to(device=node_feats.device).long()
        fragment_atom_features = self._pool_fragment_atoms(
            node_feats,
            fragment_atom_index,
            fragment_atom_ptr,
        )
        graph_batch_device = (
            graph_batch.to(device=node_feats.device).long()
            if graph_batch is not None
            else None
        )
        molecule_atom_features = self._pool_molecule_atoms(
            node_feats,
            graph_batch_device,
            batch_size=batch_size,
        )
        fragment_descriptor = fragments['features'].to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        fragment_inputs = torch.cat(
            [
                fragment_atom_features,
                fragment_descriptor,
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
        formula_features = self.fragment_encoder(
            self.fragment_input_dropout(fragment_inputs)
        )
        formula_features = formula_features + self._formula_count_delta(
            fragments,
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        context_features = self.context_encoder(
            self.context_input_dropout(context_inputs)
        )
        collision_inputs = self._collision_energy_feature(metadata_features)[
            formula_batch
        ]
        collision_features = self.collision_encoder(
            self.collision_input_dropout(collision_inputs)
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
        if self.fragment_path_primary:
            formula_logits = self._fragment_path_log_scores(
                formula_features,
                context_features,
                collision_features,
                edge_index,
                edge_attr,
            )
        else:
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
        formula_logits = formula_logits + self._fragment_action_primary_delta(
            node_feats,
            formula_features,
            context_features,
            collision_features,
            fragment_descriptor,
            fragments,
            formula_batch,
            batch_size=batch_size,
            base_logits=formula_logits,
            graph=graph,
        )
        bond_break_logits = node_feats.new_empty(0)
        bond_break_formula_index = torch.empty(
            0, dtype=torch.long, device=node_feats.device
        )
        bond_break_batch = torch.empty(0, dtype=torch.long, device=node_feats.device)
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
            'oos_logits': self._oos_logits(metadata_features),
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
            'bond_break_logits': bond_break_logits,
            'bond_break_formula_index': bond_break_formula_index,
            'bond_break_batch': bond_break_batch,
        }

    def _fragment_bond_event_inputs(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragment_descriptor: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        *,
        include_formula: bool,
        include_fragment_descriptor: bool,
        graph: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        bond_atom_index = fragments.get('bond_atom_index')
        bond_ptr = fragments.get('bond_ptr')
        bond_features = fragments.get('bond_features')
        if bond_atom_index is None or bond_ptr is None or bond_features is None:
            return None
        bond_atom_index = bond_atom_index.to(device=node_feats.device).long()
        bond_ptr = bond_ptr.to(device=node_feats.device).long()
        bond_features = bond_features.to(
            device=node_feats.device, dtype=node_feats.dtype
        )
        num_formulas = int(formula_features.shape[0])
        if bond_ptr.numel() != num_formulas + 1:
            raise ValueError(
                'Fragment bond pointer length does not match formula count: '
                f'ptr={int(bond_ptr.numel())} formulas={num_formulas}'
            )
        if bond_atom_index.numel() == 0:
            return None
        if bond_atom_index.ndim != 2 or bond_atom_index.shape[1] != 2:
            raise ValueError('Fragment bond_atom_index must have shape [num_bonds, 2].')
        if bond_features.shape[0] != bond_atom_index.shape[0]:
            raise ValueError(
                'Fragment bond features do not match bond atom rows: '
                f'features={int(bond_features.shape[0])} '
                f'bonds={int(bond_atom_index.shape[0])}'
            )
        if bond_features.ndim != 2 or bond_features.shape[1] < BOND_BREAK_FEATURE_DIM:
            raise ValueError(
                'Fragment bond features must have shape '
                f'[num_bonds, >= {BOND_BREAK_FEATURE_DIM}].'
            )
        counts = (bond_ptr[1:] - bond_ptr[:-1]).clamp_min(0)
        formula_index = torch.repeat_interleave(
            torch.arange(num_formulas, device=node_feats.device), counts
        )
        if formula_index.numel() != bond_atom_index.shape[0]:
            raise ValueError(
                'Fragment bond pointer ranges do not match bond rows: '
                f'ptr_count={int(formula_index.numel())} '
                f'bond_count={int(bond_atom_index.shape[0])}'
            )
        inside = node_feats[bond_atom_index[:, 0]]
        outside = node_feats[bond_atom_index[:, 1]]
        pieces = [
            inside,
            outside,
            inside * outside,
            torch.abs(inside - outside),
        ]
        if include_formula:
            pieces.append(formula_features[formula_index])
        pieces.extend(
            [context_features[formula_index], collision_features[formula_index]]
        )
        if include_fragment_descriptor:
            pieces.append(fragment_descriptor[formula_index])
        pieces.append(bond_features)
        geometry_features = self._fragment_action_geometry_event_features(
            fragments,
            bond_atom_index,
            formula_index,
            graph,
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        if geometry_features.numel() > 0:
            pieces.append(geometry_features)
        return torch.cat(pieces, dim=-1), formula_index, counts

    def _fragment_action_geometry_event_features(
        self,
        fragments: dict[str, torch.Tensor],
        bond_atom_index: torch.Tensor,
        formula_index: torch.Tensor,
        graph: dict[str, torch.Tensor] | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return invariant conformer geometry features for broken-bond events."""
        if not self.bond_break_geometry_features:
            return torch.empty(bond_atom_index.shape[0], 0, dtype=dtype, device=device)
        if graph is None or 'positions' not in graph:
            raise ValueError(
                '--fragment-action-geometry-features requires graph positions.'
            )
        positions = graph['positions'].to(device=device, dtype=dtype)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('graph positions must have shape [num_atoms, 3].')
        if int(positions.shape[0]) <= int(bond_atom_index.max().item()):
            raise ValueError(
                'fragment bond atom indices exceed graph position rows: '
                f'positions={int(positions.shape[0])} '
                f'max_index={int(bond_atom_index.max().item())}'
            )

        src = bond_atom_index[:, 0].long()
        dst = bond_atom_index[:, 1].long()
        src_pos = positions[src]
        dst_pos = positions[dst]
        delta = src_pos - dst_pos
        length = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)

        atom_index = fragments['atom_index'].to(device=device).long()
        atom_ptr = fragments['atom_ptr'].to(device=device).long()
        fragment_sum = self._sum_fragment_atoms(positions, atom_index, atom_ptr)
        fragment_counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(1).to(device=device)
        fragment_centroid = fragment_sum / fragment_counts.to(dtype=dtype).unsqueeze(-1)

        formula_batch = fragments['formula_batch'].to(device=device).long()
        graph_batch = graph.get('batch')
        if graph_batch is None:
            graph_batch = torch.zeros(
                positions.shape[0], dtype=torch.long, device=device
            )
        else:
            graph_batch = graph_batch.to(device=device).long()
        batch_size = 1
        if formula_batch.numel() > 0:
            batch_size = max(batch_size, int(formula_batch.max().item()) + 1)
        if graph_batch.numel() > 0:
            batch_size = max(batch_size, int(graph_batch.max().item()) + 1)
        molecule_sum = positions.new_zeros(batch_size, 3)
        molecule_sum.index_add_(0, graph_batch, positions)
        molecule_count = positions.new_zeros(batch_size, 1)
        molecule_count.index_add_(
            0, graph_batch, positions.new_ones(positions.shape[0], 1)
        )
        molecule_centroid = molecule_sum / molecule_count.clamp_min(1.0)
        event_batch = formula_batch[formula_index]
        event_fragment_centroid = fragment_centroid[formula_index]
        event_molecule_centroid = molecule_centroid[event_batch]
        midpoint = 0.5 * (src_pos + dst_pos)

        covalent = self._covalent_radius_sum(
            graph, src, dst, dtype=dtype, device=device
        )
        ratio = length / covalent.clamp_min(1e-6)
        excess = length - covalent
        inv_length = torch.reciprocal(length)
        return torch.stack(
            [
                length / 4.0,
                inv_length,
                ratio / 2.0,
                excess / 2.0,
                torch.linalg.vector_norm(midpoint - event_molecule_centroid, dim=-1)
                / 6.0,
                torch.linalg.vector_norm(src_pos - event_fragment_centroid, dim=-1)
                / 6.0,
                torch.linalg.vector_norm(dst_pos - event_fragment_centroid, dim=-1)
                / 6.0,
                torch.linalg.vector_norm(midpoint - event_fragment_centroid, dim=-1)
                / 6.0,
            ],
            dim=-1,
        )

    @staticmethod
    def _covalent_radius_sum(
        graph: dict[str, torch.Tensor],
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        atomic_numbers = graph.get('atomic_numbers')
        if atomic_numbers is None:
            return torch.full(src.shape, 1.5, dtype=dtype, device=device)
        atomic_numbers = atomic_numbers.to(device=device).long().clamp_min(0)
        max_z = max(
            int(atomic_numbers.max().item()) if atomic_numbers.numel() else 0, 54
        )
        radii = torch.full((max_z + 1,), 0.75, dtype=dtype, device=device)
        values = {
            1: 0.31,
            5: 0.84,
            6: 0.76,
            7: 0.71,
            8: 0.66,
            9: 0.57,
            14: 1.11,
            15: 1.07,
            16: 1.05,
            17: 1.02,
            35: 1.20,
            53: 1.39,
        }
        for z, radius in values.items():
            if z < radii.shape[0]:
                radii[z] = radius
        safe_atomic_numbers = atomic_numbers.clamp_max(radii.shape[0] - 1)
        return radii[safe_atomic_numbers[src]] + radii[safe_atomic_numbers[dst]]

    def _fragment_action_primary_delta(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragment_descriptor: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
        base_logits: torch.Tensor,
        graph: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Return a latent bond-action residual for formula logits.

        Each retained formula owns zero or more oriented broken-bond actions. The
        pair scorer evaluates those local chemistry actions, multiple breaks are
        aggregated per formula, and a zero-initialized residual injects the
        standardized action evidence into the spectrum scorer.
        """
        if self.fragment_action_primary_layers <= 0 or formula_features.numel() == 0:
            return base_logits.new_zeros(base_logits.shape)
        if (
            self.fragment_action_primary_pair_scorer is None
            or self.fragment_action_primary_input_dropout is None
            or self.fragment_action_primary_residual is None
        ):
            return base_logits.new_zeros(base_logits.shape)

        event = self._fragment_bond_event_inputs(
            node_feats,
            formula_features,
            context_features,
            collision_features,
            fragment_descriptor,
            fragments,
            include_formula=True,
            include_fragment_descriptor=True,
            graph=graph,
        )
        if event is None:
            self._materialize_empty_fragment_action_primary_scorer(
                node_feats,
                formula_features,
                context_features,
                collision_features,
                fragment_descriptor,
            )
            return base_logits.new_zeros(base_logits.shape)
        pair_inputs, formula_index, counts = event
        num_formulas = int(formula_features.shape[0])
        pair_logits = self.fragment_action_primary_pair_scorer(
            self.fragment_action_primary_input_dropout(pair_inputs)
        ).squeeze(-1)
        action_sum = base_logits.new_zeros(num_formulas)
        action_sum.index_add_(0, formula_index, pair_logits)
        action_count = counts.to(device=base_logits.device, dtype=base_logits.dtype)
        action_score = action_sum / torch.sqrt(action_count.clamp_min(1.0))
        action_score = torch.where(
            action_count > 0, action_score, base_logits.new_zeros(action_score.shape)
        )
        normalized_action = self._standardize_by_batch(
            action_score,
            formula_batch,
            batch_size=batch_size,
        )
        normalized_base = self._standardize_by_batch(
            base_logits.detach(),
            formula_batch,
            batch_size=batch_size,
        )
        count_feature = torch.log1p(action_count)
        residual_inputs = torch.stack(
            [
                normalized_action,
                normalized_base,
                normalized_action - normalized_base,
                count_feature,
            ],
            dim=-1,
        )
        return self.fragment_action_primary_residual(residual_inputs).squeeze(-1)

    def _materialize_empty_fragment_action_primary_scorer(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragment_descriptor: torch.Tensor,
    ) -> None:
        """Initialize lazy action-primary layers for empty-provenance batches."""
        if self.fragment_action_primary_pair_scorer is None:
            return
        if not any(
            is_lazy(param)
            for param in self.fragment_action_primary_pair_scorer.parameters()
        ):
            return
        input_dim = (
            4 * int(node_feats.shape[-1])
            + int(formula_features.shape[-1])
            + int(context_features.shape[-1])
            + int(collision_features.shape[-1])
            + int(fragment_descriptor.shape[-1])
            + self._bond_break_feature_dim()
        )
        with torch.no_grad():
            self.fragment_action_primary_pair_scorer(node_feats.new_zeros(1, input_dim))

    def _bond_break_logits(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        graph: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Return candidate formula scores projected from explicit cut-bond logits.

        This diagnostic projection sums oriented edge-side logits over each
        multi-break candidate path. The main prediction path keeps the generic
        fragment scorer primary and exposes pair logits for an auxiliary
        FIORA-like loss.
        """
        pair_logits, formula_index = self._bond_break_event_logits(
            node_feats,
            formula_features,
            context_features,
            collision_features,
            fragments,
            graph=graph,
        )
        delta = formula_features.new_zeros(formula_features.shape[0])
        if pair_logits.numel() > 0:
            delta.index_add_(0, formula_index, pair_logits)
        return delta

    def _fragment_path_log_scores(
        self,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return root-to-fragment path logits over the retained fragment DAG.

        Root scores represent direct precursor-to-fragment evidence. Directed
        parent-child transition scores are accumulated in log space, so
        multi-break fragments receive log-summed scores over all retained paths
        leading to that fragment.
        """
        if self.fragment_path_layers <= 0 or formula_features.numel() == 0:
            return formula_features.new_zeros(formula_features.shape[0])
        if (
            self.fragment_path_edge_scorer is None
            or self.fragment_path_root_scorer is None
        ):
            return formula_features.new_zeros(formula_features.shape[0])

        root_inputs = torch.cat(
            [formula_features, context_features, collision_features], dim=-1
        )
        root_scores = self.fragment_path_root_scorer(root_inputs).squeeze(-1)
        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return root_scores

        parent_mask = edge_attr[:, 0] > 0.5
        if not bool(parent_mask.any()):
            return root_scores
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
        return torch.where(
            self._fragment_path_reachable(path_scores), path_scores, root_scores
        )

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
        return self._fragment_path_residual_logits(
            formula_features,
            context_features,
            collision_features,
            path_features,
        )

    def _fragment_path_residual_logits(
        self,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        path_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the path residual in chunks to avoid a large peak concat tensor.
        """
        if self.fragment_path_residual is None:
            return formula_features.new_zeros(formula_features.shape[0])
        chunk_size = 8192
        outputs: list[torch.Tensor] = []
        for start in range(0, int(formula_features.shape[0]), chunk_size):
            end = min(start + chunk_size, int(formula_features.shape[0]))
            formula_chunk = formula_features[start:end]
            context_chunk = context_features[start:end]
            collision_chunk = collision_features[start:end]
            path_chunk = path_features[start:end]
            residual_inputs = torch.cat(
                [
                    formula_chunk,
                    context_chunk,
                    formula_chunk * context_chunk,
                    collision_chunk,
                    formula_chunk * collision_chunk,
                    path_chunk,
                    formula_chunk * path_chunk,
                    context_chunk * path_chunk,
                ],
                dim=-1,
            )
            outputs.append(self.fragment_path_residual(residual_inputs).squeeze(-1))
        if not outputs:
            return formula_features.new_zeros(0)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _fragment_path_root_frontier(
        root_scores: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Return root scores only for fragments that are not child nodes."""
        frontier = root_scores.new_full(root_scores.shape, float('-inf'))
        if root_scores.numel() == 0:
            return frontier
        is_child = torch.zeros(
            root_scores.shape[0], dtype=torch.bool, device=root_scores.device
        )
        if dst.numel() > 0:
            is_child[dst.to(device=root_scores.device)] = True
        frontier[~is_child] = root_scores[~is_child]
        return frontier

    def _propagate_fragment_path_scores(
        self,
        root_frontier: torch.Tensor,
        edge_scores: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        """Propagate log-space path scores through retained fragment edges."""
        if num_nodes <= 0:
            return root_frontier.new_zeros(0)
        scores = root_frontier.new_full((num_nodes,), float('-inf'))
        scores[: root_frontier.numel()] = root_frontier
        if src.numel() == 0:
            return scores
        src = src.to(device=scores.device)
        dst = dst.to(device=scores.device)
        edge_scores = edge_scores.to(device=scores.device, dtype=scores.dtype)
        for _ in range(max(1, int(self.fragment_path_layers))):
            messages = scores[src] + edge_scores
            updated = scores.clone()
            updated.scatter_reduce_(0, dst, messages, reduce='amax', include_self=True)
            if torch.equal(updated, scores):
                break
            scores = updated
        return scores

    @staticmethod
    def _fragment_path_reachable(path_scores: torch.Tensor) -> torch.Tensor:
        """Return which path scores contain finite path evidence."""
        return torch.isfinite(path_scores)

    @staticmethod
    def _standardize_by_batch(
        values: torch.Tensor,
        batch: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Center and scale one scalar per candidate within each spectrum."""
        if values.numel() == 0:
            return values
        out = values.new_zeros(values.shape)
        for batch_idx in range(int(batch_size)):
            mask = batch == batch_idx
            if not bool(mask.any()):
                continue
            selected = values[mask]
            mean = selected.mean()
            scale = selected.std(unbiased=False).clamp_min(1e-6)
            out[mask] = (selected - mean) / scale
        return out

    def _bond_break_feature_dim(self) -> int:
        """Return active per-bond feature width for action-primary inputs."""
        dim = BOND_BREAK_FEATURE_DIM
        if self.bond_break_geometry_features:
            dim += FRAGMENT_ACTION_GEOMETRY_FEATURE_DIM
        return dim

    def _collision_energy_feature(
        self, metadata_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the normalized scalar collision-energy feature."""
        if metadata_features.shape[-1] <= 1:
            return metadata_features.new_zeros(metadata_features.shape[0], 1)
        return metadata_features[:, 1:2]

    @staticmethod
    def _sum_fragment_atoms(
        node_feats: torch.Tensor,
        atom_index: torch.Tensor,
        atom_ptr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum atom features for each fragment formula using atom pointer ranges.
        """
        num_fragments = max(int(atom_ptr.numel()) - 1, 0)
        pooled = node_feats.new_zeros(num_fragments, node_feats.shape[-1])
        if num_fragments == 0:
            return pooled
        counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(0)
        if atom_index.numel() > 0:
            fragment_idx = torch.repeat_interleave(
                torch.arange(num_fragments, device=node_feats.device),
                counts.to(device=node_feats.device),
            )
            if fragment_idx.numel() != atom_index.numel():
                raise ValueError(
                    'Fragment atom pointer ranges do not match atom indices: '
                    f'ptr_count={int(fragment_idx.numel())} '
                    f'atom_count={int(atom_index.numel())}'
                )
            pooled.index_add_(0, fragment_idx, node_feats[atom_index])
        return pooled

    @staticmethod
    def _sum_molecule_atoms(
        node_feats: torch.Tensor,
        graph_batch: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Sum atom features for each precursor molecule in the batch.
        """
        pooled = node_feats.new_zeros(batch_size, node_feats.shape[-1])
        if graph_batch is None or graph_batch.numel() == 0:
            if batch_size > 0:
                pooled[0] = node_feats.sum(dim=0)
            return pooled
        graph_batch = graph_batch.clamp(0, batch_size - 1)
        pooled.index_add_(0, graph_batch, node_feats)
        return pooled

    @staticmethod
    def _molecule_atom_counts(
        node_feats: torch.Tensor,
        graph_batch: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Count atoms for each precursor molecule in the batch.
        """
        counts = torch.zeros(batch_size, dtype=torch.long, device=node_feats.device)
        if graph_batch is None or graph_batch.numel() == 0:
            if batch_size > 0:
                counts[0] = int(node_feats.shape[0])
            return counts
        graph_batch = graph_batch.clamp(0, batch_size - 1)
        counts.index_add_(0, graph_batch, torch.ones_like(graph_batch))
        return counts

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
        if num_fragments == 0:
            return node_feats.new_zeros(0, node_feats.shape[-1])
        pooled = FragmentSpectrumHead._sum_fragment_atoms(
            node_feats, atom_index, atom_ptr
        )
        counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(1)
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


class BondEventMessageLayer(nn.Module):
    """
    Message-passing layer over broken-bond events grouped by candidate formula.

    Events in the same multi-break fragment exchange mean-pooled context. This
    gives the action-primary branch a way to model coordinated multi-bond
    cleavage rather than scoring each broken bond independently.
    """

    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        event_features: torch.Tensor,
        group_index: torch.Tensor,
        *,
        num_groups: int,
    ) -> torch.Tensor:
        if event_features.numel() == 0:
            return event_features
        group_index = group_index.to(device=event_features.device).long()
        group_sum = event_features.new_zeros(num_groups, event_features.shape[-1])
        group_sum.index_add_(0, group_index, event_features)
        group_count = event_features.new_zeros(num_groups, 1)
        group_count.index_add_(
            0,
            group_index,
            event_features.new_ones(event_features.shape[0], 1),
        )
        peer_sum = group_sum[group_index] - event_features
        peer_count = (group_count[group_index] - 1.0).clamp_min(1.0)
        peer_mean = peer_sum / peer_count
        update = self.message(torch.cat([event_features, peer_mean], dim=-1))
        return self.norm(event_features + self.dropout(update))


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
