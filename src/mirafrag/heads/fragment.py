from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mirafrag.config import MiraFragConfig
from mirafrag.fragments import FRAGMENT_EDGE_FEATURE_DIM


class FragmentSpectrumHead(nn.Module):
    """
    Sparse candidate-based spectrum head.

    The head pools foundation encoder atom features over fragment formulas, appends fragment features and precursor metadata, optionally runs message passing over the fragment graph, scores formulas, expands scores to isotope/adduct peak candidates, and predicts an OOS logit.
    """

    def __init__(self, config: MiraFragConfig) -> None:
        """
        Create fragment encoders, optional graph message layers, candidate scorer, and OOS scorer.
        """
        super().__init__()
        self.num_bins = int(config.num_bins)
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
        layers: list[nn.Module] = []
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
        self.oos_scorer = self._make_oos_scorer(config)

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
        features = torch.cat(
            [
                fragment_atom_features,
                fragments['features'].to(
                    device=node_feats.device,
                    dtype=node_feats.dtype,
                ),
                metadata_features[formula_batch],
            ],
            dim=-1,
        )
        formula_features = self.fragment_encoder(features)
        edge_index = fragments['edge_index'].to(device=node_feats.device)
        edge_attr = fragments['edge_attr'].to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        for layer in self.fragment_gnn_layers:
            formula_features = layer(formula_features, edge_index, edge_attr)
        formula_index = fragments['formula_index'].to(device=node_feats.device).long()
        formula_logits = self.scorer(formula_features).squeeze(-1)
        logits = formula_logits[formula_index]
        log_prior = fragments['log_prior'].to(
            device=node_feats.device,
            dtype=logits.dtype,
        )
        logits = logits + log_prior
        return {
            'kind': 'sparse',
            'logits': logits,
            'oos_logits': self.oos_scorer(metadata_features).squeeze(-1),
            'mzs': fragments['mz'].to(device=node_feats.device, dtype=node_feats.dtype),
            'bins': fragments['bin']
            .to(device=node_feats.device)
            .long()
            .clamp(0, self.num_bins - 1),
            'log_prior': log_prior,
            'formula_index': formula_index,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'formula_batch': formula_batch,
            'batch': fragment_batch,
            'batch_size': batch_size,
            'num_bins': self.num_bins,
        }

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


class FragmentGraphMessageLayer(nn.Module):
    """
    One message-passing layer over the fragment relationship graph.

    Messages are gated by edge features, averaged at each destination formula, and combined with residual MLP updates and layer normalization.
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
    ) -> torch.Tensor:
        """
        Apply one round of edge-conditioned message passing to fragment features.
        """
        if edge_index.numel() == 0:
            return node_features

        src = edge_index[0].long()
        dst = edge_index[1].long()
        messages = self.source(node_features[src]) * torch.sigmoid(
            self.edge_gate(edge_attr)
        )
        aggregated = node_features.new_zeros(node_features.shape)
        aggregated.index_add_(0, dst, messages)
        degree = node_features.new_zeros(node_features.shape[0], 1)
        degree.index_add_(0, dst, node_features.new_ones(dst.shape[0], 1))
        aggregated = aggregated / degree.clamp_min(1.0)

        hidden = self.message_norm(node_features + self.dropout(aggregated))
        return self.update_norm(hidden + self.dropout(self.update(hidden)))
