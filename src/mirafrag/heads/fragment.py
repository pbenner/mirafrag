from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import is_lazy

from mirafrag.config import MiraFragConfig
from mirafrag.data import MOLECULE_DESCRIPTOR_DIM
from mirafrag.fragments import (
    BOND_BREAK_FEATURE_DIM,
    FRAGMENT_EDGE_FEATURE_DIM,
    FRAGMENT_FEATURE_DIM,
    FRAGMENT_FORMULA_DIM,
)


def _collision_energy_embedding_mode(config: MiraFragConfig) -> str:
    """
    Return the collision-energy embedding mode, preserving the legacy basis flag.
    """
    mode = str(getattr(config, 'ce_embedding', 'scalar') or 'scalar').lower()
    legacy_basis = bool(getattr(config, 'ce_basis_features', False))
    if legacy_basis and mode == 'scalar':
        mode = 'basis'
    if mode not in {'scalar', 'basis', 'fourier'}:
        raise ValueError(
            f"Unsupported ce_embedding={mode!r}; expected 'scalar', 'basis', or 'fourier'."
        )
    return mode


def _collision_energy_feature_dim(mode: str, fourier_frequencies: int) -> int:
    """
    Return the number of channels produced by the CE feature transform.
    """
    if mode == 'scalar':
        return 1
    if mode == 'basis':
        return 7
    if mode == 'fourier':
        return 1 + 2 * max(1, int(fourier_frequencies))
    raise ValueError(f'Unsupported collision-energy embedding mode: {mode!r}')


def _bond_break_geometry_features_enabled(config: MiraFragConfig) -> bool:
    """Return the effective bond-event geometry setting, including the legacy flag."""
    return bool(
        getattr(config, 'bond_break_geometry_features', False)
        or getattr(config, 'fragment_action_geometry_features', False)
    )


AIMNET_CHARGE_FEATURE_DIM = 6
AIMNET_CHARGE_SUMMARY_DIM = 5 * AIMNET_CHARGE_FEATURE_DIM
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
        self.fragment_action_path_layers = max(
            0, int(getattr(config, 'fragment_action_path_layers', 0))
        )
        self.fragment_action_primary_layers = max(
            0, int(getattr(config, 'fragment_action_primary_layers', 0))
        )
        self.fragment_action_primary_ce_gate_enabled = bool(
            getattr(config, 'fragment_action_primary_ce_gate', False)
        )
        self.bond_break_geometry_features = _bond_break_geometry_features_enabled(
            config
        )
        self.fragment_action_geometry_features = self.bond_break_geometry_features
        self.fragment_action_bond_gnn_layers = max(
            0, int(getattr(config, 'fragment_action_bond_gnn_layers', 0))
        )
        self.fragment_bond_break_layers = max(
            0, int(getattr(config, 'fragment_bond_break_layers', 0))
        )
        self.candidate_transformer_layers = max(
            0, int(getattr(config, 'candidate_transformer_layers', 0))
        )
        self.candidate_transformer_heads = max(
            1, int(getattr(config, 'candidate_transformer_heads', 8))
        )
        self.candidate_transformer_max_tokens = max(
            1, int(getattr(config, 'candidate_transformer_max_tokens', 256))
        )
        self.conditional_expert_heads = max(
            1, int(getattr(config, 'conditional_expert_heads', 1))
        )
        self.conditional_expert_hidden_dim = max(
            1, int(getattr(config, 'conditional_expert_hidden_dim', 128))
        )
        self.conditional_expert_dropout = max(
            0.0, float(getattr(config, 'conditional_expert_dropout', 0.0))
        )
        self.candidate_suppression_gate_hidden_dim = max(
            0, int(getattr(config, 'candidate_suppression_gate_hidden_dim', 0))
        )
        self.candidate_suppression_gate_dropout = max(
            0.0, float(getattr(config, 'candidate_suppression_gate_dropout', 0.0))
        )
        self.candidate_suppression_gate_initial_penalty = max(
            1e-12,
            float(getattr(config, 'candidate_suppression_gate_initial_penalty', 1e-6)),
        )
        self.molecule_descriptor_features = bool(
            getattr(config, 'molecule_descriptor_features', False)
        )
        self.molecule_descriptor_hidden_dim = max(
            1, int(getattr(config, 'molecule_descriptor_hidden_dim', 128))
        )
        self.molecule_descriptor_dropout = max(
            0.0, float(getattr(config, 'molecule_descriptor_dropout', 0.0))
        )
        self.fragnnet_dag_num_layers = max(
            0, int(getattr(config, 'fragnnet_dag_layers', 0))
        )
        self.fragnnet_dag_num_hs = max(
            0, int(getattr(config, 'fragnnet_dag_num_hs', 4))
        )
        self.ce_embedding = _collision_energy_embedding_mode(config)
        self.ce_basis_features = self.ce_embedding == 'basis'
        self.ce_fourier_frequencies = max(
            1, int(getattr(config, 'ce_fourier_frequencies', 8))
        )
        self.aimnet_charge_features = bool(
            getattr(config, 'aimnet_charge_features', False)
        )
        self.collision_feature_dim = _collision_energy_feature_dim(
            self.ce_embedding,
            self.ce_fourier_frequencies,
        )
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
        if self.fragnnet_dag_num_layers > 0:
            self.fragnnet_dag_input_dropout = nn.Dropout(config.dropout)
            self.fragnnet_dag_node_encoder = nn.Sequential(
                nn.LazyLinear(config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            self.fragnnet_dag_message_layers = nn.ModuleList(
                [
                    FragmentGraphMessageLayer(
                        hidden_dim=config.hidden_dim,
                        edge_dim=FRAGMENT_EDGE_FEATURE_DIM,
                        dropout=config.dropout,
                    )
                    for _ in range(self.fragnnet_dag_num_layers)
                ]
            )
            self.fragnnet_dag_node_scorer = self._make_fragnnet_dag_scorer(config)
            self.fragnnet_dag_formula_scorer = self._make_fragnnet_dag_variant_scorer(
                config,
                num_hs=self.fragnnet_dag_num_hs,
            )
            self.fragnnet_dag_null_scorer = self._make_fragnnet_dag_scorer(config)
        else:
            self.fragnnet_dag_input_dropout = None
            self.fragnnet_dag_node_encoder = None
            self.fragnnet_dag_message_layers = nn.ModuleList()
            self.fragnnet_dag_node_scorer = None
            self.fragnnet_dag_formula_scorer = None
            self.fragnnet_dag_null_scorer = None
        if self.aimnet_charge_features:
            self.aimnet_charge_input_dropout = nn.Dropout(config.dropout)
            self.aimnet_charge_residual = nn.Sequential(
                nn.Linear(AIMNET_CHARGE_SUMMARY_DIM, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self._reset_aimnet_charge_residual()
        else:
            self.aimnet_charge_input_dropout = None
            self.aimnet_charge_residual = None
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
        if self.candidate_transformer_layers > 0:
            self.candidate_transformer_input = nn.Sequential(
                nn.LazyLinear(config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            if config.hidden_dim % self.candidate_transformer_heads != 0:
                raise ValueError(
                    '--candidate-transformer-heads must divide --hidden-dim.'
                )
            candidate_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=self.candidate_transformer_heads,
                dim_feedforward=2 * config.hidden_dim,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.candidate_transformer = nn.TransformerEncoder(
                candidate_layer,
                num_layers=self.candidate_transformer_layers,
                enable_nested_tensor=False,
            )
            self.candidate_transformer_out = nn.Linear(config.hidden_dim, 1)
            self._reset_candidate_transformer_out()
        else:
            self.candidate_transformer_input = None
            self.candidate_transformer = None
            self.candidate_transformer_out = None
        if self.conditional_expert_heads > 1:
            self.conditional_expert_input_dropout = nn.Dropout(
                self.conditional_expert_dropout
            )
            self.conditional_expert_scorer = self._make_conditional_expert_scorer(
                config
            )
            self.conditional_expert_gate = self._make_conditional_expert_gate(config)
            self._reset_conditional_expert_scorer()
        else:
            self.conditional_expert_input_dropout = None
            self.conditional_expert_scorer = None
            self.conditional_expert_gate = None
        if self.candidate_suppression_gate_hidden_dim > 0:
            self.candidate_suppression_gate_input_dropout = nn.Dropout(
                self.candidate_suppression_gate_dropout
            )
            self.candidate_suppression_gate = self._make_candidate_suppression_gate(
                config
            )
            self._reset_candidate_suppression_gate()
        else:
            self.candidate_suppression_gate_input_dropout = None
            self.candidate_suppression_gate = None
        if self.molecule_descriptor_features:
            self.molecule_descriptor_input_dropout = nn.Dropout(
                self.molecule_descriptor_dropout
            )
            self.molecule_descriptor_formula_scorer = (
                self._make_molecule_descriptor_formula_scorer(config)
            )
            self.molecule_descriptor_oos_scorer = (
                self._make_molecule_descriptor_oos_scorer(config)
            )
            self._reset_molecule_descriptor_scorers()
        else:
            self.molecule_descriptor_input_dropout = None
            self.molecule_descriptor_formula_scorer = None
            self.molecule_descriptor_oos_scorer = None
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
        if self.fragment_action_path_layers > 0:
            with torch.random.fork_rng(devices=[]):
                self.fragment_action_root_scorer = (
                    self._make_fragment_action_root_scorer(config)
                )
                self.fragment_action_edge_scorer = (
                    self._make_fragment_action_edge_scorer(config)
                )
                self.fragment_action_residual = self._make_fragment_action_residual(
                    config
                )
                self._reset_fragment_action_residual()
        else:
            self.fragment_action_root_scorer = None
            self.fragment_action_edge_scorer = None
            self.fragment_action_residual = None
        if self.fragment_action_primary_layers > 0:
            with torch.random.fork_rng(devices=[]):
                self.fragment_action_primary_input_dropout = nn.Dropout(config.dropout)
                self.fragment_action_primary_pair_scorer = (
                    self._make_fragment_action_primary_pair_scorer(config)
                )
                self.fragment_action_primary_residual = (
                    self._make_fragment_action_primary_residual(config)
                )
                if self.fragment_action_primary_ce_gate_enabled:
                    self.fragment_action_primary_ce_gate = (
                        self._make_fragment_action_primary_ce_gate(config)
                    )
                    self._reset_fragment_action_primary_ce_gate()
                else:
                    self.fragment_action_primary_ce_gate = None
                self._reset_fragment_action_primary_residual()
        else:
            self.fragment_action_primary_input_dropout = None
            self.fragment_action_primary_pair_scorer = None
            self.fragment_action_primary_residual = None
            self.fragment_action_primary_ce_gate = None
        if self.fragment_action_bond_gnn_layers > 0:
            with torch.random.fork_rng(devices=[]):
                self.fragment_action_bond_gnn_input_dropout = nn.Dropout(config.dropout)
                self.fragment_action_bond_gnn_encoder = (
                    self._make_fragment_action_bond_gnn_encoder(config)
                )
                self.fragment_action_bond_gnn_layers_module = nn.ModuleList(
                    [
                        BondEventMessageLayer(
                            hidden_dim=config.hidden_dim,
                            dropout=config.dropout,
                        )
                        for _ in range(self.fragment_action_bond_gnn_layers)
                    ]
                )
                self.fragment_action_bond_gnn_scorer = nn.Linear(config.hidden_dim, 1)
                self.fragment_action_bond_gnn_residual = (
                    self._make_fragment_action_primary_residual(config)
                )
                self._reset_fragment_action_bond_gnn_residual()
        else:
            self.fragment_action_bond_gnn_input_dropout = None
            self.fragment_action_bond_gnn_encoder = None
            self.fragment_action_bond_gnn_layers_module = nn.ModuleList()
            self.fragment_action_bond_gnn_scorer = None
            self.fragment_action_bond_gnn_residual = None
        if self.fragment_bond_break_layers > 0:
            with torch.random.fork_rng(devices=[]):
                self.bond_break_input_dropout = nn.Dropout(config.dropout)
                self.bond_break_pair_scorer = self._make_bond_break_pair_scorer(config)
                self._reset_bond_break_pair_scorer()
        else:
            self.bond_break_input_dropout = None
            self.bond_break_pair_scorer = None
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
    def _make_fragnnet_dag_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build one branch of the FraGNNet-style node/formula scorer.
        """
        hidden_dim = int(config.hidden_dim)
        mlp_hidden_dim = max(32, min(128, hidden_dim // 4))
        return nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    @staticmethod
    def _make_fragnnet_dag_variant_scorer(
        config: MiraFragConfig, *, num_hs: int
    ) -> nn.Module:
        """Build the fixed H-shift conditional formula scorer."""
        hidden_dim = int(config.hidden_dim)
        mlp_hidden_dim = max(32, min(128, hidden_dim // 4))
        return nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, 2 * int(num_hs) + 1),
        )

    def _fragnnet_dag_formula_logits(
        self,
        node_atom_features: torch.Tensor,
        node_atom_sums: torch.Tensor,
        molecule_atom_features: torch.Tensor,
        molecule_atom_sums: torch.Tensor,
        node_descriptor: torch.Tensor,
        node_count_features: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
        node_edge_index: torch.Tensor,
        node_edge_attr: torch.Tensor,
        collision_features: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Factorized FraGNNet-style scoring over structural nodes and variants.

        Structural fragment nodes are scored once. Formula/H candidates attached
        to the same node receive a conditional log-probability, normalized over
        that node's retained variants plus a learned null/no-emission option.
        The returned formula logits are ``node_logit + log P(variant | node)``.
        """
        if (
            self.fragnnet_dag_node_encoder is None
            or self.fragnnet_dag_node_scorer is None
            or self.fragnnet_dag_formula_scorer is None
            or self.fragnnet_dag_null_scorer is None
        ):
            raise RuntimeError('FraGNNet DAG scorer is not enabled.')
        if not bool(fragments.get('has_fragment_nodes', torch.tensor(False)).item()):
            raise ValueError(
                'FraGNNet DAG scorer requires fragment caches with structural '
                'node fields. Rebuild the fragment cache for this experiment.'
            )
        formula_node_index = fragments.get('formula_node_index')
        formula_h_shift = fragments.get('formula_h_shift')
        node_batch = fragments.get('node_batch')
        if formula_node_index is None or formula_h_shift is None or node_batch is None:
            raise ValueError(
                'FraGNNet DAG scorer requires fragment caches with structural '
                'node fields. Rebuild the fragment cache for this experiment.'
            )
        formula_node_index = formula_node_index.to(device=device).long()
        formula_h_shift = formula_h_shift.to(device=device).long()
        node_batch = node_batch.to(device=device).long()
        molecule_context = molecule_atom_features[node_batch]
        molecule_sum_context = molecule_atom_sums[node_batch]
        connected_component_mean = node_atom_features - molecule_context
        connected_component_sum = node_atom_sums - molecule_sum_context
        node_inputs = torch.cat(
            [
                connected_component_mean,
                connected_component_sum,
                molecule_context,
                molecule_sum_context,
                node_descriptor,
                node_count_features,
                metadata_features[node_batch],
            ],
            dim=-1,
        )
        node_state = self.fragnnet_dag_node_encoder(
            self.fragnnet_dag_input_dropout(node_inputs)
        )
        node_collision_features = collision_features.new_zeros(
            node_state.shape[0], collision_features.shape[-1]
        )
        if formula_node_index.numel() > 0:
            node_collision_features.index_add_(
                0,
                formula_node_index,
                collision_features,
            )
            node_counts = node_collision_features.new_zeros(node_state.shape[0])
            node_counts.index_add_(
                0,
                formula_node_index,
                node_counts.new_ones(formula_node_index.shape[0]),
            )
            node_collision_features = node_collision_features / node_counts.clamp_min(
                1.0
            ).unsqueeze(-1)
        for layer in self.fragnnet_dag_message_layers:
            node_state = layer(
                node_state, node_edge_index, node_edge_attr, node_collision_features
            )
        node_logits = self.fragnnet_dag_node_scorer(node_state).squeeze(-1)
        node_log_normalizer = self._scatter_logsumexp(
            node_logits,
            node_batch,
            size=int(metadata_features.shape[0]),
        )
        node_logprobs = node_logits - node_log_normalizer[node_batch]
        slot_logits = self.fragnnet_dag_formula_scorer(node_state)
        null_logits = self.fragnnet_dag_null_scorer(node_state).squeeze(-1)
        slot_log_normalizer = torch.logaddexp(
            torch.logsumexp(slot_logits, dim=1),
            null_logits,
        )
        slot = formula_h_shift + self.fragnnet_dag_num_hs
        valid = (slot >= 0) & (slot < slot_logits.shape[1])
        safe_slot = slot.clamp(0, slot_logits.shape[1] - 1)
        selected_slot_logits = slot_logits[formula_node_index, safe_slot]
        formula_logits = (
            node_logprobs[formula_node_index]
            + selected_slot_logits
            - slot_log_normalizer[formula_node_index]
        )
        return torch.where(
            valid,
            formula_logits,
            formula_logits.new_full(formula_logits.shape, -1e9),
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
            max(0, int(getattr(config, 'fragment_action_path_layers', 1)) - 1)
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

    @classmethod
    def _make_fragment_action_root_scorer(cls, config: MiraFragConfig) -> nn.Module:
        """
        Score direct precursor-to-fragment actions from local descriptors.
        """
        return cls._make_fragment_action_mlp(
            FRAGMENT_FEATURE_DIM + 1,
            config,
        )

    @classmethod
    def _make_fragment_action_edge_scorer(cls, config: MiraFragConfig) -> nn.Module:
        """
        Score parent-to-child fragmentation actions from local descriptors.

        The inputs are deliberately constrained to hand-crafted parent/child
        fragment descriptors, their descriptor delta, the typed retained edge
        features, and the normalized collision-energy scalar.
        """
        return cls._make_fragment_action_mlp(
            3 * FRAGMENT_FEATURE_DIM + FRAGMENT_EDGE_FEATURE_DIM + 1,
            config,
        )

    @classmethod
    def _make_fragment_action_residual(cls, config: MiraFragConfig) -> nn.Module:
        """
        Calibrate action-path evidence as a zero-initialized residual.
        """
        return cls._make_fragment_action_mlp(3, config, output_zero=True)

    def _reset_fragment_action_residual(self) -> None:
        """
        Initialize the action-path residual's output layer as an exact no-op.
        """
        if self.fragment_action_residual is None:
            return
        final_layer = self.fragment_action_residual[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

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

    @staticmethod
    def _make_fragment_action_bond_gnn_encoder(config: MiraFragConfig) -> nn.Module:
        hidden_dim = int(config.hidden_dim)
        return nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _make_fragment_action_primary_ce_gate(config: MiraFragConfig) -> nn.Module:
        """Build a zero-initialized CE gate for action-primary evidence."""
        hidden_dim = int(config.hidden_dim)
        gate_hidden_dim = max(16, hidden_dim // 4)
        return nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def _reset_fragment_action_primary_ce_gate(self) -> None:
        """Initialize the CE gate so it multiplies action evidence by one."""
        gate = getattr(self, 'fragment_action_primary_ce_gate', None)
        if gate is None:
            return
        final_layer = gate[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

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

    def _reset_fragment_action_bond_gnn_residual(self) -> None:
        """Initialize the bond-event GNN residual as an exact no-op."""
        if self.fragment_action_bond_gnn_residual is None:
            return
        final_layer = self.fragment_action_bond_gnn_residual[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    @staticmethod
    def _make_bond_break_pair_scorer(config: MiraFragConfig) -> nn.Module:
        """
        Build the optional scorer for oriented fragment boundary bonds.

        The first layer is lazy because encoder node-feature dimensions differ
        across AIMNet/MACE. This scorer is supervised by a FIORA-like auxiliary
        loss and can be projected through the existing multi-break fragment
        candidates without changing the fragment cache format.
        """
        hidden_dim = int(config.hidden_dim)
        layers: list[nn.Module] = [
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(
            max(0, int(getattr(config, 'fragment_bond_break_layers', 1)) - 1)
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

    def _reset_bond_break_pair_scorer(self) -> None:
        """
        Initialize the auxiliary bond-break branch as a neutral no-op.
        """
        if self.bond_break_pair_scorer is None:
            return
        final_layer = self.bond_break_pair_scorer[-1]
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

    def _oos_logits(
        self,
        metadata_features: torch.Tensor,
        molecule_descriptors: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        return logits + self._molecule_descriptor_oos_delta(
            metadata_features, molecule_descriptors
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
        graph_batch: torch.Tensor | None = None,
        aux_node_features: dict[str, torch.Tensor] | None = None,
        molecule_descriptors: torch.Tensor | None = None,
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
                'oos_logits': self._oos_logits(metadata_features, molecule_descriptors),
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
        node_atom_index = (
            fragments.get('node_atom_index', fragment_atom_index)
            .to(device=node_feats.device)
            .long()
        )
        node_atom_ptr = (
            fragments.get('node_atom_ptr', fragment_atom_ptr)
            .to(device=node_feats.device)
            .long()
        )
        node_atom_sums = self._sum_fragment_atoms(
            node_feats,
            node_atom_index,
            node_atom_ptr,
        )
        node_atom_features = self._pool_fragment_atoms(
            node_feats,
            node_atom_index,
            node_atom_ptr,
        )
        graph_batch_device = (
            graph_batch.to(device=node_feats.device).long()
            if graph_batch is not None
            else None
        )
        molecule_atom_sums = self._sum_molecule_atoms(
            node_feats,
            graph_batch_device,
            batch_size=batch_size,
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
        node_descriptor = fragments.get('node_features', fragment_descriptor).to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        node_count_features = fragments.get(
            'node_formula_counts',
            torch.zeros(
                node_descriptor.shape[0],
                FRAGMENT_FORMULA_DIM,
                dtype=node_feats.dtype,
                device=node_feats.device,
            ),
        ).to(device=node_feats.device, dtype=node_feats.dtype)
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
        if self.aimnet_charge_features:
            formula_features = formula_features + self._aimnet_charge_delta(
                aux_node_features,
                fragments,
                graph_batch.to(device=node_feats.device).long()
                if graph_batch is not None
                else None,
                formula_batch,
                batch_size=batch_size,
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
        candidate_transformer_features = torch.cat(
            [
                formula_features,
                context_features,
                formula_features * context_features,
                collision_features,
                formula_features * collision_features,
            ],
            dim=-1,
        )
        if self.fragnnet_dag_num_layers > 0:
            node_edge_index = fragments.get('node_edge_index', edge_index).to(
                device=node_feats.device
            )
            node_edge_attr = fragments.get('node_edge_attr', edge_attr).to(
                device=node_feats.device,
                dtype=node_feats.dtype,
            )
            formula_logits = self._fragnnet_dag_formula_logits(
                node_atom_features,
                node_atom_sums,
                molecule_atom_features,
                molecule_atom_sums,
                node_descriptor,
                node_count_features,
                fragments,
                metadata_features,
                node_edge_index,
                node_edge_attr,
                collision_features,
                dtype=node_feats.dtype,
                device=node_feats.device,
            )
        elif self.fragment_path_primary:
            formula_logits = self._fragment_path_log_scores(
                formula_features,
                context_features,
                collision_features,
                edge_index,
                edge_attr,
            )
        else:
            formula_logits = self.scorer(candidate_transformer_features).squeeze(-1)
            formula_logits = formula_logits + self._fragment_path_delta(
                formula_features,
                context_features,
                collision_features,
                edge_index,
                edge_attr,
                formula_batch,
                batch_size=batch_size,
            )
            formula_logits = formula_logits + self._fragment_action_path_delta(
                fragment_descriptor,
                collision_inputs,
                edge_index,
                edge_attr,
                formula_batch,
                batch_size=batch_size,
                base_logits=formula_logits,
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
        formula_logits = formula_logits + self._fragment_action_bond_gnn_delta(
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
        formula_logits = formula_logits + self._candidate_transformer_delta(
            candidate_transformer_features,
            formula_logits,
            formula_batch,
            batch_size=batch_size,
        )
        formula_logits = formula_logits + self._conditional_expert_delta(
            candidate_transformer_features,
            formula_logits,
            formula_batch,
            molecule_atom_features,
            metadata_features,
            batch_size=batch_size,
        )
        formula_logits = formula_logits + self._molecule_descriptor_formula_delta(
            formula_features,
            context_features,
            collision_features,
            formula_logits,
            formula_batch,
            molecule_descriptors,
        )
        formula_logits = formula_logits - self._candidate_suppression_penalty(
            candidate_transformer_features,
            fragment_descriptor,
            formula_logits,
            formula_batch,
            batch_size=batch_size,
        )
        bond_break_logits, bond_break_formula_index = self._bond_break_event_logits(
            node_feats,
            formula_features,
            context_features,
            collision_features,
            fragments,
            graph=graph,
        )
        bond_break_batch = (
            formula_batch[bond_break_formula_index]
            if bond_break_formula_index.numel() > 0
            else formula_batch.new_empty(0)
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
            'oos_logits': self._oos_logits(metadata_features, molecule_descriptors),
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

    def _make_molecule_descriptor_formula_scorer(
        self, config: MiraFragConfig
    ) -> nn.Module:
        hidden_dim = int(self.molecule_descriptor_hidden_dim)
        return nn.Sequential(
            nn.Linear(
                3 * int(config.hidden_dim) + MOLECULE_DESCRIPTOR_DIM + 1, hidden_dim
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.molecule_descriptor_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _make_molecule_descriptor_oos_scorer(self, config: MiraFragConfig) -> nn.Module:
        hidden_dim = int(self.molecule_descriptor_hidden_dim)
        metadata_feature_dim = 2 + 2 * int(config.metadata_dim)
        return nn.Sequential(
            nn.Linear(metadata_feature_dim + MOLECULE_DESCRIPTOR_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.molecule_descriptor_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _reset_molecule_descriptor_scorers(self) -> None:
        for scorer in (
            self.molecule_descriptor_formula_scorer,
            self.molecule_descriptor_oos_scorer,
        ):
            if scorer is None:
                continue
            final_layer = scorer[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

    def _molecule_descriptor_formula_delta(
        self,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        formula_logits: torch.Tensor,
        formula_batch: torch.Tensor,
        molecule_descriptors: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not self.molecule_descriptor_features
            or self.molecule_descriptor_formula_scorer is None
            or molecule_descriptors is None
            or formula_logits.numel() == 0
        ):
            return formula_logits.new_zeros(formula_logits.shape)
        descriptors = molecule_descriptors.to(
            device=formula_logits.device,
            dtype=formula_logits.dtype,
        )
        if descriptors.ndim != 2 or descriptors.shape[-1] != MOLECULE_DESCRIPTOR_DIM:
            raise ValueError(
                'molecule_descriptors must have shape '
                f'[batch, {MOLECULE_DESCRIPTOR_DIM}].'
            )
        inputs = torch.cat(
            [
                formula_features,
                context_features,
                collision_features,
                descriptors[formula_batch.long()],
                formula_logits.detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.molecule_descriptor_input_dropout is not None:
            inputs = self.molecule_descriptor_input_dropout(inputs)
        return self.molecule_descriptor_formula_scorer(inputs).squeeze(-1)

    def _molecule_descriptor_oos_delta(
        self,
        metadata_features: torch.Tensor,
        molecule_descriptors: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not self.molecule_descriptor_features
            or self.molecule_descriptor_oos_scorer is None
            or molecule_descriptors is None
        ):
            return metadata_features.new_zeros(metadata_features.shape[0])
        descriptors = molecule_descriptors.to(
            device=metadata_features.device,
            dtype=metadata_features.dtype,
        )
        if descriptors.ndim != 2 or descriptors.shape[-1] != MOLECULE_DESCRIPTOR_DIM:
            raise ValueError(
                'molecule_descriptors must have shape '
                f'[batch, {MOLECULE_DESCRIPTOR_DIM}].'
            )
        inputs = torch.cat([metadata_features, descriptors], dim=-1)
        if self.molecule_descriptor_input_dropout is not None:
            inputs = self.molecule_descriptor_input_dropout(inputs)
        return self.molecule_descriptor_oos_scorer(inputs).squeeze(-1)

    def _make_candidate_suppression_gate(self, config: MiraFragConfig) -> nn.Module:
        hidden_dim = int(self.candidate_suppression_gate_hidden_dim)
        return nn.Sequential(
            nn.Linear(
                5 * int(config.hidden_dim) + FRAGMENT_FEATURE_DIM + 5, hidden_dim
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.candidate_suppression_gate_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _reset_candidate_suppression_gate(self) -> None:
        if self.candidate_suppression_gate is None:
            return
        final = self.candidate_suppression_gate[-1]
        if not isinstance(final, nn.Linear):
            return
        nn.init.zeros_(final.weight)
        bias = torch.log(
            torch.expm1(torch.tensor(self.candidate_suppression_gate_initial_penalty))
        )
        nn.init.constant_(final.bias, float(bias))

    def _candidate_suppression_penalty(
        self,
        candidate_features: torch.Tensor,
        fragment_descriptor: torch.Tensor,
        formula_logits: torch.Tensor,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if (
            self.candidate_suppression_gate_hidden_dim <= 0
            or self.candidate_suppression_gate is None
            or candidate_features.numel() == 0
        ):
            return formula_logits.new_zeros(formula_logits.shape)

        detached_logits = formula_logits.detach()
        formula_batch = formula_batch.long()
        counts = formula_logits.new_zeros(batch_size)
        counts.index_add_(
            0, formula_batch, formula_logits.new_ones(formula_logits.shape)
        )
        safe_counts = counts.clamp_min(1.0)

        sums = formula_logits.new_zeros(batch_size)
        sums.index_add_(0, formula_batch, detached_logits)
        means = sums / safe_counts
        centered = detached_logits - means[formula_batch]
        variances = formula_logits.new_zeros(batch_size)
        variances.index_add_(0, formula_batch, centered.square())
        stds = torch.sqrt(variances / safe_counts)
        max_logits = self._scatter_max(detached_logits, formula_batch, size=batch_size)
        normalized_logits = centered / stds[formula_batch].clamp_min(1e-6)
        rank_quantile = self._candidate_rank_quantile(
            detached_logits, formula_batch, batch_size=batch_size
        )
        support_stats = torch.stack(
            [
                torch.log1p(counts[formula_batch]) / 8.0,
                normalized_logits,
                detached_logits - max_logits[formula_batch],
                rank_quantile,
                stds[formula_batch],
            ],
            dim=-1,
        )
        inputs = torch.cat(
            [candidate_features, fragment_descriptor, support_stats],
            dim=-1,
        )
        if self.candidate_suppression_gate_input_dropout is not None:
            inputs = self.candidate_suppression_gate_input_dropout(inputs)
        raw_penalty = F.softplus(self.candidate_suppression_gate(inputs).squeeze(-1))
        penalty = raw_penalty - float(self.candidate_suppression_gate_initial_penalty)
        return penalty.clamp_min(0.0)

    @staticmethod
    def _candidate_rank_quantile(
        logits: torch.Tensor,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        out = logits.new_zeros(logits.shape)
        for sample_idx in range(int(batch_size)):
            idx = torch.nonzero(formula_batch == sample_idx, as_tuple=False).flatten()
            if idx.numel() <= 1:
                continue
            order = torch.argsort(logits[idx].detach(), descending=True)
            ranks = torch.empty_like(order, dtype=logits.dtype)
            ranks[order] = torch.arange(
                order.numel(), device=logits.device, dtype=logits.dtype
            )
            out[idx] = ranks / float(max(int(order.numel()) - 1, 1))
        return out

    def _reset_candidate_transformer_out(self) -> None:
        """Initialize the candidate-set residual as an exact no-op."""
        if self.candidate_transformer_out is None:
            return
        nn.init.zeros_(self.candidate_transformer_out.weight)
        if self.candidate_transformer_out.bias is not None:
            nn.init.zeros_(self.candidate_transformer_out.bias)

    def _make_conditional_expert_scorer(self, config: MiraFragConfig) -> nn.Module:
        hidden_dim = int(self.conditional_expert_hidden_dim)
        return nn.Sequential(
            nn.Linear(5 * int(config.hidden_dim) + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.conditional_expert_dropout),
            nn.Linear(hidden_dim, int(self.conditional_expert_heads)),
        )

    def _make_conditional_expert_gate(self, config: MiraFragConfig) -> nn.Module:
        hidden_dim = int(self.conditional_expert_hidden_dim)
        metadata_feature_dim = 2 + 2 * int(config.metadata_dim)
        gate_input_dim = int(config.hidden_dim) + metadata_feature_dim + 4
        return nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.conditional_expert_dropout),
            nn.Linear(hidden_dim, int(self.conditional_expert_heads)),
        )

    def _reset_conditional_expert_scorer(self) -> None:
        if self.conditional_expert_scorer is None:
            return
        final_layer = self.conditional_expert_scorer[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def _conditional_expert_delta(
        self,
        candidate_features: torch.Tensor,
        formula_logits: torch.Tensor,
        formula_batch: torch.Tensor,
        molecule_atom_features: torch.Tensor,
        metadata_features: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Add a gated mixture of no-op initialized residual formula scorers."""
        if (
            self.conditional_expert_heads <= 1
            or self.conditional_expert_scorer is None
            or self.conditional_expert_gate is None
            or formula_logits.numel() == 0
        ):
            return formula_logits.new_zeros(formula_logits.shape)

        formula_batch = formula_batch.long()
        counts = formula_logits.new_zeros(batch_size)
        ones = formula_logits.new_ones(formula_logits.shape[0])
        counts.index_add_(0, formula_batch, ones)
        safe_counts = counts.clamp_min(1.0)

        detached_logits = formula_logits.detach()
        sums = formula_logits.new_zeros(batch_size)
        sums.index_add_(0, formula_batch, detached_logits)
        means = sums / safe_counts
        centered = detached_logits - means[formula_batch]
        variances = formula_logits.new_zeros(batch_size)
        variances.index_add_(0, formula_batch, centered.square())
        stds = torch.sqrt(variances / safe_counts)
        max_logits = self._scatter_max(detached_logits, formula_batch, size=batch_size)

        support_stats = torch.stack(
            [torch.log1p(counts) / 8.0, means, stds, max_logits],
            dim=-1,
        )
        support_stats = torch.nan_to_num(support_stats, nan=0.0, posinf=0.0, neginf=0.0)
        gate_inputs = torch.cat(
            [molecule_atom_features, metadata_features, support_stats],
            dim=-1,
        )
        if self.conditional_expert_input_dropout is not None:
            gate_inputs = self.conditional_expert_input_dropout(gate_inputs)
        gate_weights = torch.softmax(self.conditional_expert_gate(gate_inputs), dim=-1)

        expert_inputs = torch.cat(
            [candidate_features, detached_logits.unsqueeze(-1)],
            dim=-1,
        )
        if self.conditional_expert_input_dropout is not None:
            expert_inputs = self.conditional_expert_input_dropout(expert_inputs)
        expert_deltas = self.conditional_expert_scorer(expert_inputs)
        return (expert_deltas * gate_weights[formula_batch]).sum(dim=-1)

    @staticmethod
    def _scatter_max(
        values: torch.Tensor,
        index: torch.Tensor,
        *,
        size: int,
    ) -> torch.Tensor:
        out = values.new_full((size,), -torch.inf)
        if values.numel() == 0:
            return out.fill_(0.0)
        if hasattr(out, 'scatter_reduce_'):
            out.scatter_reduce_(
                0, index.long(), values, reduce='amax', include_self=True
            )
        else:
            for row, value in zip(index.long(), values, strict=True):
                out[row] = torch.maximum(out[row], value)
        return torch.where(torch.isfinite(out), out, out.new_zeros(out.shape))

    def _candidate_transformer_delta(
        self,
        candidate_features: torch.Tensor,
        formula_logits: torch.Tensor,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Add a residual from self-attention over top-scoring candidates per spectrum.

        Full attention over every retained fragment can be prohibitively expensive.
        The residual therefore attends only over the current top-k candidates within
        each molecule, where global competition is most likely to affect the final
        spectrum. The output projection is zero-initialized, so enabling this branch
        preserves checkpoint predictions before training.
        """
        if (
            self.candidate_transformer_layers <= 0
            or self.candidate_transformer_input is None
            or self.candidate_transformer is None
            or self.candidate_transformer_out is None
            or candidate_features.numel() == 0
        ):
            return formula_logits.new_zeros(formula_logits.shape)
        token_inputs = torch.cat(
            [candidate_features, formula_logits.unsqueeze(-1)], dim=-1
        )
        token_state = self.candidate_transformer_input(token_inputs)
        delta = formula_logits.new_zeros(formula_logits.shape)
        max_tokens = int(self.candidate_transformer_max_tokens)
        for sample_idx in range(int(batch_size)):
            idx = torch.nonzero(formula_batch == sample_idx, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() > max_tokens:
                top_local = torch.topk(
                    formula_logits[idx].detach(), k=max_tokens, sorted=False
                ).indices
                idx = idx[top_local]
            encoded = self.candidate_transformer(token_state[idx].unsqueeze(0)).squeeze(
                0
            )
            delta[idx] = self.candidate_transformer_out(encoded).squeeze(-1)
        return delta

    def _reset_aimnet_charge_residual(self) -> None:
        """
        Start the optional AIMNet charge branch as an exact no-op.
        """
        if self.aimnet_charge_residual is None:
            return
        final = self.aimnet_charge_residual[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def _aimnet_charge_delta(
        self,
        aux_node_features: dict[str, torch.Tensor] | None,
        fragments: dict[str, torch.Tensor],
        graph_batch: torch.Tensor | None,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return a zero-initialized residual from AIMNet charge summaries.
        """
        if (
            self.aimnet_charge_residual is None
            or self.aimnet_charge_input_dropout is None
        ):
            raise RuntimeError('AIMNet charge residual is not initialized.')
        summary = self._aimnet_charge_summary(
            aux_node_features,
            fragments,
            graph_batch,
            formula_batch,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        return self.aimnet_charge_residual(self.aimnet_charge_input_dropout(summary))

    @staticmethod
    def _aimnet_charge_summary(
        aux_node_features: dict[str, torch.Tensor] | None,
        fragments: dict[str, torch.Tensor],
        graph_batch: torch.Tensor | None,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Summarize AIMNet atom charge channels over fragments and complements.
        """
        if (
            aux_node_features is None
            or 'aimnet_charge_features' not in aux_node_features
        ):
            raise ValueError(
                'AIMNet charge features are enabled, but the encoder did not '
                'return `aimnet_charge_features`.'
            )
        charge_feats = aux_node_features['aimnet_charge_features'].to(
            device=device, dtype=dtype
        )
        if charge_feats.ndim != 2:
            raise ValueError('AIMNet charge features must have shape [num_atoms, dim].')
        dim = int(charge_feats.shape[-1])
        if dim != AIMNET_CHARGE_FEATURE_DIM:
            raise ValueError(
                'AIMNet charge feature dimension mismatch: '
                f'expected {AIMNET_CHARGE_FEATURE_DIM}, got {dim}.'
            )

        atom_index = fragments['atom_index'].to(device=device).long()
        atom_ptr = fragments['atom_ptr'].to(device=device).long()
        fragment_sum = FragmentSpectrumHead._sum_fragment_atoms(
            charge_feats, atom_index, atom_ptr
        )
        counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(0).to(device=device)
        fragment_mean = fragment_sum / counts.clamp_min(1).to(dtype=dtype).unsqueeze(-1)

        molecule_sum = FragmentSpectrumHead._sum_molecule_atoms(
            charge_feats, graph_batch, batch_size=batch_size
        )
        molecule_counts = FragmentSpectrumHead._molecule_atom_counts(
            charge_feats, graph_batch, batch_size=batch_size
        )
        complement_sum = molecule_sum[formula_batch] - fragment_sum
        complement_counts = (molecule_counts[formula_batch] - counts).clamp_min(1)
        complement_mean = complement_sum / complement_counts.to(dtype=dtype).unsqueeze(
            -1
        )
        return torch.cat(
            [
                fragment_mean,
                fragment_sum,
                complement_mean,
                complement_sum,
                fragment_mean - complement_mean,
            ],
            dim=-1,
        )

    def _fragment_action_bond_gnn_delta(
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
        Return a local bond-event message-passing residual for formula logits.

        The branch builds one token per oriented broken bond in each retained
        formula candidate. Tokens belonging to the same multi-break candidate
        exchange messages before being scored and pooled back to the formula.
        The residual output is zero-initialized, preserving loaded checkpoint
        predictions until this branch is fine-tuned.
        """
        if self.fragment_action_bond_gnn_layers <= 0 or formula_features.numel() == 0:
            return base_logits.new_zeros(base_logits.shape)
        if (
            self.fragment_action_bond_gnn_input_dropout is None
            or self.fragment_action_bond_gnn_encoder is None
            or self.fragment_action_bond_gnn_scorer is None
            or self.fragment_action_bond_gnn_residual is None
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
            self._materialize_empty_fragment_action_bond_gnn(
                node_feats,
                formula_features,
                context_features,
                collision_features,
                fragment_descriptor,
            )
            return base_logits.new_zeros(base_logits.shape)
        pair_inputs, formula_index, counts = event
        event_state = self.fragment_action_bond_gnn_encoder(
            self.fragment_action_bond_gnn_input_dropout(pair_inputs)
        )
        for layer in self.fragment_action_bond_gnn_layers_module:
            event_state = layer(
                event_state,
                formula_index,
                num_groups=int(formula_features.shape[0]),
            )
        event_logits = self.fragment_action_bond_gnn_scorer(event_state).squeeze(-1)
        action_sum = base_logits.new_zeros(formula_features.shape[0])
        action_sum.index_add_(0, formula_index, event_logits)
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
        return self.fragment_action_bond_gnn_residual(residual_inputs).squeeze(-1)

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

    def _materialize_empty_fragment_action_bond_gnn(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragment_descriptor: torch.Tensor,
    ) -> None:
        if self.fragment_action_bond_gnn_encoder is None:
            return
        if not any(
            is_lazy(param)
            for param in self.fragment_action_bond_gnn_encoder.parameters()
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
            self.fragment_action_bond_gnn_encoder(node_feats.new_zeros(1, input_dim))

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
        ce_gate_logit = self._fragment_action_primary_ce_gate_logit(
            collision_features,
            dtype=base_logits.dtype,
        )
        ce_gate_scalar = ce_gate_logit.squeeze(-1)
        gated_action_score = action_score * (1.0 + torch.tanh(ce_gate_scalar))
        normalized_action = self._standardize_by_batch(
            gated_action_score,
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

    def _fragment_action_primary_ce_gate_logit(
        self,
        collision_features: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        gate = getattr(self, 'fragment_action_primary_ce_gate', None)
        if gate is None:
            return collision_features.new_zeros(
                collision_features.shape[0], 1, dtype=dtype
            )
        return gate(collision_features).to(dtype=dtype)

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

    def _bond_break_event_logits(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        graph: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return oriented bond-break logits and their candidate formula indices.
        """
        empty_index = torch.empty(0, dtype=torch.long, device=formula_features.device)
        if self.fragment_bond_break_layers <= 0:
            return formula_features.new_empty(0), empty_index
        if self.bond_break_pair_scorer is None or self.bond_break_input_dropout is None:
            return formula_features.new_empty(0), empty_index
        if formula_features.numel() == 0:
            self._materialize_empty_bond_break_scorer(
                node_feats, formula_features, context_features, collision_features
            )
            return formula_features.new_empty(0), empty_index

        event = self._fragment_bond_event_inputs(
            node_feats,
            formula_features,
            context_features,
            collision_features,
            formula_features.new_empty(formula_features.shape[0], 0),
            fragments,
            include_formula=False,
            include_fragment_descriptor=False,
            graph=graph,
        )
        if event is None:
            self._materialize_empty_bond_break_scorer(
                node_feats, formula_features, context_features, collision_features
            )
            return formula_features.new_empty(0), empty_index
        pair_inputs, formula_index, _counts = event
        pair_logits = self.bond_break_pair_scorer(
            self.bond_break_input_dropout(pair_inputs)
        ).squeeze(-1)
        return pair_logits, formula_index

    def _bond_break_feature_dim(self) -> int:
        width = BOND_BREAK_FEATURE_DIM
        if bool(getattr(self.config, 'physical_bond_features', False)):
            columns = tuple(
                getattr(self.config, 'physical_bond_feature_columns', ()) or ()
            )
            width += len(columns) + 1
        if self.bond_break_geometry_features:
            width += FRAGMENT_ACTION_GEOMETRY_FEATURE_DIM
        return width

    def _materialize_empty_bond_break_scorer(
        self,
        node_feats: torch.Tensor,
        formula_features: torch.Tensor,
        context_features: torch.Tensor,
        collision_features: torch.Tensor,
    ) -> None:
        """
        Initialize lazy bond-break layers even when a batch has no cut bonds.
        """
        if self.bond_break_pair_scorer is None:
            return
        if not any(
            is_lazy(param) for param in self.bond_break_pair_scorer.parameters()
        ):
            return
        input_dim = (
            4 * int(node_feats.shape[-1])
            + int(context_features.shape[-1])
            + int(collision_features.shape[-1])
            + self._bond_break_feature_dim()
        )
        with torch.no_grad():
            self.bond_break_pair_scorer(node_feats.new_zeros(1, input_dim))

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

    def _fragment_action_path_delta(
        self,
        fragment_descriptor: torch.Tensor,
        collision_inputs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        formula_batch: torch.Tensor,
        *,
        batch_size: int,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a constrained action-path residual for formula logits.

        Unlike the generic path branch, this scores local fragmentation actions
        from hand-crafted fragment and edge descriptors, aggregates retained
        parent-child paths, then calibrates the result against the existing
        formula scorer with a zero-initialized residual.
        """
        if self.fragment_action_path_layers <= 0 or fragment_descriptor.numel() == 0:
            return base_logits.new_zeros(base_logits.shape[0])
        if (
            self.fragment_action_root_scorer is None
            or self.fragment_action_edge_scorer is None
            or self.fragment_action_residual is None
        ):
            return base_logits.new_zeros(base_logits.shape[0])

        root_inputs = torch.cat([fragment_descriptor, collision_inputs], dim=-1)
        root_scores = self.fragment_action_root_scorer(root_inputs).squeeze(-1)
        action_scores = root_scores
        if edge_index.numel() > 0 and edge_attr.numel() > 0:
            parent_mask = edge_attr[:, 0] > 0.5
            if bool(parent_mask.any()):
                src = edge_index[0, parent_mask].long()
                dst = edge_index[1, parent_mask].long()
                parent_descriptor = fragment_descriptor[src]
                child_descriptor = fragment_descriptor[dst]
                edge_inputs = torch.cat(
                    [
                        parent_descriptor,
                        child_descriptor,
                        child_descriptor - parent_descriptor,
                        edge_attr[parent_mask],
                        collision_inputs[dst],
                    ],
                    dim=-1,
                )
                edge_scores = self.fragment_action_edge_scorer(edge_inputs).squeeze(-1)
                root_frontier = self._fragment_path_root_frontier(root_scores, dst)
                propagated = self._propagate_fragment_path_scores(
                    root_frontier,
                    edge_scores,
                    src,
                    dst,
                    num_nodes=fragment_descriptor.shape[0],
                )
                action_scores = torch.where(
                    self._fragment_path_reachable(propagated),
                    propagated,
                    root_scores,
                )

        normalized_action = self._standardize_by_batch(
            action_scores,
            formula_batch,
            batch_size=batch_size,
        )
        normalized_base = self._standardize_by_batch(
            base_logits.detach(),
            formula_batch,
            batch_size=batch_size,
        )
        residual_inputs = torch.stack(
            [
                normalized_action,
                normalized_base,
                normalized_action - normalized_base,
            ],
            dim=-1,
        )
        return self.fragment_action_residual(residual_inputs).squeeze(-1)

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

    def _collision_energy_feature(
        self, metadata_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Return scalar, smooth-basis, or Fourier collision-energy features.

        Metadata is built as ``[precursor_mz, collision_energy, adduct_embedding,
        instrument_embedding]``. The first returned channel is always the old
        scalar CE value, so checkpoint surgery can preserve old behavior by
        copying scalar collision weights into the first column.
        """
        if metadata_features.shape[-1] <= 1:
            z = metadata_features.new_zeros(metadata_features.shape[0], 1)
        else:
            z = metadata_features[:, 1:2]
        mode = self.ce_embedding
        if mode == 'scalar':
            return z
        z = z.clamp(min=-5.0, max=5.0)
        if mode == 'basis':
            centers = z.new_tensor([-2.0, -0.75, 0.75, 2.0]).view(1, -1)
            rbf = torch.exp(-0.5 * (z - centers).square())
            return torch.cat([z, z.square(), z.pow(3), rbf], dim=-1)
        if mode == 'fourier':
            periods = torch.pow(
                z.new_tensor(2.0),
                torch.arange(
                    self.ce_fourier_frequencies,
                    dtype=z.dtype,
                    device=z.device,
                ),
            ).view(1, -1)
            angles = 2.0 * torch.pi * z / periods
            return torch.cat([z, torch.sin(angles), torch.cos(angles)], dim=-1)
        raise RuntimeError(f'Unsupported collision-energy embedding mode: {mode!r}')

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
