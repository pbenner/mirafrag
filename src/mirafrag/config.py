from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any


@dataclass
class MiraFragConfig:
    """
    Serializable model and candidate-support configuration.

    The dataclass stores head architecture, fragment generation limits, encoder family, foundation model identifiers, and fine-tuning strategy. It is saved in checkpoints and must remain explicit so old and new runs can be distinguished.
    """

    num_bins: int
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    metadata_dim: int = 32
    max_fragment_tree_depth: int = 3
    max_fragment_broken_bonds: int = 6
    max_fragments: int = 2048
    max_fragment_edges: int = 8192
    high_ce_fragment_threshold: float | None = None
    high_ce_max_fragment_tree_depth: int | None = None
    high_ce_max_fragment_broken_bonds: int | None = None
    high_ce_max_fragments: int | None = None
    high_ce_max_fragment_edges: int | None = None
    include_fragment_isotopes: bool = True
    fragment_isotope_threshold: float = 0.001
    max_fragment_isotope_peaks: int = 1
    fragment_gnn_layers: int = 2
    fragment_path_layers: int = 0
    fragment_path_primary: bool = False
    fragment_action_path_layers: int = 0
    fragment_action_primary_layers: int = 0
    fragment_action_primary_ce_gate: bool = False
    bond_break_geometry_features: bool = False
    fragment_action_geometry_features: bool = False
    fragment_action_bond_gnn_layers: int = 0
    fragment_bond_break_layers: int = 0
    physical_bond_features: bool = False
    physical_bond_feature_columns: tuple[str, ...] = ()
    fragnnet_dag_layers: int = 0
    fragnnet_dag_num_hs: int = 4
    ce_basis_features: bool = False
    ce_embedding: str = 'scalar'
    ce_fourier_frequencies: int = 8
    aimnet_charge_features: bool = False
    aimnet_multipass_features: bool = False
    encoder_bond_adapter_layers: int = 0
    encoder_bond_adapter_feature_dim: int = 128
    encoder_bond_adapter_dropout: float = 0.0
    encoder_metadata_adapter_layers: int = 0
    encoder_metadata_adapter_feature_dim: int = 128
    encoder_metadata_adapter_dropout: float = 0.0
    metadata_ce_interaction: bool = False
    candidate_transformer_layers: int = 0
    candidate_transformer_heads: int = 8
    candidate_transformer_max_tokens: int = 256
    conditional_expert_heads: int = 1
    conditional_expert_hidden_dim: int = 128
    conditional_expert_dropout: float = 0.0
    candidate_suppression_gate_hidden_dim: int = 0
    candidate_suppression_gate_dropout: float = 0.0
    candidate_suppression_gate_initial_penalty: float = 1e-6
    molecule_descriptor_features: bool = False
    molecule_descriptor_hidden_dim: int = 128
    molecule_descriptor_dropout: float = 0.0
    retrieval_calibration_head: bool = False
    encoder_type: str = 'mace'
    encoder_finetune_strategy: str = 'head'
    foundation_source: str = 'off'
    foundation_model: str | None = 'medium'
    foundation_path: str | None = None
    aimnet_model: str | None = 'aimnet2'
    aimnet_path: str | None = None
    unimol_model_name: str = 'unimolv1'
    unimol_model_size: str = '84m'
    unimol_pretrained_model_path: str | None = None
    unimol_pretrained_dict_path: str | None = None
    unimol_max_atoms: int = 512
    unimol_mode: str = 'trainable'


_LEGACY_EXPERIMENTAL_FIELDS = {
    'neutral_loss_layers',
    'ce_regime_heads',
    'aimnet_adapter_layers',
    'aimnet_adapter_hidden_dim',
    'aimnet_adapter_feature_dim',
    'aimnet_adapter_dropout',
    'fragment_transition_features',
}


def mirafrag_config_from_dict(data: dict[str, Any]) -> MiraFragConfig:
    """
    Reconstruct :class:`MiraFragConfig` from checkpoint data.

    The loader is strict about unknown fields and required fields without defaults. Missing optional fields are filled from dataclass defaults so checkpoints created before optional config extensions remain usable.
    """
    expected = {field.name for field in fields(MiraFragConfig)}
    supplied = set(data)
    missing = {
        field.name
        for field in fields(MiraFragConfig)
        if field.name not in supplied and field.default is MISSING
    }
    unknown = supplied - expected - _LEGACY_EXPERIMENTAL_FIELDS
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f'missing={sorted(missing)}')
        if unknown:
            parts.append(f'unknown={sorted(unknown)}')
        raise ValueError(
            'Invalid MiraFragConfig checkpoint payload: ' + ', '.join(parts)
        )
    values = {}
    for field in fields(MiraFragConfig):
        if field.name in data:
            values[field.name] = data[field.name]
        elif field.default is not MISSING:
            values[field.name] = field.default
    return MiraFragConfig(**values)
