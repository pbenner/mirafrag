from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class MiraFragConfig:
    num_bins: int
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    metadata_dim: int = 32
    max_fragment_tree_depth: int = 3
    max_fragment_broken_bonds: int = 6
    max_fragments: int = 2048
    max_fragment_edges: int = 8192
    include_fragment_isotopes: bool = True
    fragment_isotope_threshold: float = 0.001
    max_fragment_isotope_peaks: int = 1
    fragment_gnn_layers: int = 2
    encoder_type: str = 'mace'
    encoder_finetune_strategy: str = 'head'
    foundation_source: str = 'off'
    foundation_model: str | None = 'medium'
    foundation_path: str | None = None
    aimnet_model: str | None = 'aimnet2'
    aimnet_path: str | None = None


def mirafrag_config_from_dict(data: dict[str, Any]) -> MiraFragConfig:
    expected = {field.name for field in fields(MiraFragConfig)}
    supplied = set(data)
    missing = expected - supplied
    unknown = supplied - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f'missing={sorted(missing)}')
        if unknown:
            parts.append(f'unknown={sorted(unknown)}')
        raise ValueError(
            'Invalid MiraFragConfig checkpoint payload: ' + ', '.join(parts)
        )
    return MiraFragConfig(
        **{field.name: data[field.name] for field in fields(MiraFragConfig)}
    )
