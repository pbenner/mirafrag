from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirafrag.adducts import PROTON_MASS


@dataclass(frozen=True)
class FragmentConfig:
    max_tree_depth: int = 3
    max_broken_bonds: int = 6
    max_fragments: int = 2048
    max_edges: int = 8192
    include_isotopes: bool = True
    isotope_threshold: float = 0.001
    max_isotope_peaks: int = 1
    include_root: bool = True
    proton_mass: float = PROTON_MASS


def fragment_config_from_model_config(config: Any) -> FragmentConfig:
    return FragmentConfig(
        max_tree_depth=config.max_fragment_tree_depth,
        max_broken_bonds=config.max_fragment_broken_bonds,
        max_fragments=config.max_fragments,
        max_edges=config.max_fragment_edges,
        include_isotopes=config.include_fragment_isotopes,
        isotope_threshold=config.fragment_isotope_threshold,
        max_isotope_peaks=config.max_fragment_isotope_peaks,
    )
