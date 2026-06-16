from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mirafrag.adducts import PROTON_MASS


@dataclass(frozen=True)
class FragmentConfig:
    """
    Configuration for recursive fragment candidate generation.

    The settings bound tree depth, broken-bond and hydrogen-transfer budget, retained formulas, fragment-graph edges, isotope expansion, and the proton mass fallback for missing adducts.
    """

    max_tree_depth: int = 3
    max_broken_bonds: int = 6
    max_fragments: int = 2048
    max_edges: int = 8192
    include_isotopes: bool = True
    isotope_threshold: float = 0.001
    max_isotope_peaks: int = 1
    include_root: bool = True
    proton_mass: float = PROTON_MASS


@dataclass(frozen=True)
class FragmentSupportProfile:
    """
    Row-level fragment support profile.

    Most spectra use base. Rows whose raw collision energy is at or above
    high_ce_threshold use high_ce when it is configured. This keeps the
    normal cache compact while allowing targeted high-collision-energy support
    experiments.
    """

    base: FragmentConfig
    high_ce_threshold: float | None = None
    high_ce: FragmentConfig | None = None

    def config_for_collision_energy(
        self, collision_energy: float | None
    ) -> FragmentConfig:
        """
        Return the effective fragment config for one row.
        """
        if self.high_ce is None or self.high_ce_threshold is None:
            return self.base
        value = _finite_float_or_none(collision_energy)
        threshold = _finite_float_or_none(self.high_ce_threshold)
        if value is None or threshold is None:
            return self.base
        if value >= threshold:
            return self.high_ce
        return self.base

    def is_enabled(self) -> bool:
        """
        Return whether the profile can ever select high-CE support.
        """
        return self.high_ce is not None and self.high_ce_threshold is not None


def fragment_config_from_model_config(config: Any) -> FragmentConfig:
    """
    Extract fragment-generation settings from a MiraFrag model config.
    """
    return FragmentConfig(
        max_tree_depth=config.max_fragment_tree_depth,
        max_broken_bonds=config.max_fragment_broken_bonds,
        max_fragments=config.max_fragments,
        max_edges=config.max_fragment_edges,
        include_isotopes=config.include_fragment_isotopes,
        isotope_threshold=config.fragment_isotope_threshold,
        max_isotope_peaks=config.max_fragment_isotope_peaks,
    )


def high_ce_fragment_config_from_model_config(
    config: Any,
    *,
    base: FragmentConfig | None = None,
) -> FragmentConfig | None:
    """
    Build the optional high-collision-energy fragment config from model settings.

    Passing only high_ce_fragment_threshold enables a conservative support
    expansion: one extra tree level, two extra broken-bond/H-transfer units, and
    doubled formula/edge budgets. Explicit high-CE settings override these
    defaults. Isotope settings are inherited from the base fragment config.
    """
    threshold = getattr(config, 'high_ce_fragment_threshold', None)
    if threshold is None:
        return None
    base = base or fragment_config_from_model_config(config)
    return FragmentConfig(
        max_tree_depth=int(
            _value_or_default(
                getattr(config, 'high_ce_max_fragment_tree_depth', None),
                max(int(base.max_tree_depth) + 1, 4),
            )
        ),
        max_broken_bonds=int(
            _value_or_default(
                getattr(config, 'high_ce_max_fragment_broken_bonds', None),
                max(int(base.max_broken_bonds) + 2, 8),
            )
        ),
        max_fragments=int(
            _value_or_default(
                getattr(config, 'high_ce_max_fragments', None),
                max(int(base.max_fragments) * 2, 4096),
            )
        ),
        max_edges=int(
            _value_or_default(
                getattr(config, 'high_ce_max_fragment_edges', None),
                max(int(base.max_edges) * 2, 16384),
            )
        ),
        include_isotopes=bool(base.include_isotopes),
        isotope_threshold=float(base.isotope_threshold),
        max_isotope_peaks=int(base.max_isotope_peaks),
        include_root=bool(base.include_root),
        proton_mass=float(base.proton_mass),
    )


def fragment_support_profile_from_model_config(config: Any) -> FragmentSupportProfile:
    """
    Extract base and optional high-CE fragment support from a model config.
    """
    base = fragment_config_from_model_config(config)
    return FragmentSupportProfile(
        base=base,
        high_ce_threshold=getattr(config, 'high_ce_fragment_threshold', None),
        high_ce=high_ce_fragment_config_from_model_config(config, base=base),
    )


def _finite_float_or_none(value: Any) -> float | None:
    """Return a finite float or None for nonnumeric values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _value_or_default(value: Any, default: Any) -> Any:
    """Return default when value is unset."""
    return default if value is None else value
