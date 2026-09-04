from __future__ import annotations

from types import SimpleNamespace

import torch

from mirafrag.fragments import (
    FragmentConfig,
    high_ce_fragment_config_from_model_config,
)
from mirafrag.spectra import num_spectrum_bins

BASE_FRAGMENT_CONFIG_ARG_FIELDS = (
    'max_fragment_tree_depth',
    'max_fragment_broken_bonds',
    'max_fragments',
    'max_fragment_edges',
    'include_fragment_isotopes',
    'fragment_isotope_threshold',
    'max_fragment_isotope_peaks',
    'fragment_action_primary_layers',
    'bond_break_geometry_features',
    'fragment_action_geometry_features',
)
HIGH_CE_FRAGMENT_CONFIG_ARG_FIELDS = (
    'high_ce_fragment_threshold',
    'high_ce_max_fragment_tree_depth',
    'high_ce_max_fragment_broken_bonds',
    'high_ce_max_fragments',
    'high_ce_max_fragment_edges',
)
FRAGMENT_CONFIG_ARG_FIELDS = (
    *BASE_FRAGMENT_CONFIG_ARG_FIELDS,
    *HIGH_CE_FRAGMENT_CONFIG_ARG_FIELDS,
)


def resolve_device(device: str) -> str:
    """
    Resolve the CLI device string to a concrete torch device string.

    ``auto`` selects the first CUDA device when available and otherwise CPU. Explicit device strings are returned unchanged.
    """
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def validate_checkpoint_bin_config(
    model,
    *,
    mz_max: float,
    bin_width: float,
) -> None:
    """
    Ensure runtime bin settings match a loaded checkpoint.

    The spectrum head has a fixed number of output bins. This guard prevents evaluating or resuming a checkpoint with incompatible ``mz_max`` and ``bin_width`` settings.
    """
    requested_bins = num_spectrum_bins(mz_max, bin_width)
    if requested_bins != model.config.num_bins:
        raise SystemExit(
            'Checkpoint/bin mismatch: '
            f'checkpoint has {model.config.num_bins} output bins, but '
            f'--mz-max {mz_max} and --bin-width {bin_width} imply '
            f"{requested_bins} bins. Pass the checkpoint's original bin settings "
            'or retrain with the MassSpecGym bin definition.'
        )


def add_high_ce_fragment_support_args(parser) -> None:
    """
    Add opt-in high-collision-energy fragment support arguments.
    """
    parser.add_argument(
        '--high-ce-fragment-threshold',
        type=float,
        default=None,
        help=(
            'Raw collision-energy threshold above which rows switch to expanded '
            'fragment support; unset keeps one global fragment config.'
        ),
    )
    parser.add_argument(
        '--high-ce-max-fragment-tree-depth',
        type=int,
        default=None,
        help='High-CE override for maximum recursive fragment tree depth.',
    )
    parser.add_argument(
        '--high-ce-max-fragment-broken-bonds',
        type=int,
        default=None,
        help='High-CE override for broken-bond and hydrogen-transfer budget.',
    )
    parser.add_argument(
        '--high-ce-max-fragments',
        type=int,
        default=None,
        help='High-CE override for retained fragment formula candidates.',
    )
    parser.add_argument(
        '--high-ce-max-fragment-edges',
        type=int,
        default=None,
        help='High-CE override for retained fragment relationship edges.',
    )


def apply_fragment_args_to_model_config(config, args) -> None:
    """
    Apply safe fragment-candidate overrides to a model config.

    Candidate-generation settings are updated directly.
    """
    for field_name in FRAGMENT_CONFIG_ARG_FIELDS:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config, field_name, value)
    geometry_features = getattr(args, 'bond_break_geometry_features', None)
    if geometry_features is None:
        geometry_features = getattr(args, 'fragment_action_geometry_features', None)
    if geometry_features is not None:
        enabled = bool(geometry_features)
        setattr(config, 'bond_break_geometry_features', enabled)
        setattr(config, 'fragment_action_geometry_features', enabled)


def high_ce_fragment_config_from_args(
    base: FragmentConfig,
    args,
) -> FragmentConfig | None:
    """
    Build high-CE fragment support from CLI args without a model config.
    """
    if getattr(args, 'high_ce_fragment_threshold', None) is None:
        return None
    namespace = SimpleNamespace(
        high_ce_fragment_threshold=args.high_ce_fragment_threshold,
        high_ce_max_fragment_tree_depth=getattr(
            args, 'high_ce_max_fragment_tree_depth', None
        ),
        high_ce_max_fragment_broken_bonds=getattr(
            args, 'high_ce_max_fragment_broken_bonds', None
        ),
        high_ce_max_fragments=getattr(args, 'high_ce_max_fragments', None),
        high_ce_max_fragment_edges=getattr(args, 'high_ce_max_fragment_edges', None),
    )
    return high_ce_fragment_config_from_model_config(namespace, base=base)


def value_or_default(value, default):
    """
    Return a fallback when an optional CLI value is unset.

    This keeps CLI-to-config conversion explicit without repeating ``if value is None`` logic at every field.
    """
    return default if value is None else value
