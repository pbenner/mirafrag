from __future__ import annotations

import torch

from mirafrag.spectra import num_spectrum_bins

FRAGMENT_CONFIG_ARG_FIELDS = (
    'max_fragment_tree_depth',
    'max_fragment_broken_bonds',
    'max_fragments',
    'max_fragment_edges',
    'include_fragment_isotopes',
    'fragment_isotope_threshold',
    'max_fragment_isotope_peaks',
)


def resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def validate_checkpoint_bin_config(
    model,
    *,
    mz_max: float,
    bin_width: float,
) -> None:
    requested_bins = num_spectrum_bins(mz_max, bin_width)
    if requested_bins != model.config.num_bins:
        raise SystemExit(
            'Checkpoint/bin mismatch: '
            f'checkpoint has {model.config.num_bins} output bins, but '
            f'--mz-max {mz_max} and --bin-width {bin_width} imply '
            f"{requested_bins} bins. Pass the checkpoint's original bin settings "
            'or retrain with the MassSpecGym bin definition.'
        )


def apply_fragment_args_to_model_config(config, args) -> None:
    # Candidate-generation settings are safe to override on a loaded checkpoint.
    # Head architecture fields, such as fragment_gnn_layers, are intentionally
    # excluded because they belong to the saved spectrum head.
    for field_name in FRAGMENT_CONFIG_ARG_FIELDS:
        value = getattr(args, field_name)
        if value is not None:
            setattr(config, field_name, value)


def value_or_default(value, default):
    return default if value is None else value
