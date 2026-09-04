from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from mirafrag.adducts import parse_adduct
from mirafrag.chem import (
    GraphConfig,
    collate_graphs,
    quiet_rdkit_logs,
    smiles_to_graph,
)
from mirafrag.fragments import (
    FragmentConfig,
    FragmentSupportProfile,
    collate_fragment_candidates,
    smiles_to_fragment_candidates,
)
from mirafrag.fragments.mol import _bond_break_stats, _fragment_bond_breaks
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    bin_spectrum,
    parse_peaks,
)

SMILES_ALIASES = ('smiles', 'SMILES', 'Smiles')
PRECURSOR_ALIASES = ('precursor_mz', 'PrecursorMZ', 'PRECURSORMZ', 'PEPMASS')
ADDUCT_ALIASES = ('adduct', 'precursor_type', 'Precursor_type', 'PRECURSORTYPE')
INSTRUMENT_ALIASES = ('instrument_type', 'Instrument_type', 'INSTRUMENTTYPE')
CE_ALIASES = ('collision_energy', 'CE', 'CollisionEnergy', 'COLLISIONENERGY')
RAW_COLLISION_ENERGY_COLUMN = '_mirafrag_raw_collision_energy'
SAMPLE_WEIGHT_COLUMN = '_mirafrag_sample_weight'
FEATURE_CACHE_VERSION = 'v12'
FEATURE_CACHE_FORMAT = 'mirafrag-feature-v1'


@dataclass
class MetadataConfig:
    """
    Serializable metadata vocabulary and normalization statistics.

    It stores categorical vocabularies for adducts and instruments plus robust collision-energy scaling learned from the training split. The model uses this object to build metadata feature vectors.
    """

    adduct_to_idx: dict[str, int]
    instrument_to_idx: dict[str, int]
    precursor_mz_max: float = MASS_SPEC_GYM_MZ_MAX
    collision_energy_max: float = 100.0
    collision_energy_center: float = 0.0
    collision_energy_scale: float = 100.0
    collision_energy_by_instrument: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    collision_energy_mode: str = 'raw'

    @property
    def num_adducts(self) -> int:
        """
        Return the adduct vocabulary size including the unknown bucket.
        """
        return len(self.adduct_to_idx) + 1

    @property
    def num_instruments(self) -> int:
        """
        Return the instrument vocabulary size including the unknown bucket.
        """
        return len(self.instrument_to_idx) + 1

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        precursor_mz_max: float = MASS_SPEC_GYM_MZ_MAX,
        collision_energy_max: float = 100.0,
        collision_energy_mode: str = 'raw',
    ) -> MetadataConfig:
        """
        Learn metadata vocabularies and collision-energy statistics from a dataframe.

        Adduct and instrument labels are collected from non-missing values. Collision energy is standardized with robust median/IQR statistics globally and, when enough examples exist, per instrument.
        """
        adduct_col = find_column(df, ADDUCT_ALIASES, required=False)
        instrument_col = find_column(df, INSTRUMENT_ALIASES, required=False)
        ce_col = find_column(df, CE_ALIASES, required=False)
        adducts = (
            sorted(str(x) for x in df[adduct_col].dropna().unique())
            if adduct_col is not None
            else []
        )
        instruments = (
            sorted(str(x) for x in df[instrument_col].dropna().unique())
            if instrument_col is not None
            else []
        )
        ce_mode = str(collision_energy_mode).lower()
        if ce_mode not in {'raw', 'normalized'}:
            raise ValueError('collision_energy_mode must be one of: raw, normalized.')
        ce_values = (
            _finite_float_array(df[ce_col]) if ce_col is not None else np.array([])
        )
        ce_center, ce_scale = _robust_center_scale(
            ce_values,
            fallback_scale=collision_energy_max,
        )
        ce_by_instrument = (
            _collision_energy_stats_by_instrument(
                df,
                ce_col=ce_col,
                instrument_col=instrument_col,
                fallback_scale=collision_energy_max,
            )
            if ce_col is not None and instrument_col is not None
            else {}
        )
        return cls(
            adduct_to_idx={value: i for i, value in enumerate(adducts)},
            instrument_to_idx={value: i for i, value in enumerate(instruments)},
            precursor_mz_max=precursor_mz_max,
            collision_energy_max=collision_energy_max,
            collision_energy_center=ce_center,
            collision_energy_scale=ce_scale,
            collision_energy_by_instrument=ce_by_instrument,
            collision_energy_mode=ce_mode,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetadataConfig:
        """
        Load metadata configuration from a strict checkpoint dictionary.

        Unknown or missing keys raise errors so checkpoint metadata cannot drift silently from the model code's expectations.
        """
        required = {
            'adduct_to_idx',
            'instrument_to_idx',
            'precursor_mz_max',
            'collision_energy_max',
            'collision_energy_center',
            'collision_energy_scale',
            'collision_energy_by_instrument',
        }
        optional = {'collision_energy_mode'}
        missing = required - set(data)
        unknown = set(data) - required - optional
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f'missing={sorted(missing)}')
            if unknown:
                parts.append(f'unknown={sorted(unknown)}')
            raise ValueError(
                'Invalid MetadataConfig checkpoint payload: ' + ', '.join(parts)
            )
        return cls(
            adduct_to_idx={str(k): int(v) for k, v in data['adduct_to_idx'].items()},
            instrument_to_idx={
                str(k): int(v) for k, v in data['instrument_to_idx'].items()
            },
            precursor_mz_max=float(data['precursor_mz_max']),
            collision_energy_max=float(data['collision_energy_max']),
            collision_energy_center=float(data['collision_energy_center']),
            collision_energy_scale=float(data['collision_energy_scale']),
            collision_energy_by_instrument={
                str(instrument): {
                    'center': float(stats['center']),
                    'scale': float(stats['scale']),
                }
                for instrument, stats in data['collision_energy_by_instrument'].items()
            },
            collision_energy_mode=str(data.get('collision_energy_mode', 'raw')),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dictionary representation.
        """
        return asdict(self)


def find_column(
    df: pd.DataFrame,
    names: tuple[str, ...] | list[str],
    *,
    required: bool = True,
) -> str | None:
    """
    Find the first available column among known aliases.

    When ``required`` is true, a descriptive error is raised if none of the aliases exist. Optional lookups return ``None`` instead.
    """
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f'None of the expected columns were found: {list(names)}')
    return None


def read_table(path: str | Path | None) -> pd.DataFrame:
    """
    Read a CSV, TSV, JSONL, or MassSpecGym-provided table.

    When ``path`` is omitted, the function tries to load MassSpecGym through its Python package. Local paths are parsed based on file suffix.
    """
    if path is None:
        try:
            from massspecgym.utils import load_massspecgym
        except Exception as exc:
            raise RuntimeError(
                'No --input was provided and massspecgym is not installed. '
                'Pass a local MassSpecGym TSV or install massspecgym separately.'
            ) from exc
        return load_massspecgym()

    path = Path(path)
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t')
    if path.suffix.lower() == '.jsonl':
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def normalize_collision_energy_dataframe(
    df: pd.DataFrame,
    *,
    metadata_config: MetadataConfig | None = None,
    reference_df: pd.DataFrame | None = None,
    collision_energy_max: float = 100.0,
) -> pd.DataFrame:
    """
    Replace collision energy with train-reference normalized collision energy.

    This mirrors FraGNNet's NCE-style usage at the dataframe boundary: the model
    can then consume CE as an already normalized scalar without applying a second
    robust transform. Per-instrument robust stats are used when available, with a
    global robust fallback.
    """
    ce_col = find_column(df, CE_ALIASES, required=False)
    if ce_col is None:
        return df.reset_index(drop=True).copy()
    if metadata_config is None:
        if reference_df is None:
            raise ValueError(
                'Pass metadata_config or reference_df for CE normalization.'
            )
        ref_ce_col = find_column(reference_df, CE_ALIASES, required=False)
        if ref_ce_col is None:
            return df.reset_index(drop=True).copy()
        metadata_config = MetadataConfig.from_dataframe(
            reference_df,
            collision_energy_max=collision_energy_max,
            collision_energy_mode='normalized',
        )
    instrument_col = find_column(df, INSTRUMENT_ALIASES, required=False)
    global_center = float(metadata_config.collision_energy_center)
    global_scale = float(metadata_config.collision_energy_scale)
    by_instrument = metadata_config.collision_energy_by_instrument

    out = df.reset_index(drop=True).copy()
    if RAW_COLLISION_ENERGY_COLUMN not in out.columns:
        out[RAW_COLLISION_ENERGY_COLUMN] = pd.to_numeric(
            out[ce_col],
            errors='coerce',
        )
    source_ce_col = (
        RAW_COLLISION_ENERGY_COLUMN
        if RAW_COLLISION_ENERGY_COLUMN in out.columns
        else ce_col
    )
    normalized: list[float] = []
    for _, row in out.iterrows():
        value = _coerce_optional_float(row.get(source_ce_col))
        if not np.isfinite(value):
            normalized.append(0.0)
            continue
        center = global_center
        scale = global_scale
        if instrument_col is not None:
            stats = by_instrument.get(_coerce_string(row.get(instrument_col)))
            if stats:
                center = float(stats['center'])
                scale = float(stats['scale'])
        normalized.append((float(value) - center) / max(float(scale), 1e-6))
    out[ce_col] = normalized
    return out


def merge_group_spectra(
    df: pd.DataFrame,
    *,
    mz_max: float = MASS_SPEC_GYM_MZ_MAX,
    bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
    group_cols: str | list[str] | tuple[str, ...] = 'auto',
    exclude_precursor_peaks: bool = True,
    precursor_peak_tolerance: float = MASS_SPEC_GYM_BIN_WIDTH,
) -> pd.DataFrame:
    """
    Merge replicate spectra inside the already selected split.

    The default grouping keeps molecule structure, adduct, and instrument fixed,
    while allowing collision energies to merge into a single consensus target.
    Peaks are accumulated on the configured m/z grid and exported as sparse
    bin-center peaks. Collision-energy values are preserved as JSON lists so
    downstream metadata encoders can aggregate them without losing replicate CE
    information.
    """
    if df.empty:
        return df.reset_index(drop=True).copy()
    precursor_col = find_column(df, PRECURSOR_ALIASES, required=False)
    ce_col = find_column(df, CE_ALIASES, required=False)
    resolved_group_cols = _resolve_merge_group_cols(df, group_cols)
    rows: list[pd.Series] = []
    for _, group in df.groupby(resolved_group_cols, dropna=False, sort=False):
        first = group.iloc[0].copy()
        spectrum = None
        for _, row in group.iterrows():
            precursor_mz = (
                _coerce_float(row.get(precursor_col), 0.0)
                if precursor_col is not None
                else 0.0
            )
            mzs, intensities = parse_peaks(
                row,
                precursor_mz=precursor_mz,
                exclude_precursor=exclude_precursor_peaks,
                precursor_tolerance=precursor_peak_tolerance,
            )
            binned = bin_spectrum(
                mzs,
                intensities,
                mz_max=mz_max,
                bin_width=bin_width,
                normalize='none',
            ).numpy()
            spectrum = binned if spectrum is None else spectrum + binned
        assert spectrum is not None
        nonzero = np.flatnonzero(spectrum > 0)
        first['mzs'] = ','.join(
            f'{(float(idx) + 0.5) * float(bin_width):.6g}' for idx in nonzero
        )
        first['intensities'] = ','.join(
            f'{float(spectrum[idx]):.8g}' for idx in nonzero
        )
        if 'peaks' in first.index:
            first['peaks'] = np.nan
        if ce_col is not None:
            ce_values = _finite_float_array(group[ce_col])
            first[ce_col] = _json_float_list(ce_values) if ce_values.size else np.nan
        if RAW_COLLISION_ENERGY_COLUMN in first.index:
            raw_ce_values = _finite_float_array(group[RAW_COLLISION_ENERGY_COLUMN])
            first[RAW_COLLISION_ENERGY_COLUMN] = (
                _json_float_list(raw_ce_values) if raw_ce_values.size else np.nan
            )
        if 'identifier' in first.index:
            first['identifier'] = '|'.join(str(x) for x in group['identifier'].tolist())
        rows.append(first)
    return pd.DataFrame(rows).reset_index(drop=True)


def _resolve_merge_group_cols(
    df: pd.DataFrame,
    group_cols: str | list[str] | tuple[str, ...],
) -> list[str]:
    if group_cols != 'auto':
        if isinstance(group_cols, str):
            cols = [col.strip() for col in group_cols.split(',') if col.strip()]
        else:
            cols = [str(col) for col in group_cols]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f'Merge group columns are missing: {missing}')
        return cols
    cols: list[str] = []
    smiles_col = find_column(df, SMILES_ALIASES, required=False)
    adduct_col = find_column(df, ADDUCT_ALIASES, required=False)
    instrument_col = find_column(df, INSTRUMENT_ALIASES, required=False)
    for col in (smiles_col, adduct_col, instrument_col):
        if col is not None and col not in cols:
            cols.append(col)
    if not cols:
        raise ValueError('Could not infer merge group columns.')
    return cols


def filter_massspecgym_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select MassSpecGym spectrum-simulation rows with collision energy.

    The simulation-challenge flag is interpreted from booleans, numeric values, and common truthy strings. Rows without collision energy are removed because the training task uses CE metadata.
    """
    out = df.copy()
    if 'simulation_challenge' in out.columns:
        out = out[_boolean_mask(out['simulation_challenge'])]
    ce_col = find_column(out, CE_ALIASES, required=False)
    if ce_col is not None:
        out = out[~out[ce_col].isna()]
    return out.reset_index(drop=True)


def _boolean_mask(series: pd.Series) -> pd.Series:
    """
    Convert a heterogeneous series to a boolean mask.

    The helper accepts real booleans, numeric values, and common truthy strings while treating missing values as false.
    """
    truthy = {'1', 'true', 't', 'yes', 'y'}

    def coerce(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return str(value).strip().lower() in truthy

    return series.map(coerce).astype(bool)


def filter_supported_elements(
    df: pd.DataFrame,
    *,
    supported_atomic_numbers: tuple[int, ...] | list[int],
    smiles_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop molecules containing elements unsupported by the encoder.

    The function parses each SMILES with RDKit, compares atomic numbers with the encoder support set, and returns both the filtered dataframe and detailed drop statistics.
    """
    smiles_col = smiles_col or find_column(df, SMILES_ALIASES)
    supported = {int(z) for z in supported_atomic_numbers}
    keep = []
    dropped_invalid = 0
    dropped_unsupported = 0
    unsupported_counts: dict[int, int] = {}

    for smiles in df[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            keep.append(False)
            dropped_invalid += 1
            continue
        atom_numbers = {int(atom.GetAtomicNum()) for atom in mol.GetAtoms()}
        missing = sorted(atom_numbers - supported)
        if missing:
            keep.append(False)
            dropped_unsupported += 1
            for atomic_number in missing:
                unsupported_counts[atomic_number] = (
                    unsupported_counts.get(atomic_number, 0) + 1
                )
            continue
        keep.append(True)

    stats = {
        'input': int(len(df)),
        'kept': int(sum(keep)),
        'dropped_invalid_smiles': int(dropped_invalid),
        'dropped_unsupported_elements': int(dropped_unsupported),
    }
    for atomic_number, count in sorted(unsupported_counts.items()):
        stats[f'unsupported_Z_{atomic_number}'] = int(count)
    return df.loc[keep].reset_index(drop=True), stats


def select_split(
    df: pd.DataFrame,
    *,
    split: str,
    split_col: str = 'auto',
    split_value: str | None = None,
) -> pd.DataFrame:
    """
    Select rows from a named train/validation/test split.

    The split column can be inferred from common names, or a caller can pass an explicit column and split value. Labels are normalized to handle common aliases such as ``valid`` and ``validation``.
    """
    if split_value is not None:
        if split_col == 'auto':
            split_col = _infer_split_col(df)
        values = df[split_col].astype(str).str.strip()
        return df[values == str(split_value).strip()].reset_index(drop=True)

    if split_col == 'auto':
        split_col = _infer_split_col(df, required=False)
    if split_col is None:
        return df.reset_index(drop=True)

    aliases = {
        'train': {'train', 'training', '0'},
        'val': {'val', 'valid', 'validation', '1'},
        'validation': {'val', 'valid', 'validation', '1'},
        'test': {'test', 'testing', '2'},
    }
    wanted = aliases.get(split, {split})
    values = df[split_col].astype(str).str.strip().str.lower()
    return df[values.isin(wanted)].reset_index(drop=True)


def _infer_split_col(df: pd.DataFrame, required: bool = True) -> str | None:
    """
    Infer the dataframe column that stores split labels.
    """
    for col in ('split', 'fold', 'datasplit', 'data_split'):
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            'Could not infer split column. Pass --split-col and optionally --split-value.'
        )
    return None


def _coerce_float(value, default: float = 0.0) -> float:
    """
    Convert a value to float with a default for missing or invalid values.
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _coerce_string(value) -> str:
    """
    Convert a dataframe value to a clean string, mapping missing values to empty string.
    """
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value)


def _coerce_optional_float(value) -> float:
    """
    Convert a value to float and use NaN for missing or invalid values.
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float('nan')
        out = float(value)
        return out if np.isfinite(out) else float('nan')
    except Exception:
        return float('nan')


def _finite_float_array(values) -> np.ndarray:
    """
    Convert a sequence to a finite-only NumPy float array.

    Entries may themselves be merged collision-energy lists encoded as Python
    sequences or JSON strings. Those nested values are flattened so metadata
    normalization statistics are learned from the original spectra, not from
    per-group averages.
    """
    out: list[float] = []
    for value in values:
        out.extend(_coerce_float_list(value))
    array = np.asarray(out, dtype=float)
    return array[np.isfinite(array)]


def _coerce_float_list(value: Any) -> list[float]:
    """Return finite float values from a scalar or merged CE list."""
    out: list[float] = []
    if value is None or (isinstance(value, float) and np.isnan(value)):
        pass
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        for item in value:
            out.extend(_coerce_float_list(item))
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith('[') and text.endswith(']'):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if parsed is not None:
                out.extend(_coerce_float_list(parsed))
        elif ';' in text:
            for part in text.split(';'):
                out.extend(_coerce_float_list(part))
        elif text:
            number = _coerce_optional_float(text)
            if np.isfinite(number):
                out.append(float(number))
    else:
        number = _coerce_optional_float(value)
        if np.isfinite(number):
            out.append(float(number))
    return out


def _collision_energy_scalar(value: Any, *, default: float = 0.0) -> float:
    """Return the mean CE for scalar compatibility with old code paths."""
    values = _coerce_float_list(value)
    return float(np.mean(values)) if values else float(default)


def _collision_energy_support_value(value: Any) -> float | None:
    """Return a conservative CE value for fragment-support selection."""
    values = _coerce_float_list(value)
    return float(np.max(values)) if values else None


def _json_float_list(values: np.ndarray) -> str:
    """Serialize finite CE values without losing replicate-level metadata."""
    return json.dumps([float(value) for value in values if np.isfinite(value)])


def _robust_center_scale(
    values: np.ndarray,
    *,
    fallback_scale: float,
) -> tuple[float, float]:
    """
    Compute robust center and scale for collision-energy normalization.

    The center is the median and the scale is IQR adjusted to standard-deviation units. Standard deviation and a safe fallback are used when the robust scale collapses.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    safe_fallback = max(float(fallback_scale), 1.0)
    if finite.size == 0:
        return 0.0, safe_fallback

    center = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if scale <= 1e-8:
        scale = float(np.std(finite))
    if scale <= 1e-8:
        scale = safe_fallback
    return center, scale


def _collision_energy_stats_by_instrument(
    df: pd.DataFrame,
    *,
    ce_col: str,
    instrument_col: str,
    fallback_scale: float,
    min_values: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Compute per-instrument robust collision-energy normalization statistics.

    Instrument groups with too few finite values are skipped so rare instruments fall back to the global normalization.
    """
    stats: dict[str, dict[str, float]] = {}
    for instrument, group in df.groupby(instrument_col, dropna=True):
        values = _finite_float_array(group[ce_col])
        if values.size < min_values:
            continue
        center, scale = _robust_center_scale(values, fallback_scale=fallback_scale)
        stats[str(instrument)] = {'center': center, 'scale': scale}
    return stats


def _graph_config_cache_settings(config: GraphConfig) -> dict[str, Any]:
    """
    Return cache-key settings for graph generation.

    Default RDKit preprocessing intentionally preserves the historical graph
    cache key even though newer ``GraphConfig`` objects contain AIMNet relaxation
    fields. Opting into AIMNet relaxation records those fields and therefore
    uses a separate graph namespace while sharing fragment caches.
    """
    settings = asdict(config)
    relaxation = str(settings.get('relaxation', 'rdkit')).lower()
    if relaxation == 'rdkit':
        for key in (
            'relaxation',
            'aimnet_relax_model',
            'aimnet_relax_steps',
            'aimnet_relax_fmax',
            'aimnet_relax_device',
        ):
            settings.pop(key, None)
    return settings


def _fragment_config_cache_settings(config: FragmentConfig) -> dict[str, Any]:
    """
    Return fragment-cache settings without invalidating old default caches.

    Formula-composition features are part of the fragment payload schema, so a
    schema marker prevents old disk-cache entries from silently disabling them.
    Bond-break provenance is derived from retained atom masks on cache load, so
    it intentionally does not fork the recursive-fragment cache namespace.
    """
    settings = asdict(config)
    settings['fragment_feature_schema'] = 3
    settings.pop('include_bond_breaks', None)
    return settings


def _ensure_fragment_bond_break_fields(
    fragments: dict[str, Any],
    smiles: str,
) -> dict[str, Any]:
    """
    Add bond-break provenance to cached fragment payloads when requested.

    The expensive fragment tree already stores retained atom masks. Boundary-bond
    provenance is a deterministic function of those masks and the molecule bond
    table, so it can be reconstructed cheaply without forking the fragment-cache
    key or rebuilding recursive candidates.
    """
    atom_indices = fragments.get('atom_indices')
    if atom_indices is None:
        return fragments
    bond_atom_indices = fragments.get('bond_atom_indices')
    bond_features = fragments.get('bond_features')
    if (
        bond_atom_indices is not None
        and bond_features is not None
        and len(bond_atom_indices) == len(atom_indices)
        and len(bond_features) == len(atom_indices)
    ):
        return fragments

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    bond_stats = _bond_break_stats(mol)
    max_bond_score = max((float(item['score']) for item in bond_stats), default=1.0)
    denominator = max(max_bond_score, 1.0)
    pairs_out: list[tuple[tuple[int, int], ...]] = []
    features_out: list[tuple[tuple[float, float, float], ...]] = []
    for indices in atom_indices:
        atom_set = {int(idx) for idx in indices}
        bond_breaks = _fragment_bond_breaks(atom_set, bond_stats)
        pairs_out.append(
            tuple((int(item['inside']), int(item['outside'])) for item in bond_breaks)
        )
        features_out.append(
            tuple(
                (
                    float(item['order_weight']) / 3.0,
                    float(item['score']) / denominator,
                    1.0 if bool(item.get('hetero', False)) else 0.0,
                )
                for item in bond_breaks
            )
        )
    out = dict(fragments)
    out['bond_atom_indices'] = pairs_out
    out['bond_features'] = features_out
    return out


class BinnedSpectrumDataset(Dataset):
    """
    Dataset that yields graph, metadata, spectrum targets, and fragment candidates.

    Despite the historical name, predictions are sparse fragment candidates; the dense bin settings are used for target binning, cache keys, and MassSpecGym metric compatibility. Optional disk and memory caches avoid recomputing expensive graph and fragment features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        graph_config: GraphConfig,
        metadata_config: MetadataConfig,
        mz_max: float = MASS_SPEC_GYM_MZ_MAX,
        bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
        require_spectrum: bool = True,
        memory_cache: bool = False,
        disk_cache_dir: str | Path | None = None,
        include_fragments: bool = False,
        fragment_config: FragmentConfig | None = None,
        fragment_support_profile: FragmentSupportProfile | None = None,
        slow_sample_seconds: float = 0.0,
        trace_samples: bool = False,
        exclude_precursor_peaks: bool = True,
        precursor_peak_tolerance: float = MASS_SPEC_GYM_BIN_WIDTH,
        sample_weight_col: str | None = None,
    ) -> None:
        """
        Initialize dataframe columns, cache settings, and feature-generation configs.

        The dataset keeps a reset-index copy of the dataframe and resolves column aliases once. Graphs and fragments can be cached in memory per worker and/or on disk across runs.
        """
        self.df = df.reset_index(drop=True).copy()
        self.graph_config = graph_config
        self.metadata_config = metadata_config
        self.mz_max = float(mz_max)
        self.bin_width = float(bin_width)
        self.require_spectrum = require_spectrum
        self.memory_cache = memory_cache
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.include_fragments = include_fragments
        self.fragment_config = (
            fragment_config
            or (fragment_support_profile.base if fragment_support_profile else None)
            or FragmentConfig()
        )
        self.fragment_support_profile = (
            fragment_support_profile
            or FragmentSupportProfile(base=self.fragment_config)
        )
        self.slow_sample_seconds = float(slow_sample_seconds)
        self.trace_samples = bool(trace_samples)
        self.exclude_precursor_peaks = bool(exclude_precursor_peaks)
        self.precursor_peak_tolerance = float(precursor_peak_tolerance)
        self._graph_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._fragment_cache: dict[int, dict[str, Any]] = {}
        self.sample_weight_col = sample_weight_col

        self.smiles_col = find_column(self.df, SMILES_ALIASES)
        self.precursor_col = find_column(self.df, PRECURSOR_ALIASES, required=False)
        self.adduct_col = find_column(self.df, ADDUCT_ALIASES, required=False)
        self.instrument_col = find_column(self.df, INSTRUMENT_ALIASES, required=False)
        self.ce_col = find_column(self.df, CE_ALIASES, required=False)
        self.raw_ce_col = (
            RAW_COLLISION_ENERGY_COLUMN
            if RAW_COLLISION_ENERGY_COLUMN in self.df.columns
            else self.ce_col
        )

    def __len__(self) -> int:
        """
        Return the number of dataframe rows available to the dataset.
        """
        return len(self.df)

    def _graph(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Load or compute the encoder graph for one row.

        Disk cache keys include the SMILES string and graph configuration, and graph files are additionally grouped under a graph-configuration namespace. Different encoders or embedding settings therefore do not share incompatible graph tensors, while fragment caches can still be shared from the same cache root.
        """
        if self.memory_cache and idx in self._graph_cache:
            return self._graph_cache[idx]
        smiles = str(self.df.at[idx, self.smiles_col])
        if self.disk_cache_dir is not None:
            path = self._feature_cache_path(
                'graphs',
                smiles,
                {
                    'graph_config': _graph_config_cache_settings(self.graph_config),
                },
            )
            graph = _load_feature_cache(path)
            if graph is not None:
                if self.memory_cache:
                    self._graph_cache[idx] = graph
                return graph
        graph = smiles_to_graph(smiles, self.graph_config)
        if self.disk_cache_dir is not None:
            _save_feature_cache(path, graph)
        if self.memory_cache:
            self._graph_cache[idx] = graph
        return graph

    def _fragment_config_for_row(self, idx: int) -> FragmentConfig:
        """
        Return base or high-collision-energy fragment support for one row.
        """
        collision_energy = (
            _collision_energy_support_value(self.df.at[idx, self.raw_ce_col])
            if self.raw_ce_col is not None
            else None
        )
        return self.fragment_support_profile.config_for_collision_energy(
            collision_energy
        )

    def _fragments(self, idx: int) -> dict[str, Any]:
        """
        Load or compute fragment candidates for one row.

        Fragment cache keys include SMILES, adduct, mass range, bin width, and fragment settings because each of those changes candidate m/z support.
        """
        if self.memory_cache and idx in self._fragment_cache:
            return self._fragment_cache[idx]
        smiles = str(self.df.at[idx, self.smiles_col])
        adduct = (
            _coerce_string(self.df.at[idx, self.adduct_col]) if self.adduct_col else ''
        )
        fragment_config = self._fragment_config_for_row(idx)
        if self.disk_cache_dir is not None:
            path = self._feature_cache_path(
                'fragments',
                smiles,
                {
                    'fragment_config': _fragment_config_cache_settings(fragment_config),
                    'adduct': adduct,
                    'mz_max': self.mz_max,
                    'bin_width': self.bin_width,
                },
            )
            fragments = _load_feature_cache(path)
            if fragments is not None:
                if bool(fragment_config.include_bond_breaks):
                    fragments = _ensure_fragment_bond_break_fields(fragments, smiles)
                    if (
                        'bond_atom_indices' not in fragments
                        or 'bond_features' not in fragments
                    ):
                        raise ValueError('Fragment bond-break augmentation failed.')
                if self.memory_cache:
                    self._fragment_cache[idx] = fragments
                return fragments
        fragments = smiles_to_fragment_candidates(
            smiles,
            mz_max=self.mz_max,
            bin_width=self.bin_width,
            adduct=adduct,
            config=fragment_config,
        )
        if self.disk_cache_dir is not None:
            _save_feature_cache(path, fragments)
        if self.memory_cache:
            self._fragment_cache[idx] = fragments
        return fragments

    def _feature_cache_path(
        self,
        kind: str,
        smiles: str,
        settings: dict[str, Any],
    ) -> Path:
        """
        Build the deterministic disk cache path for one feature payload.

        The filename is a SHA-256 digest over format version, feature kind,
        SMILES, and settings, which keeps filenames short while preserving
        invalidation behavior. Graph features are placed below an additional
        graph-settings namespace so one disk cache root can share fragment files
        across encoders while keeping graph tensors separate.
        """
        if self.disk_cache_dir is None:
            raise RuntimeError('disk_cache_dir is not configured.')
        payload = {
            'version': FEATURE_CACHE_VERSION,
            'kind': kind,
            'smiles': smiles,
            'settings': settings,
        }
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        directory = self.disk_cache_dir / kind
        if kind == 'graphs':
            namespace_payload = {
                'version': FEATURE_CACHE_VERSION,
                'kind': kind,
                'settings': settings,
            }
            namespace_text = json.dumps(
                namespace_payload,
                sort_keys=True,
                separators=(',', ':'),
            )
            namespace = hashlib.sha256(namespace_text.encode('utf-8')).hexdigest()[:16]
            directory = directory / f'graph-{namespace}'
        return directory / f'{digest}.pt'

    def _metadata(self, row) -> dict[str, torch.Tensor]:
        """
        Convert one dataframe row into model metadata tensors.

        The returned tensors include precursor m/z, collision energy, categorical
        indices, and parsed adduct charge for charge-aware encoders such as AIMNet.
        """
        precursor_mz = (
            _coerce_float(row.get(self.precursor_col), 0.0)
            if self.precursor_col
            else 0.0
        )
        ce_values = _coerce_float_list(row.get(self.ce_col)) if self.ce_col else []
        if not ce_values:
            ce_values = [0.0]
        collision_energy = float(np.mean(ce_values))
        adduct = _coerce_string(row.get(self.adduct_col)) if self.adduct_col else ''
        instrument = (
            _coerce_string(row.get(self.instrument_col)) if self.instrument_col else ''
        )
        return {
            'precursor_mz': torch.tensor(precursor_mz, dtype=torch.float32),
            'collision_energy': torch.tensor(collision_energy, dtype=torch.float32),
            'collision_energy_values': torch.tensor(
                ce_values,
                dtype=torch.float32,
            ),
            'adduct': torch.tensor(
                self.metadata_config.adduct_to_idx.get(
                    adduct, len(self.metadata_config.adduct_to_idx)
                ),
                dtype=torch.long,
            ),
            'adduct_charge': torch.tensor(
                parse_adduct(adduct).charge,
                dtype=torch.float32,
            ),
            'instrument_type': torch.tensor(
                self.metadata_config.instrument_to_idx.get(
                    instrument, len(self.metadata_config.instrument_to_idx)
                ),
                dtype=torch.long,
            ),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return one training/evaluation sample dictionary.

        The item always contains graph, metadata, identifiers, and bin width. Spectrum targets and fragment candidates are included according to dataset configuration, with optional slow-sample diagnostics.
        """
        start = time.perf_counter()
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_col])
        identifier = str(row.get('identifier', idx))
        if self.trace_samples:
            print(
                f'start MiraFrag sample idx={idx} identifier={identifier!r} '
                f'smiles={smiles!r}',
                file=sys.stderr,
                flush=True,
            )
        graph_start = start
        graph = self._graph(idx)
        graph_seconds = time.perf_counter() - graph_start
        item: dict[str, Any] = {
            'graph': graph,
            'smiles': smiles,
            'identifier': identifier,
            'bin_width': torch.tensor(self.bin_width, dtype=torch.float32),
        }
        item.update(self._metadata(row))
        fragment_seconds = 0.0
        if self.include_fragments:
            fragment_start = time.perf_counter()
            item['fragments'] = self._fragments(idx)
            fragment_seconds = time.perf_counter() - fragment_start

        if '_retrieval_query_identifier' in row.index:
            item['_retrieval_query_identifier'] = str(
                row.get('_retrieval_query_identifier')
            )
        if '_retrieval_is_true' in row.index:
            item['_retrieval_is_true'] = bool(row.get('_retrieval_is_true'))
        if self.sample_weight_col is not None:
            item['sample_weight'] = torch.tensor(
                float(row.get(self.sample_weight_col)),
                dtype=torch.float32,
            )
        if self.require_spectrum:
            precursor_mz = float(item['precursor_mz'].item())
            mzs, intensities = parse_peaks(
                row,
                precursor_mz=precursor_mz,
                exclude_precursor=self.exclude_precursor_peaks,
                precursor_tolerance=self.precursor_peak_tolerance,
            )
            item['true_mzs'] = mzs
            item['true_intensities'] = intensities
        elapsed = time.perf_counter() - start
        if self.slow_sample_seconds > 0 and elapsed >= self.slow_sample_seconds:
            print(
                'slow MiraFrag sample '
                f'idx={idx} identifier={item["identifier"]!r} '
                f'elapsed={elapsed:.2f}s graph={graph_seconds:.2f}s '
                f'fragments={fragment_seconds:.2f}s '
                f'smiles={item["smiles"]!r}',
                file=sys.stderr,
                flush=True,
            )
        return item


def collate_spectrum_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate dataset samples into a model-ready batch.

    Graphs are concatenated, metadata tensors are stacked, variable-length target peaks are flattened with a target-batch index, and fragment candidates are offset into global graph node coordinates.
    """
    graph = collate_graphs([item['graph'] for item in items])
    batch = {
        'graph': graph,
        'precursor_mz': torch.stack([item['precursor_mz'] for item in items]),
        'collision_energy': torch.stack([item['collision_energy'] for item in items]),
        'collision_energy_values': torch.cat(
            [item['collision_energy_values'] for item in items]
        ),
        'collision_energy_batch': torch.cat(
            [
                torch.full(
                    (int(item['collision_energy_values'].numel()),),
                    batch_idx,
                    dtype=torch.long,
                )
                for batch_idx, item in enumerate(items)
            ]
        ),
        'adduct': torch.stack([item['adduct'] for item in items]),
        'adduct_charge': torch.stack([item['adduct_charge'] for item in items]),
        'instrument_type': torch.stack([item['instrument_type'] for item in items]),
        'bin_width': torch.stack([item['bin_width'] for item in items]),
        'smiles': [item['smiles'] for item in items],
        'identifier': [item['identifier'] for item in items],
    }
    if 'sample_weight' in items[0]:
        batch['sample_weight'] = torch.stack([item['sample_weight'] for item in items])
    if '_retrieval_query_identifier' in items[0]:
        group_lookup: dict[str, int] = {}
        group_indices: list[int] = []
        for item in items:
            query_id = str(item['_retrieval_query_identifier'])
            if query_id not in group_lookup:
                group_lookup[query_id] = len(group_lookup)
            group_indices.append(group_lookup[query_id])
        batch['_retrieval_group'] = torch.tensor(group_indices, dtype=torch.long)
        batch['_retrieval_is_true'] = torch.tensor(
            [bool(item.get('_retrieval_is_true', False)) for item in items],
            dtype=torch.bool,
        )
    if 'true_mzs' in items[0]:
        target_mzs: list[torch.Tensor] = []
        target_intensities: list[torch.Tensor] = []
        target_batch: list[torch.Tensor] = []
        for batch_idx, item in enumerate(items):
            mzs = torch.as_tensor(item['true_mzs'], dtype=torch.get_default_dtype())
            intensities = torch.as_tensor(
                item['true_intensities'],
                dtype=torch.get_default_dtype(),
            )
            target_mzs.append(mzs)
            target_intensities.append(intensities)
            target_batch.append(
                torch.full((int(mzs.numel()),), batch_idx, dtype=torch.long)
            )
        batch['target_mz'] = (
            torch.cat(target_mzs)
            if target_mzs
            else torch.empty(0, dtype=torch.get_default_dtype())
        )
        batch['target_intensity'] = (
            torch.cat(target_intensities)
            if target_intensities
            else torch.empty(0, dtype=torch.get_default_dtype())
        )
        batch['target_batch'] = (
            torch.cat(target_batch)
            if target_batch
            else torch.empty(0, dtype=torch.long)
        )
        batch['true_mzs'] = [item['true_mzs'] for item in items]
        batch['true_intensities'] = [item['true_intensities'] for item in items]
    if 'fragments' in items[0]:
        node_offsets = [int(offset) for offset in graph['ptr'][:-1].tolist()]
        batch['fragments'] = collate_fragment_candidates(
            [item['fragments'] for item in items],
            node_offsets=node_offsets,
        )
    return batch


def dataloader_performance_kwargs(
    *,
    num_workers: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """
    Return DataLoader options tuned for the selected worker/device setup.

    CUDA loaders use pinned memory for faster host-to-device transfers. When CUDA is active and workers are requested, workers are started with ``spawn`` instead of ``fork`` so they do not inherit an already-initialized CUDA runtime from checkpoint loading, cache prefill, or initial validation.
    """
    kwargs: dict[str, Any] = {}
    use_cuda = _is_cuda_device(device)
    if use_cuda:
        kwargs['pin_memory'] = True
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = _quiet_rdkit_worker_init
        if use_cuda:
            kwargs['multiprocessing_context'] = 'spawn'
    return kwargs


def _load_feature_cache(path: Path) -> Any | None:
    """
    Load a feature cache file and validate its format/version.

    Invalid, stale, or unreadable cache files are deleted and treated as misses so future accesses can regenerate them.
    """
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError('feature cache payload is not a dict')
        if payload.get('cache_format') != FEATURE_CACHE_FORMAT:
            raise ValueError('feature cache format mismatch')
        if payload.get('feature_cache_version') != FEATURE_CACHE_VERSION:
            raise ValueError('feature cache version mismatch')
        return payload['value']
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None


def _save_feature_cache(path: Path, value: Any) -> None:
    """
    Atomically write a graph or fragment feature cache file.

    A process-specific temporary file is written first and then replaced into position to reduce corruption when multiple workers fill the same cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(
            {
                'cache_format': FEATURE_CACHE_FORMAT,
                'feature_cache_version': FEATURE_CACHE_VERSION,
                'value': value,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _quiet_rdkit_worker_init(_worker_id: int) -> None:
    """
    Silence RDKit logs inside a DataLoader worker process.
    """
    quiet_rdkit_logs()


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device | str
) -> dict[str, Any]:
    """
    Move tensors in a collated batch to a torch device.

    Nested graph and fragment dictionaries are handled recursively one level deep, while non-tensor metadata such as SMILES strings and identifiers stays on the host.
    """
    non_blocking = _is_cuda_device(device)
    out = {}
    for key, value in batch.items():
        if key in {'graph', 'fragments'}:
            out[key] = {
                k: v.to(device, non_blocking=non_blocking) for k, v in value.items()
            }
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def _is_cuda_device(device: torch.device | str) -> bool:
    """
    Return whether a torch device string or object refers to CUDA.
    """
    return torch.device(device).type == 'cuda'
