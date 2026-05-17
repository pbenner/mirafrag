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
    collate_fragment_candidates,
    smiles_to_fragment_candidates,
)
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    parse_peaks,
)

SMILES_ALIASES = ('smiles', 'SMILES', 'Smiles')
PRECURSOR_ALIASES = ('precursor_mz', 'PrecursorMZ', 'PRECURSORMZ', 'PEPMASS')
ADDUCT_ALIASES = ('adduct', 'precursor_type', 'Precursor_type', 'PRECURSORTYPE')
INSTRUMENT_ALIASES = ('instrument_type', 'Instrument_type', 'INSTRUMENTTYPE')
CE_ALIASES = ('collision_energy', 'CE', 'CollisionEnergy', 'COLLISIONENERGY')
FEATURE_CACHE_VERSION = 'v12'
FEATURE_CACHE_FORMAT = 'mirafrag-feature-v1'


@dataclass
class MetadataConfig:
    adduct_to_idx: dict[str, int]
    instrument_to_idx: dict[str, int]
    precursor_mz_max: float = MASS_SPEC_GYM_MZ_MAX
    collision_energy_max: float = 100.0
    collision_energy_center: float = 0.0
    collision_energy_scale: float = 100.0
    collision_energy_by_instrument: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    @property
    def num_adducts(self) -> int:
        return len(self.adduct_to_idx) + 1

    @property
    def num_instruments(self) -> int:
        return len(self.instrument_to_idx) + 1

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        precursor_mz_max: float = MASS_SPEC_GYM_MZ_MAX,
        collision_energy_max: float = 100.0,
    ) -> MetadataConfig:
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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetadataConfig:
        required = {
            'adduct_to_idx',
            'instrument_to_idx',
            'precursor_mz_max',
            'collision_energy_max',
            'collision_energy_center',
            'collision_energy_scale',
            'collision_energy_by_instrument',
        }
        missing = required - set(data)
        unknown = set(data) - required
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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_column(
    df: pd.DataFrame,
    names: tuple[str, ...] | list[str],
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f'None of the expected columns were found: {list(names)}')
    return None


def read_table(path: str | Path | None) -> pd.DataFrame:
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


def filter_massspecgym_simulation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'simulation_challenge' in out.columns:
        out = out[_boolean_mask(out['simulation_challenge'])]
    ce_col = find_column(out, CE_ALIASES, required=False)
    if ce_col is not None:
        out = out[~out[ce_col].isna()]
    return out.reset_index(drop=True)


def _boolean_mask(series: pd.Series) -> pd.Series:
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
    for col in ('split', 'fold', 'datasplit', 'data_split'):
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            'Could not infer split column. Pass --split-col and optionally --split-value.'
        )
    return None


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _coerce_string(value) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value)


def _coerce_optional_float(value) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float('nan')
        out = float(value)
        return out if np.isfinite(out) else float('nan')
    except Exception:
        return float('nan')


def _finite_float_array(values) -> np.ndarray:
    out = np.asarray([_coerce_optional_float(value) for value in values], dtype=float)
    return out[np.isfinite(out)]


def _robust_center_scale(
    values: np.ndarray,
    *,
    fallback_scale: float,
) -> tuple[float, float]:
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
    stats: dict[str, dict[str, float]] = {}
    for instrument, group in df.groupby(instrument_col, dropna=True):
        values = _finite_float_array(group[ce_col])
        if values.size < min_values:
            continue
        center, scale = _robust_center_scale(values, fallback_scale=fallback_scale)
        stats[str(instrument)] = {'center': center, 'scale': scale}
    return stats


class BinnedSpectrumDataset(Dataset):
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
        slow_sample_seconds: float = 0.0,
        trace_samples: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.graph_config = graph_config
        self.metadata_config = metadata_config
        self.mz_max = float(mz_max)
        self.bin_width = float(bin_width)
        self.require_spectrum = require_spectrum
        self.memory_cache = memory_cache
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.include_fragments = include_fragments
        self.fragment_config = fragment_config or FragmentConfig()
        self.slow_sample_seconds = float(slow_sample_seconds)
        self.trace_samples = bool(trace_samples)
        self._graph_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._fragment_cache: dict[int, dict[str, Any]] = {}

        self.smiles_col = find_column(self.df, SMILES_ALIASES)
        self.precursor_col = find_column(self.df, PRECURSOR_ALIASES, required=False)
        self.adduct_col = find_column(self.df, ADDUCT_ALIASES, required=False)
        self.instrument_col = find_column(self.df, INSTRUMENT_ALIASES, required=False)
        self.ce_col = find_column(self.df, CE_ALIASES, required=False)

    def __len__(self) -> int:
        return len(self.df)

    def _graph(self, idx: int) -> dict[str, torch.Tensor]:
        if self.memory_cache and idx in self._graph_cache:
            return self._graph_cache[idx]
        smiles = str(self.df.at[idx, self.smiles_col])
        if self.disk_cache_dir is not None:
            path = self._feature_cache_path(
                'graphs',
                smiles,
                {
                    'graph_config': asdict(self.graph_config),
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

    def _fragments(self, idx: int) -> dict[str, Any]:
        if self.memory_cache and idx in self._fragment_cache:
            return self._fragment_cache[idx]
        smiles = str(self.df.at[idx, self.smiles_col])
        adduct = (
            _coerce_string(self.df.at[idx, self.adduct_col]) if self.adduct_col else ''
        )
        if self.disk_cache_dir is not None:
            path = self._feature_cache_path(
                'fragments',
                smiles,
                {
                    'fragment_config': asdict(self.fragment_config),
                    'adduct': adduct,
                    'mz_max': self.mz_max,
                    'bin_width': self.bin_width,
                },
            )
            fragments = _load_feature_cache(path)
            if fragments is not None:
                if self.memory_cache:
                    self._fragment_cache[idx] = fragments
                return fragments
        fragments = smiles_to_fragment_candidates(
            smiles,
            mz_max=self.mz_max,
            bin_width=self.bin_width,
            adduct=adduct,
            config=self.fragment_config,
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
        return self.disk_cache_dir / kind / f'{digest}.pt'

    def _metadata(self, row) -> dict[str, torch.Tensor]:
        precursor_mz = (
            _coerce_float(row.get(self.precursor_col), 0.0)
            if self.precursor_col
            else 0.0
        )
        collision_energy = (
            _coerce_float(row.get(self.ce_col), 0.0) if self.ce_col else 0.0
        )
        adduct = _coerce_string(row.get(self.adduct_col)) if self.adduct_col else ''
        instrument = (
            _coerce_string(row.get(self.instrument_col)) if self.instrument_col else ''
        )
        return {
            'precursor_mz': torch.tensor(precursor_mz, dtype=torch.float32),
            'collision_energy': torch.tensor(collision_energy, dtype=torch.float32),
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

        if self.require_spectrum:
            mzs, intensities = parse_peaks(row)
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
    graph = collate_graphs([item['graph'] for item in items])
    batch = {
        'graph': graph,
        'precursor_mz': torch.stack([item['precursor_mz'] for item in items]),
        'collision_energy': torch.stack([item['collision_energy'] for item in items]),
        'adduct': torch.stack([item['adduct'] for item in items]),
        'adduct_charge': torch.stack([item['adduct_charge'] for item in items]),
        'instrument_type': torch.stack([item['instrument_type'] for item in items]),
        'bin_width': torch.stack([item['bin_width'] for item in items]),
        'smiles': [item['smiles'] for item in items],
        'identifier': [item['identifier'] for item in items],
    }
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
    kwargs: dict[str, Any] = {}
    if _is_cuda_device(device):
        kwargs['pin_memory'] = True
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
        kwargs['worker_init_fn'] = _quiet_rdkit_worker_init
    return kwargs


def _load_feature_cache(path: Path) -> Any | None:
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
    quiet_rdkit_logs()


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device | str
) -> dict[str, Any]:
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
    return torch.device(device).type == 'cuda'
