from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from mirafrag.checkpoint import CHECKPOINT_FORMAT
from mirafrag.chem import GraphConfig, quiet_rdkit_logs
from mirafrag.config import MiraFragConfig, mirafrag_config_from_dict
from mirafrag.data import (
    BinnedSpectrumDataset,
    MetadataConfig,
    filter_massspecgym_simulation,
    filter_supported_elements,
    read_table,
    select_split,
)
from mirafrag.evaluation import support_diagnostics
from mirafrag.fragments import (
    collate_fragment_candidates,
    fragment_config_from_model_config,
)
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    num_spectrum_bins,
    parse_peaks,
)

_ORACLE_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for fragment-support oracle diagnostics.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-oracle',
        description='Compute candidate-support oracle bounds without model scoring.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='MiraFrag checkpoint defining encoder filter and fragment settings.',
    )
    parser.add_argument('-o', '--output', default=None, help='Optional per-row CSV.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split', default='train')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Rows per oracle worker task; defaults to --batch-size.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Parallel workers for fragment-candidate oracle computation.',
    )
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--memory-cache',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep decoded fragment candidates in process memory.',
    )
    parser.add_argument(
        '--disk-cache-dir',
        default=None,
        help='Optional disk cache for fragment candidates.',
    )
    parser.add_argument(
        '--mass-tolerance',
        type=float,
        default=0.01,
        help='Absolute or relative m/z tolerance for tolerance oracle.',
    )
    parser.add_argument(
        '--relative-mass-tolerance',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Interpret --mass-tolerance as a relative tolerance.',
    )
    parser.add_argument(
        '--mass-tolerance-min-mz',
        type=float,
        default=200.0,
        help='Minimum m/z denominator for relative tolerance.',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress while computing diagnostics.',
    )
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    """
    Compute candidate-support oracle metrics for a MassSpecGym split.
    """
    args = parse_args()
    quiet_rdkit_logs()
    # Kept for CLI symmetry; oracle diagnostics run on CPU tensors.
    _ = args.device
    config, metadata_config, graph_config = _load_oracle_checkpoint_config(
        args.model,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )
    fragment_config = fragment_config_from_model_config(config)

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    df = select_split(
        df,
        split=args.split,
        split_col=args.split_col,
        split_value=args.split_value,
    )
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()
    if df.empty:
        raise SystemExit('No rows selected for oracle diagnostics.')

    df, element_stats = filter_supported_elements(
        df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'Oracle element filter: {element_stats}')
    if df.empty:
        raise SystemExit('No rows left after encoder element filtering.')

    ds = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        require_spectrum=True,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_config=fragment_config,
    )

    rows, summary = compute_oracle_diagnostics(
        ds,
        batch_size=args.batch_size,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        mass_tolerance=args.mass_tolerance,
        relative_mass_tolerance=args.relative_mass_tolerance,
        mass_tolerance_min_mz=args.mass_tolerance_min_mz,
        show_progress=args.progress,
        split_name=str(args.split_value or args.split),
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    print(
        f'n={int(summary["n"])} '
        f'candidate_coverage_mean={summary["candidate_coverage_mean"]:.5f} '
        f'oos_target_mass_mean={summary["oos_target_mass_mean"]:.5f} '
        f'oracle_binned_cosine_mean={summary["oracle_binned_cosine_mean"]:.5f} '
        f'oracle_tolerance_cosine_mean={summary["oracle_tolerance_cosine_mean"]:.5f}'
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output, index=False)
        print(f'Wrote oracle diagnostics to {output}')


def _load_oracle_checkpoint_config(
    path: str,
    *,
    mz_max: float,
    bin_width: float,
) -> tuple[MiraFragConfig, MetadataConfig, GraphConfig]:
    """
    Load checkpoint metadata needed for oracle diagnostics without building encoders.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('checkpoint_format') != CHECKPOINT_FORMAT:
        raise ValueError(
            'Unsupported MiraFrag checkpoint format '
            f'{payload.get("checkpoint_format")!r}; expected {CHECKPOINT_FORMAT!r}.'
        )
    config = mirafrag_config_from_dict(payload['mirafrag_config'])
    expected_bins = num_spectrum_bins(mz_max, bin_width)
    if int(config.num_bins) != int(expected_bins):
        raise ValueError(
            'Checkpoint/bin mismatch: checkpoint has '
            f'{config.num_bins} bins but mz_max={mz_max} and bin_width={bin_width} '
            f'imply {expected_bins} bins.'
        )
    metadata_config = MetadataConfig.from_dict(payload['metadata_config'])
    graph_config = _graph_config_from_state_dict(payload['model_state_dict'])
    return config, metadata_config, graph_config


def _graph_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> GraphConfig:
    """
    Reconstruct encoder element support and cutoff from checkpoint buffers.
    """
    atomic_numbers = _state_tensor_by_suffix(state_dict, 'atomic_numbers')
    r_max = _state_tensor_by_suffix(state_dict, 'r_max')
    return GraphConfig(
        atomic_numbers=tuple(int(value) for value in atomic_numbers.detach().cpu()),
        cutoff=float(r_max.detach().cpu().item()),
    )


def _state_tensor_by_suffix(
    state_dict: dict[str, torch.Tensor],
    suffix: str,
) -> torch.Tensor:
    """
    Return the encoder metadata tensor whose checkpoint key ends with ``suffix``.
    """
    preferred_suffixes = (f'encoder.{suffix}', f'base_module.{suffix}')
    for key, value in state_dict.items():
        if key.endswith(preferred_suffixes):
            return value
    matches = [value for key, value in state_dict.items() if key.endswith(suffix)]
    if not matches:
        raise ValueError(f'Checkpoint state_dict is missing *.{suffix}.')
    return matches[0]


def compute_oracle_diagnostics(
    dataset: BinnedSpectrumDataset,
    *,
    batch_size: int,
    mz_max: float,
    bin_width: float,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    show_progress: bool = True,
    split_name: str = 'split',
    num_workers: int = 0,
    chunk_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute support oracle metrics from fragment candidates and target spectra.
    """
    batch_size = max(1, int(batch_size))
    chunk_size = max(1, int(chunk_size or batch_size))
    num_bins = num_spectrum_bins(mz_max, bin_width)
    chunks = [
        (start, min(start + chunk_size, len(dataset)))
        for start in range(0, len(dataset), chunk_size)
    ]
    settings = {
        'num_bins': num_bins,
        'bin_width': bin_width,
        'mass_tolerance': mass_tolerance,
        'relative_mass_tolerance': relative_mass_tolerance,
        'mass_tolerance_min_mz': mass_tolerance_min_mz,
    }
    num_workers = max(0, int(num_workers))
    if num_workers <= 0:
        results = (_oracle_chunk(dataset, chunk, settings) for chunk in chunks)
        return _consume_oracle_results(
            results,
            total_chunks=len(chunks),
            show_progress=show_progress,
            split_name=split_name,
        )

    context = mp.get_context(_multiprocessing_start_method())
    with context.Pool(
        processes=num_workers,
        initializer=_init_oracle_worker,
        initargs=(dataset, settings),
    ) as pool:
        results = pool.imap_unordered(_oracle_worker_chunk, chunks, chunksize=1)
        return _consume_oracle_results(
            results,
            total_chunks=len(chunks),
            show_progress=show_progress,
            split_name=split_name,
        )


def _consume_oracle_results(
    results: Any,
    *,
    total_chunks: int,
    show_progress: bool,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Aggregate unordered oracle chunk results into per-row output and means.
    """
    rows: list[dict[str, Any]] = []
    all_candidate_coverage: list[float] = []
    all_oos_target_mass: list[float] = []
    all_oracle_binned: list[float] = []
    all_oracle_tolerance: list[float] = []
    progress = tqdm(
        results,
        desc=f'oracle {split_name}',
        total=total_chunks,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for result in progress:
        rows.extend(result['rows'])
        all_candidate_coverage.extend(result['candidate_coverage'])
        all_oos_target_mass.extend(result['oos_target_mass'])
        all_oracle_binned.extend(result['oracle_binned_cosine'])
        all_oracle_tolerance.extend(result['oracle_tolerance_cosine'])
        if show_progress:
            progress.set_postfix(
                {
                    'oracle_binned': f'{_mean_or_nan(all_oracle_binned):.4f}',
                    'oracle_tol': f'{_mean_or_nan(all_oracle_tolerance):.4f}',
                }
            )
    rows.sort(key=lambda row: row['row_index'])
    summary = {
        'n': len(rows),
        'candidate_coverage_mean': _mean_or_nan(all_candidate_coverage),
        'oos_target_mass_mean': _mean_or_nan(all_oos_target_mass),
        'oracle_binned_cosine_mean': _mean_or_nan(all_oracle_binned),
        'oracle_tolerance_cosine_mean': _mean_or_nan(all_oracle_tolerance),
    }
    return pd.DataFrame(rows), summary


def _oracle_chunk(
    dataset: BinnedSpectrumDataset,
    chunk: tuple[int, int],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute oracle diagnostics for one contiguous dataset chunk.
    """
    start, stop = chunk
    batch_rows, diagnostics = _oracle_batch(
        dataset,
        indices=range(start, stop),
        num_bins=int(settings['num_bins']),
        bin_width=float(settings['bin_width']),
        mass_tolerance=float(settings['mass_tolerance']),
        relative_mass_tolerance=bool(settings['relative_mass_tolerance']),
        mass_tolerance_min_mz=float(settings['mass_tolerance_min_mz']),
    )
    return {
        'rows': batch_rows,
        'candidate_coverage': [
            float(value) for value in diagnostics['candidate_coverage']
        ],
        'oos_target_mass': [float(value) for value in diagnostics['oos_target_mass']],
        'oracle_binned_cosine': [
            float(value) for value in diagnostics['oracle_binned_cosine']
        ],
        'oracle_tolerance_cosine': [
            float(value) for value in diagnostics['oracle_tolerance_cosine']
        ],
    }


def _init_oracle_worker(
    dataset: BinnedSpectrumDataset,
    settings: dict[str, Any],
) -> None:
    """
    Store oracle worker state and silence RDKit logs inside child processes.
    """
    quiet_rdkit_logs()
    _ORACLE_STATE['dataset'] = dataset
    _ORACLE_STATE['settings'] = settings


def _oracle_worker_chunk(chunk: tuple[int, int]) -> dict[str, Any]:
    """
    Compute one oracle chunk from worker-global dataset state.
    """
    dataset = _ORACLE_STATE.get('dataset')
    settings = _ORACLE_STATE.get('settings')
    if dataset is None or settings is None:
        raise RuntimeError('Oracle worker was not initialized.')
    return _oracle_chunk(dataset, chunk, settings)


def _multiprocessing_start_method() -> str:
    """
    Use fork when CUDA has not been initialized, otherwise spawn.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return 'spawn'
    return 'fork'


def _oracle_batch(
    dataset: BinnedSpectrumDataset,
    *,
    indices: range,
    num_bins: int,
    bin_width: float,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """
    Compute support diagnostics for one dataframe index range.
    """
    fragments = []
    target_mzs = []
    target_intensities = []
    target_batch = []
    rows = []
    for batch_idx, idx in enumerate(indices):
        row = dataset.df.iloc[idx]
        mzs, intensities = parse_peaks(row)
        fragments.append(dataset._fragments(idx))
        target_mzs.append(torch.as_tensor(mzs, dtype=torch.get_default_dtype()))
        target_intensities.append(
            torch.as_tensor(intensities, dtype=torch.get_default_dtype())
        )
        target_batch.append(torch.full((len(mzs),), batch_idx, dtype=torch.long))
        rows.append(
            {
                'row_index': int(idx),
                'identifier': str(row.get('identifier', idx)),
                'smiles': str(row[dataset.smiles_col]),
            }
        )
    fragment_batch = collate_fragment_candidates(
        fragments,
        node_offsets=[0 for _ in fragments],
    )
    batch = {
        'target_mz': _cat_or_empty(target_mzs, dtype=torch.get_default_dtype()),
        'target_intensity': _cat_or_empty(
            target_intensities,
            dtype=torch.get_default_dtype(),
        ),
        'target_batch': _cat_or_empty(target_batch, dtype=torch.long),
        'bin_width': torch.full(
            (len(rows),),
            float(bin_width),
            dtype=torch.get_default_dtype(),
        ),
    }
    pred = {
        'bins': fragment_batch['bin'],
        'mzs': fragment_batch['mz'],
        'batch': fragment_batch['batch'],
        'batch_size': len(rows),
        'num_bins': int(num_bins),
    }
    diagnostics = support_diagnostics(
        pred,
        batch,
        tolerance=mass_tolerance,
        relative=relative_mass_tolerance,
        tolerance_min_mz=mass_tolerance_min_mz,
    )
    for batch_idx, row in enumerate(rows):
        row['candidate_coverage'] = float(diagnostics['candidate_coverage'][batch_idx])
        row['oos_target_mass'] = float(diagnostics['oos_target_mass'][batch_idx])
        row['oracle_binned_cosine'] = float(
            diagnostics['oracle_binned_cosine'][batch_idx]
        )
        row['oracle_tolerance_cosine'] = float(
            diagnostics['oracle_tolerance_cosine'][batch_idx]
        )
    return rows, diagnostics


def _cat_or_empty(tensors: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    """
    Concatenate tensors or return an empty tensor of the requested dtype.
    """
    if tensors:
        return torch.cat(tensors)
    return torch.empty(0, dtype=dtype)


def _mean_or_nan(values: list[float]) -> float:
    """
    Compute a Python mean and return NaN for empty inputs.
    """
    return float(sum(values) / len(values)) if values else float('nan')


if __name__ == '__main__':
    main()
