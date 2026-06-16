from __future__ import annotations

import argparse

import pandas as pd
from torch.utils.data import DataLoader

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    add_high_ce_fragment_support_args,
    apply_fragment_args_to_model_config,
    resolve_device,
    validate_checkpoint_bin_config,
)
from mirafrag.data import (
    ADDUCT_ALIASES,
    CE_ALIASES,
    INSTRUMENT_ALIASES,
    PRECURSOR_ALIASES,
    BinnedSpectrumDataset,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    find_column,
    read_table,
    select_split,
)
from mirafrag.evaluation import evaluate_model, probability_mode_from_checkpoint_payload
from mirafrag.fragments import fragment_support_profile_from_model_config
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for checkpoint evaluation.

    The options select the MassSpecGym split, checkpoint, bin settings, cache behavior, and exported prediction formatting.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-eval',
        description='Evaluate a MiraFrag checkpoint on a MassSpecGym split.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument('-m', '--model', required=True, help='MiraFrag checkpoint .pt.')
    parser.add_argument('-o', '--output', default=None, help='Optional output CSV.')
    parser.add_argument(
        '--checkpoint-metric',
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', default='test')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--memory-cache',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep decoded graphs/fragments in each worker process.',
    )
    parser.add_argument(
        '--disk-cache-dir',
        default=None,
        help='Optional disk cache for precomputed encoder graphs and fragments.',
    )
    parser.add_argument('--min-intensity', type=float, default=0.001)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--mass-tolerance',
        type=float,
        default=0.01,
        help='Absolute or relative m/z tolerance for oracle support diagnostics.',
    )
    parser.add_argument(
        '--relative-mass-tolerance',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Interpret --mass-tolerance as a relative tolerance for diagnostics.',
    )
    parser.add_argument(
        '--mass-tolerance-min-mz',
        type=float,
        default=200.0,
        help='Minimum m/z denominator for relative diagnostic tolerance.',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during evaluation.',
    )
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--stratify-metadata',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print evaluation metrics grouped by instrument and collision-energy bins.',
    )
    parser.add_argument(
        '--stratify-output',
        default=None,
        help='Optional CSV path for grouped instrument/collision-energy metrics.',
    )
    parser.add_argument(
        '--stratify-min-count',
        type=int,
        default=25,
        help='Minimum samples required for a displayed stratified group.',
    )
    parser.add_argument(
        '--ce-bins',
        type=int,
        default=4,
        help='Number of global quantile bins for collision-energy stratification.',
    )
    add_high_ce_fragment_support_args(parser)
    return parser.parse_args()


def main() -> None:
    """
    Run evaluation for a MiraFrag checkpoint.

    The command loads the checkpoint, validates bin compatibility, builds a filtered dataset, evaluates sparse predictions, prints summary metrics, and optionally writes a CSV.
    """
    args = parse_args()
    quiet_rdkit_logs()
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.model, device=device)
    probability_mode = probability_mode_from_checkpoint_payload(payload)
    if probability_mode == 'decoupled':
        print(
            'Evaluation probability mode: decoupled fragment softmax with sigmoid OOS'
        )
    validate_checkpoint_bin_config(
        model,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )
    apply_fragment_args_to_model_config(model.config, args)

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
        raise SystemExit('No rows selected for evaluation.')

    graph_config = infer_graph_config(model.encoder)
    df, element_stats = filter_supported_elements(
        df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'Evaluation element filter: {element_stats}')
    if df.empty:
        raise SystemExit('No rows left after encoder element filtering.')
    ds = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=model.metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        require_spectrum=True,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_support_profile=fragment_support_profile_from_model_config(
            model.config
        ),
    )
    if args.disk_cache_dir is not None:
        prefill_feature_cache(
            ds,
            split_name=str(args.split_value or args.split),
            chunk_size=args.batch_size,
            num_workers=args.num_workers,
            show_progress=args.progress,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_spectrum_batch,
        **dataloader_performance_kwargs(
            num_workers=args.num_workers,
            device=device,
        ),
    )
    predictions, summary = evaluate_model(
        model,
        loader,
        device=device,
        min_intensity=args.min_intensity,
        top_k=args.top_k,
        show_progress=args.progress,
        mass_tolerance=args.mass_tolerance,
        relative_mass_tolerance=args.relative_mass_tolerance,
        mass_tolerance_min_mz=args.mass_tolerance_min_mz,
        probability_mode=probability_mode,
    )
    print(
        f'n={len(predictions)} '
        f'cosine_mean={summary["cosine_mean"]:.5f} '
        f'sqrt_cosine_mean={summary["sqrt_cosine_mean"]:.5f} '
        f'candidate_coverage_mean={summary["candidate_coverage_mean"]:.5f} '
        f'oos_target_mass_mean={summary["oos_target_mass_mean"]:.5f} '
        f'oracle_binned_cosine_mean={summary["oracle_binned_cosine_mean"]:.5f} '
        f'oracle_tolerance_cosine_mean={summary["oracle_tolerance_cosine_mean"]:.5f} '
        f'support_gap_mean={summary["support_gap_mean"]:.5f} '
        f'scorer_gap_mean={summary["scorer_gap_mean"]:.5f} '
        f'oos_calibration_abs_error_mean='
        f'{summary["oos_calibration_abs_error_mean"]:.5f}'
    )
    if args.stratify_metadata or args.stratify_output:
        predictions = _attach_metadata(predictions, df)
        stratified = _stratified_summary(
            predictions,
            ce_bins=args.ce_bins,
            min_count=args.stratify_min_count,
        )
        _print_stratified_summary(stratified)
        if args.stratify_output:
            stratified.to_csv(args.stratify_output, index=False)
            print(f'Wrote stratified metrics to {args.stratify_output}')
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f'Wrote predictions to {args.output}')


def _attach_metadata(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach original metadata columns to row-wise prediction metrics."""
    if len(predictions) != len(df):
        raise RuntimeError(
            'Cannot attach evaluation metadata: prediction count does not match '
            f'dataframe rows ({len(predictions)} != {len(df)}).',
        )
    out = predictions.reset_index(drop=True).copy()
    source = df.reset_index(drop=True)
    for name, aliases in (
        ('instrument_type', INSTRUMENT_ALIASES),
        ('collision_energy', CE_ALIASES),
        ('adduct', ADDUCT_ALIASES),
        ('precursor_mz', PRECURSOR_ALIASES),
    ):
        column = find_column(source, aliases, required=False)
        if column is None:
            continue
        out[name] = source[column].to_numpy()
    if 'collision_energy' in out:
        out['collision_energy'] = pd.to_numeric(
            out['collision_energy'],
            errors='coerce',
        )
    if 'precursor_mz' in out:
        out['precursor_mz'] = pd.to_numeric(out['precursor_mz'], errors='coerce')
    return out


def _stratified_summary(
    predictions: pd.DataFrame,
    *,
    ce_bins: int,
    min_count: int,
) -> pd.DataFrame:
    """Summarize evaluation metrics by instrument and collision-energy bins."""
    if 'cosine' not in predictions:
        return pd.DataFrame()
    work = predictions.copy()
    if 'instrument_type' in work:
        work['instrument_type'] = work['instrument_type'].fillna('missing').astype(str)
    if 'collision_energy' in work:
        work['collision_energy_bin'] = _collision_energy_bins(
            work['collision_energy'],
            ce_bins=ce_bins,
        )
    frames = []
    if 'instrument_type' in work:
        frames.append(
            _summarize_groups(
                work,
                ['instrument_type'],
                group_type='instrument',
                min_count=min_count,
            ),
        )
    if 'collision_energy_bin' in work:
        frames.append(
            _summarize_groups(
                work,
                ['collision_energy_bin'],
                group_type='collision_energy_bin',
                min_count=min_count,
            ),
        )
    if {'instrument_type', 'collision_energy_bin'} <= set(work.columns):
        frames.append(
            _summarize_groups(
                work,
                ['instrument_type', 'collision_energy_bin'],
                group_type='instrument_x_collision_energy_bin',
                min_count=min_count,
            ),
        )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collision_energy_bins(values: pd.Series, *, ce_bins: int) -> pd.Series:
    """Return stable string labels for quantile collision-energy bins."""
    numeric = pd.to_numeric(values, errors='coerce')
    labels = pd.Series('missing', index=values.index, dtype='object')
    finite = numeric.dropna()
    unique_values = int(finite.nunique())
    if unique_values == 0:
        return labels
    bins = max(1, min(int(ce_bins), unique_values))
    if bins == 1:
        labels.loc[finite.index] = 'all'
        return labels
    cut = pd.qcut(finite, q=bins, duplicates='drop')
    labels.loc[finite.index] = cut.astype(str).to_numpy()
    return labels


def _summarize_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    group_type: str,
    min_count: int,
) -> pd.DataFrame:
    """Aggregate one grouping level into a compact diagnostics table."""
    metric_cols = [
        'cosine',
        'sqrt_cosine',
        'cosine_no_oos',
        'candidate_coverage',
        'oos_target_mass',
        'predicted_oos_probability',
        'oracle_binned_cosine',
        'oracle_tolerance_cosine',
        'support_gap',
        'scorer_gap',
        'tolerance_scorer_gap',
        'oos_calibration_error',
        'oos_calibration_abs_error',
    ]
    rows = []
    for raw_keys, group in df.groupby(group_cols, dropna=False, observed=False):
        keys = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        metric_group = group.dropna(subset=['cosine'])
        n = int(len(metric_group))
        if n < int(min_count):
            continue
        row = {
            'group_type': group_type,
            'group': ' | '.join(str(value) for value in keys),
            'n': n,
        }
        for column in metric_cols:
            if column in metric_group:
                row[f'{column}_mean'] = float(metric_group[column].mean())
        if 'collision_energy' in metric_group:
            ce = pd.to_numeric(metric_group['collision_energy'], errors='coerce')
            row['collision_energy_mean'] = float(ce.mean())
            row['collision_energy_min'] = float(ce.min())
            row['collision_energy_max'] = float(ce.max())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ['group_type', 'cosine_mean', 'n'], ascending=[True, True, False]
    )


def _print_stratified_summary(summary: pd.DataFrame) -> None:
    """Print grouped diagnostics in short tables, lowest cosine first."""
    if summary.empty:
        print('No stratified groups passed the minimum-count threshold.')
        return
    display_cols = [
        'group',
        'n',
        'cosine_mean',
        'sqrt_cosine_mean',
        'candidate_coverage_mean',
        'oos_target_mass_mean',
        'predicted_oos_probability_mean',
        'oracle_tolerance_cosine_mean',
        'collision_energy_mean',
    ]
    formatters = {
        column: (lambda value: f'{value:.5f}')
        for column in display_cols
        if column.endswith('_mean')
    }
    print('Stratified evaluation summary, lowest cosine first:')
    for group_type, group in summary.groupby('group_type', sort=False):
        cols = [column for column in display_cols if column in group.columns]
        print(f'\n{group_type}:')
        print(group[cols].to_string(index=False, formatters=formatters))


if __name__ == '__main__':
    main()
