from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

from mirafrag.chem import quiet_rdkit_logs
from mirafrag.data import (
    CE_ALIASES,
    INSTRUMENT_ALIASES,
    SMILES_ALIASES,
    filter_massspecgym_simulation,
    find_column,
    read_table,
    select_split,
)

DEFAULT_SIMILARITY_BINS = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0)
DEFAULT_CE_BINS = (-np.inf, 20.0, 35.0, 60.0, np.inf)
DEFAULT_CE_LABELS = ('<=20', '(20,35]', '(35,60]', '>60')
METRIC_COLUMNS = (
    'cosine',
    'sqrt_cosine',
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
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for train/validation gap diagnostics."""
    parser = argparse.ArgumentParser(
        prog='mirafrag-gap-diagnostics',
        description=(
            'Analyze whether evaluation errors track chemical novelty, '
            'instrument/CE regime, or candidate-support limits.'
        ),
    )
    parser.add_argument(
        '-i', '--input', required=True, help='MassSpecGym TSV/CSV path.'
    )
    parser.add_argument(
        '-p',
        '--predictions',
        default=None,
        help='Optional prediction CSV from mirafrag.cli.eval.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Optional row-level diagnostics CSV.',
    )
    parser.add_argument(
        '--summary-output',
        default=None,
        help='Optional grouped summary CSV.',
    )
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--eval-split', default='val')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--eval-split-value', default=None)
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--fingerprint-radius', type=int, default=2)
    parser.add_argument('--fingerprint-size', type=int, default=2048)
    parser.add_argument(
        '--similarity-bins',
        default=','.join(str(value) for value in DEFAULT_SIMILARITY_BINS),
        help='Comma-separated nearest-train Tanimoto bin edges.',
    )
    parser.add_argument('--min-count', type=int, default=25)
    parser.add_argument('--max-train-rows', type=int, default=None)
    parser.add_argument('--max-eval-rows', type=int, default=None)
    parser.add_argument('--max-rows', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    """Run MassSpecGym split-gap diagnostics and print grouped summaries."""
    args = parse_args()
    quiet_rdkit_logs()

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    train_df = select_split(df, split=args.train_split, split_col=args.split_col)
    eval_df = select_split(
        df,
        split=args.eval_split,
        split_col=args.split_col,
        split_value=args.eval_split_value,
    )
    max_eval_rows = args.max_eval_rows or args.max_rows
    if args.max_train_rows:
        train_df = train_df.iloc[: args.max_train_rows].copy()
    if max_eval_rows:
        eval_df = eval_df.iloc[:max_eval_rows].copy()
    if train_df.empty:
        raise SystemExit('No training rows selected for gap diagnostics.')
    if eval_df.empty:
        raise SystemExit('No evaluation rows selected for gap diagnostics.')

    diagnostics = build_gap_diagnostics(
        train_df,
        eval_df,
        predictions=_read_predictions(args.predictions),
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_size=args.fingerprint_size,
        similarity_bins=_parse_float_sequence(args.similarity_bins),
        show_progress=args.progress,
    )
    summary = summarize_gap_diagnostics(diagnostics, min_count=args.min_count)
    _print_overview(train_df, eval_df, diagnostics)
    _print_summary(summary)
    if args.output:
        diagnostics.to_csv(args.output, index=False)
        print(f'Wrote row-level gap diagnostics to {args.output}')
    if args.summary_output:
        summary.to_csv(args.summary_output, index=False)
        print(f'Wrote gap diagnostic summary to {args.summary_output}')


def build_gap_diagnostics(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    fingerprint_radius: int = 2,
    fingerprint_size: int = 2048,
    similarity_bins: Sequence[float] = DEFAULT_SIMILARITY_BINS,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build row-level diagnostics for evaluation rows.

    Each evaluation row receives nearest-training-molecule Tanimoto similarity,
    same-SMILES/InChIKey overlap flags, optional same-molecule CE distances, and
    prediction metrics when a prediction CSV is supplied.
    """
    smiles_col = find_column(eval_df, SMILES_ALIASES)
    train_smiles_col = find_column(train_df, SMILES_ALIASES)
    train_molecules = _unique_train_molecules(
        train_df,
        smiles_col=train_smiles_col,
        fingerprint_radius=fingerprint_radius,
        fingerprint_size=fingerprint_size,
    )
    if train_molecules.empty:
        raise SystemExit('No valid training fingerprints for gap diagnostics.')
    eval_rows = _eval_molecule_rows(
        eval_df,
        smiles_col=smiles_col,
        fingerprint_radius=fingerprint_radius,
        fingerprint_size=fingerprint_size,
    )
    eval_rows = _attach_nearest_train_similarity(
        eval_rows,
        train_molecules,
        show_progress=show_progress,
    )
    eval_rows = _attach_metadata(eval_rows, eval_df)
    eval_rows = _attach_overlap_context(eval_rows, train_df)
    eval_rows['nearest_train_similarity_bin'] = _similarity_bin_labels(
        eval_rows['nearest_train_similarity'],
        bins=similarity_bins,
    )
    if 'collision_energy' in eval_rows:
        eval_rows['collision_energy_bin'] = _collision_energy_bin_labels(
            eval_rows['collision_energy']
        )
    if 'same_molecule_min_ce_delta' in eval_rows:
        eval_rows['same_molecule_ce_delta_bin'] = _ce_delta_bin_labels(
            eval_rows['same_molecule_min_ce_delta']
        )
    if predictions is not None:
        eval_rows = _merge_predictions(eval_rows, predictions)
    return eval_rows


def summarize_gap_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    min_count: int = 25,
) -> pd.DataFrame:
    """Summarize row-level diagnostics into interpretable grouped metrics."""
    group_specs = [
        ('nearest_train_similarity', ['nearest_train_similarity_bin']),
        ('same_inchikey_in_train', ['same_inchikey_in_train']),
        ('same_smiles_in_train', ['same_smiles_in_train']),
        ('instrument', ['instrument_type']),
        ('collision_energy_bin', ['collision_energy_bin']),
        (
            'instrument_x_collision_energy_bin',
            ['instrument_type', 'collision_energy_bin'],
        ),
        ('same_molecule_ce_delta', ['same_molecule_ce_delta_bin']),
    ]
    frames = []
    for group_type, columns in group_specs:
        if all(column in diagnostics.columns for column in columns):
            frame = _summarize_groups(
                diagnostics,
                columns,
                group_type=group_type,
                min_count=min_count,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_predictions(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _unique_train_molecules(
    train_df: pd.DataFrame,
    *,
    smiles_col: str,
    fingerprint_radius: int,
    fingerprint_size: int,
) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, row in train_df.iterrows():
        smiles = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        rows.append(
            {
                'train_smiles': smiles,
                'train_canonical_smiles': canonical,
                'train_inchikey': str(row.get('inchikey', '')),
                'fingerprint': _morgan_fingerprint(
                    mol,
                    radius=fingerprint_radius,
                    size=fingerprint_size,
                ),
            }
        )
    return pd.DataFrame(rows)


def _eval_molecule_rows(
    eval_df: pd.DataFrame,
    *,
    smiles_col: str,
    fingerprint_radius: int,
    fingerprint_size: int,
) -> pd.DataFrame:
    rows = []
    identifier_col = 'identifier' if 'identifier' in eval_df.columns else None
    for position, (_, row) in enumerate(eval_df.iterrows()):
        smiles = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smiles)
        canonical = (
            Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ''
        )
        rows.append(
            {
                'row_index': position,
                'identifier': str(row[identifier_col])
                if identifier_col is not None
                else str(position),
                'smiles': smiles,
                'canonical_smiles': canonical,
                'inchikey': str(row.get('inchikey', '')),
                'fingerprint': _morgan_fingerprint(
                    mol,
                    radius=fingerprint_radius,
                    size=fingerprint_size,
                )
                if mol is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


def _morgan_fingerprint(mol: Chem.Mol, *, radius: int, size: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(size),
    )
    return generator.GetFingerprint(mol)


def _attach_nearest_train_similarity(
    eval_rows: pd.DataFrame,
    train_molecules: pd.DataFrame,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    train_fps = list(train_molecules['fingerprint'])
    train_smiles = train_molecules['train_canonical_smiles'].to_numpy()
    train_inchikeys = train_molecules['train_inchikey'].to_numpy()
    similarities = []
    nearest_smiles = []
    nearest_inchikeys = []
    iterable = tqdm(
        eval_rows['fingerprint'],
        desc='nearest train molecule',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for fingerprint in iterable:
        if fingerprint is None:
            similarities.append(np.nan)
            nearest_smiles.append('')
            nearest_inchikeys.append('')
            continue
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps)
        if not values:
            similarities.append(np.nan)
            nearest_smiles.append('')
            nearest_inchikeys.append('')
            continue
        best_idx = int(np.argmax(values))
        similarities.append(float(values[best_idx]))
        nearest_smiles.append(str(train_smiles[best_idx]))
        nearest_inchikeys.append(str(train_inchikeys[best_idx]))
    out = eval_rows.drop(columns=['fingerprint']).copy()
    out['nearest_train_similarity'] = similarities
    out['nearest_train_smiles'] = nearest_smiles
    out['nearest_train_inchikey'] = nearest_inchikeys
    return out


def _attach_metadata(out: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    source = eval_df.reset_index(drop=True)
    out = out.copy()
    for output_name, aliases in (
        ('instrument_type', INSTRUMENT_ALIASES),
        ('collision_energy', CE_ALIASES),
    ):
        column = find_column(source, aliases, required=False)
        if column is None:
            continue
        out[output_name] = source[column].to_numpy()
    if 'collision_energy' in out:
        out['collision_energy'] = pd.to_numeric(
            out['collision_energy'],
            errors='coerce',
        )
    return out


def _attach_overlap_context(
    eval_rows: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    train = train_df.copy()
    train_smiles_col = find_column(train, SMILES_ALIASES)
    train['_canonical_smiles'] = [
        _canonical_smiles_or_empty(smiles) for smiles in train[train_smiles_col]
    ]
    train_inchikeys = (
        set(train['inchikey'].dropna().astype(str)) if 'inchikey' in train else set()
    )
    train_smiles = set(train['_canonical_smiles'])
    out = eval_rows.copy()
    out['same_smiles_in_train'] = out['canonical_smiles'].isin(train_smiles)
    out['same_inchikey_in_train'] = out['inchikey'].isin(train_inchikeys)
    if 'inchikey' not in train or 'collision_energy' not in out:
        return out

    ce_col = find_column(train, CE_ALIASES, required=False)
    instrument_col = find_column(train, INSTRUMENT_ALIASES, required=False)
    if ce_col is None:
        return out
    train['_collision_energy'] = pd.to_numeric(train[ce_col], errors='coerce')
    ce_by_inchikey = {
        str(key): group['_collision_energy'].dropna().to_numpy(dtype=float)
        for key, group in train.groupby('inchikey', dropna=False)
    }
    out['same_molecule_min_ce_delta'] = [
        _min_abs_delta(value, ce_by_inchikey.get(str(inchikey)))
        for inchikey, value in zip(
            out['inchikey'],
            out['collision_energy'],
            strict=False,
        )
    ]
    if instrument_col is None or 'instrument_type' not in out:
        return out
    ce_by_inchikey_instrument = {
        (str(inchikey), str(instrument)): group['_collision_energy']
        .dropna()
        .to_numpy(dtype=float)
        for (inchikey, instrument), group in train.groupby(
            ['inchikey', instrument_col],
            dropna=False,
        )
    }
    out['same_molecule_same_instrument_min_ce_delta'] = [
        _min_abs_delta(
            value,
            ce_by_inchikey_instrument.get((str(inchikey), str(instrument))),
        )
        for inchikey, instrument, value in zip(
            out['inchikey'],
            out['instrument_type'],
            out['collision_energy'],
            strict=False,
        )
    ]
    return out


def _canonical_smiles_or_empty(smiles: object) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ''


def _min_abs_delta(value: object, candidates: np.ndarray | None) -> float:
    if candidates is None or len(candidates) == 0:
        return float('nan')
    value = float(value)
    if not np.isfinite(value):
        return float('nan')
    return float(np.min(np.abs(candidates - value)))


def _similarity_bin_labels(values: pd.Series, *, bins: Sequence[float]) -> pd.Series:
    edges = sorted(float(value) for value in bins)
    if edges[0] > 0.0:
        edges.insert(0, 0.0)
    if edges[-1] < 1.0:
        edges.append(1.0)
    labels = [f'[{edges[i]:.2f},{edges[i + 1]:.2f})' for i in range(len(edges) - 1)]
    labels[-1] = labels[-1].replace(')', ']')
    cut_edges = [*edges]
    cut_edges[-1] = float(np.nextafter(cut_edges[-1], np.inf))
    return (
        pd.cut(
            pd.to_numeric(values, errors='coerce'),
            bins=cut_edges,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        .astype('object')
        .fillna('missing')
    )


def _collision_energy_bin_labels(values: pd.Series) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors='coerce'),
            bins=DEFAULT_CE_BINS,
            labels=DEFAULT_CE_LABELS,
            include_lowest=True,
        )
        .astype('object')
        .fillna('missing')
    )


def _ce_delta_bin_labels(values: pd.Series) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(values, errors='coerce'),
            bins=(-np.inf, 0.0, 5.0, 15.0, np.inf),
            labels=('0', '(0,5]', '(5,15]', '>15'),
            include_lowest=True,
        )
        .astype('object')
        .fillna('no_same_molecule')
    )


def _merge_predictions(
    diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [column for column in METRIC_COLUMNS if column in predictions.columns]
    if not metric_cols:
        return diagnostics
    if 'identifier' in predictions.columns:
        cols = ['identifier', *metric_cols]
        out = diagnostics.merge(predictions[cols], on='identifier', how='left')
        return _ensure_gap_metrics(out)
    if len(predictions) != len(diagnostics):
        raise ValueError(
            'Predictions do not contain identifier and row count does not match '
            f'({len(predictions)} != {len(diagnostics)}).'
        )
    out = diagnostics.copy()
    for column in metric_cols:
        out[column] = predictions[column].to_numpy()
    return _ensure_gap_metrics(out)


def _ensure_gap_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute support/scorer/OOS gap columns when source metrics exist."""
    out = df.copy()
    if 'oracle_binned_cosine' in out and 'support_gap' not in out:
        out['support_gap'] = 1.0 - out['oracle_binned_cosine']
    if {'oracle_binned_cosine', 'cosine'} <= set(
        out.columns
    ) and 'scorer_gap' not in out:
        out['scorer_gap'] = out['oracle_binned_cosine'] - out['cosine']
    if {'oracle_tolerance_cosine', 'cosine'} <= set(
        out.columns
    ) and 'tolerance_scorer_gap' not in out:
        out['tolerance_scorer_gap'] = out['oracle_tolerance_cosine'] - out['cosine']
    if {'predicted_oos_probability', 'oos_target_mass'} <= set(
        out.columns
    ) and 'oos_calibration_error' not in out:
        out['oos_calibration_error'] = (
            out['predicted_oos_probability'] - out['oos_target_mass']
        )
    if 'oos_calibration_error' in out and 'oos_calibration_abs_error' not in out:
        out['oos_calibration_abs_error'] = out['oos_calibration_error'].abs()
    return out


def _summarize_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    group_type: str,
    min_count: int,
) -> pd.DataFrame:
    rows = []
    metric_cols = [column for column in METRIC_COLUMNS if column in df.columns]
    for raw_keys, group in df.groupby(group_cols, dropna=False, observed=False):
        keys = raw_keys if isinstance(raw_keys, tuple) else (raw_keys,)
        n = int(len(group))
        if n < int(min_count):
            continue
        row = {
            'group_type': group_type,
            'group': ' | '.join(str(value) for value in keys),
            'n': n,
            'nearest_train_similarity_mean': float(
                group['nearest_train_similarity'].mean()
            ),
            'same_inchikey_fraction': float(group['same_inchikey_in_train'].mean()),
            'same_smiles_fraction': float(group['same_smiles_in_train'].mean()),
        }
        if 'collision_energy' in group:
            ce = pd.to_numeric(group['collision_energy'], errors='coerce')
            row['collision_energy_mean'] = float(ce.mean())
        if 'same_molecule_min_ce_delta' in group:
            row['same_molecule_min_ce_delta_mean'] = float(
                group['same_molecule_min_ce_delta'].mean()
            )
        for column in metric_cols:
            row[f'{column}_mean'] = float(group[column].mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_col = 'cosine_mean' if 'cosine_mean' in out.columns else 'n'
    return out.sort_values([sort_col, 'n'], ascending=[True, False])


def _print_overview(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    print(
        'gap diagnostics: '
        f'train_rows={len(train_df)} eval_rows={len(eval_df)} '
        f'eval_same_smiles={diagnostics["same_smiles_in_train"].mean():.2%} '
        f'eval_same_inchikey={diagnostics["same_inchikey_in_train"].mean():.2%} '
        f'nearest_similarity_mean={diagnostics["nearest_train_similarity"].mean():.4f}'
    )
    if 'cosine' in diagnostics:
        coverage = diagnostics.get('candidate_coverage', pd.Series(dtype=float)).mean()
        support_gap = diagnostics.get('support_gap', pd.Series(dtype=float)).mean()
        scorer_gap = diagnostics.get('scorer_gap', pd.Series(dtype=float)).mean()
        print(
            f'eval_cosine_mean={diagnostics["cosine"].mean():.5f} '
            f'eval_coverage_mean={coverage:.5f} '
            f'eval_support_gap_mean={support_gap:.5f} '
            f'eval_scorer_gap_mean={scorer_gap:.5f}'
        )


def _print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print('No diagnostic groups passed the minimum-count threshold.')
        return
    display_cols = [
        'group',
        'n',
        'cosine_mean',
        'candidate_coverage_mean',
        'oos_target_mass_mean',
        'oracle_binned_cosine_mean',
        'oracle_tolerance_cosine_mean',
        'support_gap_mean',
        'scorer_gap_mean',
        'tolerance_scorer_gap_mean',
        'oos_calibration_abs_error_mean',
        'nearest_train_similarity_mean',
        'same_inchikey_fraction',
        'same_molecule_min_ce_delta_mean',
        'collision_energy_mean',
    ]
    formatters = {
        column: (lambda value: f'{value:.5f}')
        for column in display_cols
        if column.endswith('_mean')
        or column.endswith('_fraction')
        or column == 'cosine_mean'
    }
    print('Gap diagnostic summary, lowest cosine first:')
    for group_type, group in summary.groupby('group_type', sort=False):
        cols = [column for column in display_cols if column in group.columns]
        print(f'\n{group_type}:')
        print(group[cols].to_string(index=False, formatters=formatters))


def _parse_float_sequence(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(',') if part.strip())
    if len(values) < 2:
        raise argparse.ArgumentTypeError('At least two bin edges are required.')
    return values


if __name__ == '__main__':
    main()
