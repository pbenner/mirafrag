from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mirafrag.data import (
    SMILES_ALIASES,
    filter_massspecgym_simulation,
    find_column,
    read_table,
    select_split,
)
from mirafrag.sparse_spectra import (
    SparseSpectrum,
    combine_sparse_spectra,
    scale_sparse_spectrum,
    sparse_cosine,
    sparse_from_peaks,
)
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX, parse_peaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mirafrag-ensemble-eval',
        description='Evaluate weighted ensembles of exported MiraFrag prediction CSVs.',
    )
    parser.add_argument(
        '-i', '--input', required=True, help='MassSpecGym TSV/CSV path.'
    )
    parser.add_argument(
        '-p',
        '--predictions',
        nargs='+',
        required=True,
        help='Prediction CSV paths. Values may also be comma-separated.',
    )
    parser.add_argument('--prediction-names', default=None)
    parser.add_argument('-o', '--output', default=None, help='Optional row CSV output.')
    parser.add_argument('--summary-output', default=None)
    parser.add_argument('--split', default='val')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument(
        '--weights-grid',
        default='auto',
        help=(
            "Weight vectors. Use 'auto', or comma-separated vectors with ':' "
            "between model weights, e.g. '0:1,0.1:0.9,0.2:0.8'."
        ),
    )
    parser.add_argument(
        '--weights',
        default=None,
        help="Evaluate a single ':'-separated weight vector, e.g. '0.3:0.7'.",
    )
    parser.add_argument(
        '--massspecgym-filter', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--progress', action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = _parse_prediction_paths(args.predictions)
    prediction_names = _prediction_names(prediction_paths, args.prediction_names)
    prediction_tables = [pd.read_csv(path) for path in prediction_paths]

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    eval_df = select_split(
        df,
        split=args.split,
        split_col=args.split_col,
        split_value=args.split_value,
    )
    if args.max_rows:
        eval_df = eval_df.iloc[: args.max_rows].copy()
    if eval_df.empty:
        raise SystemExit('No rows selected for ensemble evaluation.')

    weight_grid = _parse_weight_grid(
        args.weights if args.weights is not None else args.weights_grid,
        n_models=len(prediction_tables),
        single=args.weights is not None,
    )
    rows, summary = run_ensemble_eval(
        eval_df,
        prediction_tables,
        prediction_names=prediction_names,
        weight_grid=weight_grid,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        show_progress=args.progress,
    )
    _print_summary(summary)
    if args.output:
        rows.to_csv(args.output, index=False)
        print(f'Wrote ensemble rows to {args.output}')
    if args.summary_output:
        summary.to_csv(args.summary_output, index=False)
        print(f'Wrote ensemble summary to {args.summary_output}')


def run_ensemble_eval(
    eval_df: pd.DataFrame,
    predictions: Sequence[pd.DataFrame],
    *,
    prediction_names: Sequence[str] | None = None,
    weight_grid: Sequence[Sequence[float]] | None = None,
    mz_max: float = MASS_SPEC_GYM_MZ_MAX,
    bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions:
        raise ValueError('At least one prediction table is required.')
    names = _unique_names(
        list(prediction_names)
        if prediction_names is not None
        else [f'model{i + 1}' for i in range(len(predictions))]
    )
    if len(names) != len(predictions):
        raise ValueError('prediction_names must match the number of prediction tables.')
    for name, table in zip(names, predictions):
        if 'pred_peaks' not in table.columns:
            raise ValueError(f'Prediction table {name!r} is missing pred_peaks.')
    weights = (
        list(weight_grid)
        if weight_grid is not None
        else _auto_weight_grid(len(predictions))
    )
    weights = [_normalize_weight_vector(vector, len(predictions)) for vector in weights]

    smiles_col = find_column(eval_df, SMILES_ALIASES, required=False)
    lookups = [_prediction_lookup(table) for table in predictions]
    row_records: list[dict[str, Any]] = []
    progress = tqdm(
        list(eval_df.iterrows()),
        desc='ensemble spectra',
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for position, (idx, row) in enumerate(progress):
        identifier = _identifier(row, idx)
        target = _row_spectrum(row, mz_max=mz_max, bin_width=bin_width)
        spectra = [
            _prediction_spectrum(
                table,
                lookup,
                identifier=identifier,
                position=position,
                mz_max=mz_max,
                bin_width=bin_width,
                name=name,
            )
            for table, lookup, name in zip(predictions, lookups, names)
        ]
        record: dict[str, Any] = {'identifier': identifier}
        if smiles_col is not None:
            record['smiles'] = str(row[smiles_col])
        for name, spectrum in zip(names, spectra):
            record[f'{name}_cosine'] = sparse_cosine(spectrum, target)
            record[f'{name}_sqrt_cosine'] = sparse_cosine(spectrum, target, sqrt=True)
        for vector in weights:
            label = _weight_label(vector)
            ensemble = _weighted_sum(spectra, vector)
            record[f'ensemble_cosine_{label}'] = sparse_cosine(ensemble, target)
            record[f'ensemble_sqrt_cosine_{label}'] = sparse_cosine(
                ensemble,
                target,
                sqrt=True,
            )
        row_records.append(record)

    rows = pd.DataFrame(row_records)
    summary = _summarize_rows(rows, prediction_names=names, weight_grid=weights)
    return rows, summary


def _weighted_sum(
    spectra: Sequence[SparseSpectrum], weights: Sequence[float]
) -> SparseSpectrum:
    return combine_sparse_spectra(
        [
            scale_sparse_spectrum(spectrum, weight)
            for spectrum, weight in zip(spectra, weights)
        ]
    )


def _row_spectrum(row: pd.Series, *, mz_max: float, bin_width: float) -> SparseSpectrum:
    mzs, intensities = parse_peaks(row)
    return sparse_from_peaks(mzs, intensities, mz_max=mz_max, bin_width=bin_width)


def _prediction_lookup(predictions: pd.DataFrame) -> dict[str, int] | None:
    if 'identifier' not in predictions.columns:
        return None
    return {
        str(identifier): int(i)
        for i, identifier in enumerate(predictions['identifier'])
    }


def _prediction_spectrum(
    predictions: pd.DataFrame,
    lookup: dict[str, int] | None,
    *,
    identifier: str,
    position: int,
    mz_max: float,
    bin_width: float,
    name: str,
) -> SparseSpectrum:
    if lookup is not None and identifier in lookup:
        row = predictions.iloc[lookup[identifier]]
    elif position < len(predictions):
        row = predictions.iloc[position]
    else:
        raise RuntimeError(f'Prediction table {name!r} has no row {position}.')
    peaks = row['pred_peaks']
    if isinstance(peaks, str):
        peaks = json.loads(peaks)
    return sparse_from_peaks(
        peaks.get('mz', peaks.get('mzs', [])),
        peaks.get('intensity', peaks.get('intensities', [])),
        mz_max=mz_max,
        bin_width=bin_width,
    )


def _parse_prediction_paths(values: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        paths.extend(part.strip() for part in str(value).split(',') if part.strip())
    if not paths:
        raise ValueError('At least one prediction path is required.')
    return paths


def _prediction_names(paths: Sequence[str], names_text: str | None) -> list[str]:
    if names_text:
        names = [part.strip() for part in names_text.split(',') if part.strip()]
    else:
        names = [_sanitize_name(Path(path).stem) for path in paths]
    return _unique_names(names)


def _parse_weight_grid(
    text: str, *, n_models: int, single: bool = False
) -> list[tuple[float, ...]]:
    if text is None or str(text).strip().lower() == 'auto':
        return _auto_weight_grid(n_models)
    raw = str(text).strip()
    pieces = (
        [raw] if single else [part.strip() for part in raw.replace(';', ',').split(',')]
    )
    vectors = []
    for piece in pieces:
        if not piece:
            continue
        vector = tuple(float(part) for part in piece.split(':') if part != '')
        vectors.append(_normalize_weight_vector(vector, n_models))
    if not vectors:
        raise ValueError('No valid ensemble weight vectors parsed.')
    return vectors


def _auto_weight_grid(n_models: int) -> list[tuple[float, ...]]:
    if n_models <= 0:
        raise ValueError('n_models must be positive.')
    if n_models == 1:
        return [(1.0,)]
    if n_models == 2:
        return [(round(i / 20.0, 4), round(1.0 - i / 20.0, 4)) for i in range(21)]
    vectors: list[tuple[float, ...]] = []
    for i in range(n_models):
        vector = [0.0] * n_models
        vector[i] = 1.0
        vectors.append(tuple(vector))
    vectors.append(tuple([1.0 / n_models] * n_models))
    return vectors


def _normalize_weight_vector(
    vector: Sequence[float], n_models: int
) -> tuple[float, ...]:
    if len(vector) != n_models:
        raise ValueError(
            f'Weight vector length {len(vector)} does not match prediction count {n_models}.',
        )
    arr = np.asarray(vector, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError('Weights must be finite and non-negative.')
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError('At least one ensemble weight must be positive.')
    arr = arr / total
    return tuple(float(x) for x in arr)


def _summarize_rows(
    rows: pd.DataFrame,
    *,
    prediction_names: Sequence[str],
    weight_grid: Sequence[Sequence[float]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name in prediction_names:
        records.append(
            {
                'mode': 'prediction',
                'weights': name,
                'cosine_mean': _mean(rows[f'{name}_cosine']),
                'sqrt_cosine_mean': _mean(rows[f'{name}_sqrt_cosine']),
            }
        )
    for vector in weight_grid:
        label = _weight_label(vector)
        records.append(
            {
                'mode': 'ensemble',
                'weights': ':'.join(_format_float(value) for value in vector),
                'cosine_mean': _mean(rows[f'ensemble_cosine_{label}']),
                'sqrt_cosine_mean': _mean(rows[f'ensemble_sqrt_cosine_{label}']),
            }
        )
    return pd.DataFrame(records).sort_values(
        ['cosine_mean', 'mode'],
        ascending=[False, True],
        ignore_index=True,
    )


def _print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print('No ensemble summary rows.')
        return
    print('Ensemble summary, best cosine first:')
    for row in summary.itertuples(index=False):
        print(
            f'mode={row.mode} weights={row.weights} '
            f'cosine_mean={row.cosine_mean:.5f} '
            f'sqrt_cosine_mean={row.sqrt_cosine_mean:.5f}'
        )


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float('nan')


def _identifier(row: pd.Series, idx) -> str:
    return str(row.get('identifier', idx))


def _sanitize_name(value: str) -> str:
    out = ''.join(ch if ch.isalnum() else '_' for ch in value.strip())
    out = '_'.join(part for part in out.split('_') if part)
    return out or 'model'


def _unique_names(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        base = _sanitize_name(str(raw))
        count = counts.get(base, 0)
        counts[base] = count + 1
        out.append(base if count == 0 else f'{base}_{count + 1}')
    return out


def _weight_label(weights: Sequence[float]) -> str:
    return '_'.join(_format_float(value).replace('.', 'p') for value in weights)


def _format_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f'{value:.4f}'.rstrip('0').rstrip('.')


if __name__ == '__main__':
    main()
