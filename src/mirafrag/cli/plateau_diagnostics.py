from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from mirafrag.data import (
    CE_ALIASES,
    INSTRUMENT_ALIASES,
    SMILES_ALIASES,
    filter_massspecgym_simulation,
    find_column,
    read_table,
    select_split,
)

DEFAULT_CE_BINS = (-np.inf, 20.0, 35.0, 60.0, np.inf)
DEFAULT_CE_LABELS = ('<=20', '(20,35]', '(35,60]', '>60')
DEFAULT_QUALITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
DEFAULT_QUALITY_LABELS = ('[0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1]')

PRIMARY_METRICS = (
    'cosine',
    'candidate_coverage',
    'oracle_binned_cosine',
    'oracle_tolerance_cosine',
    'support_gap',
    'scorer_gap',
    'tolerance_scorer_gap',
    'predicted_oos_probability',
    'oos_target_mass',
    'oos_calibration_error',
    'oos_calibration_abs_error',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mirafrag-plateau-diagnostics',
        description=(
            'Explain MiraFrag validation/test plateaus from exported prediction CSVs: '
            'support headroom, scorer headroom, OOS calibration, metadata regimes, '
            'and cross-checkpoint complementarity.'
        ),
    )
    parser.add_argument(
        '-i', '--input', required=True, help='MassSpecGym TSV/CSV path.'
    )
    parser.add_argument(
        '-p',
        '--predictions',
        nargs='+',
        required=True,
        help='One or more prediction CSVs. Values may also be comma-separated.',
    )
    parser.add_argument('--prediction-names', default=None)
    parser.add_argument('--split', default='val')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument(
        '--massspecgym-filter', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--min-count', type=int, default=25)
    parser.add_argument('--output', default=None, help='Optional row-level output CSV.')
    parser.add_argument(
        '--summary-output', default=None, help='Optional grouped summary CSV.'
    )
    parser.add_argument(
        '--models-output',
        default=None,
        help='Optional model/complementarity summary CSV.',
    )
    parser.add_argument(
        '--winner-summary-output',
        default=None,
        help='Optional summary of structural/diagnostic differences between rowwise winning models.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = _parse_prediction_paths(args.predictions)
    prediction_names = _prediction_names(prediction_paths, args.prediction_names)
    predictions = [pd.read_csv(path) for path in prediction_paths]

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
        raise SystemExit('No evaluation rows selected.')

    rows, group_summary, model_summary, winner_summary = run_plateau_diagnostics(
        eval_df,
        predictions,
        prediction_names=prediction_names,
        min_count=args.min_count,
    )
    _print_model_summary(model_summary)
    _print_group_summary(group_summary)
    _print_winner_summary(winner_summary)

    if args.output:
        rows.to_csv(args.output, index=False)
        print(f'Wrote plateau row diagnostics to {args.output}')
    if args.summary_output:
        group_summary.to_csv(args.summary_output, index=False)
        print(f'Wrote plateau group summary to {args.summary_output}')
    if args.models_output:
        model_summary.to_csv(args.models_output, index=False)
        print(f'Wrote plateau model summary to {args.models_output}')
    if args.winner_summary_output:
        winner_summary.to_csv(args.winner_summary_output, index=False)
        print(f'Wrote plateau winner summary to {args.winner_summary_output}')


def run_plateau_diagnostics(
    eval_df: pd.DataFrame,
    predictions: Sequence[pd.DataFrame],
    *,
    prediction_names: Sequence[str],
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not predictions:
        raise ValueError('At least one prediction table is required.')
    rows = _base_rows(eval_df)
    model_metric_frames = []
    for name, prediction in zip(prediction_names, predictions, strict=True):
        merged = _align_prediction(rows, prediction, name=name)
        rows = rows.merge(merged, on=['row_index', 'identifier', 'smiles'], how='left')
        model_metric_frames.append(_model_metrics(rows, name))

    _attach_oracle_selection(rows, prediction_names)
    _attach_failure_labels(rows, prediction_names[0])
    group_summary = _summarize_groups(rows, prediction_names, min_count=min_count)
    winner_summary = _summarize_winner_regimes(
        rows, prediction_names, min_count=min_count
    )
    model_summary = pd.DataFrame(model_metric_frames)
    if len(prediction_names) > 1:
        model_summary = pd.concat(
            [model_summary, _complementarity_summary(rows, prediction_names)],
            ignore_index=True,
        )
    return rows, group_summary, model_summary, winner_summary


def _base_rows(eval_df: pd.DataFrame) -> pd.DataFrame:
    source = eval_df.reset_index(drop=True)
    identifier_col = 'identifier' if 'identifier' in source.columns else None
    smiles_col = find_column(source, SMILES_ALIASES)
    instrument_col = find_column(source, INSTRUMENT_ALIASES, required=False)
    ce_col = find_column(source, CE_ALIASES, required=False)
    out = pd.DataFrame(
        {
            'row_index': np.arange(len(source), dtype=np.int64),
            'identifier': source[identifier_col].astype(str).to_numpy()
            if identifier_col is not None
            else np.arange(len(source)).astype(str),
            'smiles': source[smiles_col].astype(str).to_numpy(),
        }
    )
    if instrument_col is not None:
        out['instrument_type'] = source[instrument_col].astype(str).to_numpy()
    if ce_col is not None:
        out['collision_energy'] = pd.to_numeric(
            source[ce_col], errors='coerce'
        ).to_numpy()
        out['collision_energy_bin'] = pd.cut(
            out['collision_energy'],
            bins=DEFAULT_CE_BINS,
            labels=DEFAULT_CE_LABELS,
            include_lowest=True,
        ).astype(str)
    descriptor_rows = [_molecule_descriptors(smiles) for smiles in out['smiles']]
    if descriptor_rows:
        out = pd.concat([out, pd.DataFrame(descriptor_rows)], axis=1)
    return out


def _align_prediction(
    base: pd.DataFrame, prediction: pd.DataFrame, *, name: str
) -> pd.DataFrame:
    pred = prediction.copy()
    if 'identifier' not in pred.columns:
        raise ValueError(f'{name}: prediction CSV must contain identifier.')
    if 'smiles' not in pred.columns:
        pred['smiles'] = ''
    keep = [
        'identifier',
        'smiles',
        *[column for column in PRIMARY_METRICS if column in pred.columns],
    ]
    pred = pred[keep].copy()
    pred['identifier'] = pred['identifier'].astype(str)
    pred['smiles'] = pred['smiles'].astype(str)
    merged = base[['row_index', 'identifier', 'smiles']].merge(
        pred,
        on=['identifier', 'smiles'],
        how='left',
        validate='one_to_one',
    )
    missing = int(merged['cosine'].isna().sum()) if 'cosine' in merged else len(merged)
    if missing:
        fallback = base[['row_index', 'identifier']].merge(
            pred.drop(columns=['smiles']),
            on='identifier',
            how='left',
            validate='one_to_one',
            suffixes=('', '_fallback'),
        )
        for column in PRIMARY_METRICS:
            if column in merged.columns and column in fallback.columns:
                merged[column] = merged[column].fillna(fallback[column])
    rename = {
        column: f'{name}_{column}'
        for column in PRIMARY_METRICS
        if column in merged.columns
    }
    return merged.rename(columns=rename)


def _model_metrics(rows: pd.DataFrame, name: str) -> dict[str, object]:
    cosine = _col(rows, name, 'cosine')
    oracle = _col(rows, name, 'oracle_binned_cosine')
    tolerance = _col(rows, name, 'oracle_tolerance_cosine')
    support_gap = _col(rows, name, 'support_gap')
    scorer_gap = _col(rows, name, 'scorer_gap')
    oos_error = _col(rows, name, 'oos_calibration_abs_error')
    out = {
        'model': name,
        'kind': 'single',
        'n': int(cosine.notna().sum()),
        'cosine_mean': _mean(cosine),
        'cosine_p10': _quantile(cosine, 0.10),
        'cosine_p50': _quantile(cosine, 0.50),
        'cosine_p90': _quantile(cosine, 0.90),
        'oracle_binned_mean': _mean(oracle),
        'oracle_tolerance_mean': _mean(tolerance),
        'support_gap_mean': _mean(support_gap),
        'scorer_gap_mean': _mean(scorer_gap),
        'oos_abs_error_mean': _mean(oos_error),
    }
    if np.isfinite(out['oracle_binned_mean']) and np.isfinite(out['cosine_mean']):
        out['used_oracle_fraction'] = out['cosine_mean'] / max(
            out['oracle_binned_mean'], 1e-12
        )
    else:
        out['used_oracle_fraction'] = np.nan
    return out


def _attach_oracle_selection(rows: pd.DataFrame, names: Sequence[str]) -> None:
    cosine_cols = [f'{name}_cosine' for name in names if f'{name}_cosine' in rows]
    if not cosine_cols:
        return
    values = rows[cosine_cols].to_numpy(dtype=float)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, -np.inf)
    any_finite = finite.any(axis=1)
    rows['rowwise_best_model_cosine'] = np.where(
        any_finite, np.max(safe_values, axis=1), np.nan
    )
    best_indices = np.argmax(safe_values, axis=1)
    rows['rowwise_best_model'] = [
        names[int(idx)] if ok else 'missing'
        for idx, ok in zip(best_indices, any_finite, strict=True)
    ]
    if len(cosine_cols) >= 2:
        sorted_values = np.sort(safe_values, axis=1)
        margins = sorted_values[:, -1] - sorted_values[:, -2]
        rows['rowwise_best_margin'] = np.where(any_finite, margins, np.nan)


def _attach_failure_labels(rows: pd.DataFrame, name: str) -> None:
    cosine = _col(rows, name, 'cosine')
    support_gap = _col(rows, name, 'support_gap')
    scorer_gap = _col(rows, name, 'scorer_gap')
    oos_error = _col(rows, name, 'oos_calibration_abs_error')
    rows[f'{name}_cosine_bin'] = pd.cut(
        cosine,
        bins=DEFAULT_QUALITY_BINS,
        labels=DEFAULT_QUALITY_LABELS,
        include_lowest=True,
    ).astype(str)
    dominant = []
    for sg, cg, oe in zip(support_gap, scorer_gap, oos_error, strict=False):
        values = {'support': sg, 'scorer': cg, 'oos': oe}
        finite = {key: value for key, value in values.items() if np.isfinite(value)}
        dominant.append(max(finite, key=finite.get) if finite else 'unknown')
    rows[f'{name}_dominant_error'] = dominant


def _molecule_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            'mol_descriptor_valid': 0.0,
            'mol_num_atoms': np.nan,
            'mol_num_bonds': np.nan,
            'mol_exact_mass': np.nan,
        }
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    hetero_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() not in {1, 6})
    aromatic_atoms = sum(1 for atom in atoms if atom.GetIsAromatic())
    ring_bonds = sum(1 for bond in bonds if bond.IsInRing())
    aromatic_bonds = sum(1 for bond in bonds if bond.GetIsAromatic())
    hetero_bonds = sum(
        1
        for bond in bonds
        if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() not in {1, 6}
        or mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() not in {1, 6}
    )
    carbon_hetero_bonds = sum(
        1
        for bond in bonds
        if {
            mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 6,
            mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 6,
        }
        == {True, False}
    )
    num_atoms = max(len(atoms), 1)
    num_bonds = max(len(bonds), 1)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    max_ring_size = max((len(ring) for ring in atom_rings), default=0)
    return {
        'mol_descriptor_valid': 1.0,
        'mol_num_atoms': float(mol.GetNumAtoms()),
        'mol_num_heavy_atoms': float(mol.GetNumHeavyAtoms()),
        'mol_num_bonds': float(mol.GetNumBonds()),
        'mol_exact_mass': float(Descriptors.ExactMolWt(mol)),
        'mol_tpsa': float(rdMolDescriptors.CalcTPSA(mol)),
        'mol_logp': float(Crippen.MolLogP(mol)),
        'mol_hba': float(Lipinski.NumHAcceptors(mol)),
        'mol_hbd': float(Lipinski.NumHDonors(mol)),
        'mol_rotatable_bonds': float(Lipinski.NumRotatableBonds(mol)),
        'mol_fraction_sp3': float(rdMolDescriptors.CalcFractionCSP3(mol)),
        'mol_num_rings': float(ring_info.NumRings()),
        'mol_aromatic_rings': float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        'mol_aliphatic_rings': float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        'mol_max_ring_size': float(max_ring_size),
        'mol_hetero_atoms': float(hetero_atoms),
        'mol_hetero_fraction': float(hetero_atoms) / float(num_atoms),
        'mol_aromatic_atoms': float(aromatic_atoms),
        'mol_aromatic_fraction': float(aromatic_atoms) / float(num_atoms),
        'mol_ring_bonds': float(ring_bonds),
        'mol_ring_bond_fraction': float(ring_bonds) / float(num_bonds),
        'mol_aromatic_bonds': float(aromatic_bonds),
        'mol_aromatic_bond_fraction': float(aromatic_bonds) / float(num_bonds),
        'mol_hetero_bonds': float(hetero_bonds),
        'mol_hetero_bond_fraction': float(hetero_bonds) / float(num_bonds),
        'mol_carbon_hetero_bonds': float(carbon_hetero_bonds),
        'mol_formal_charge': float(sum(atom.GetFormalCharge() for atom in atoms)),
        'mol_chiral_centers': float(len(chiral_centers)),
    }


def _summarize_winner_regimes(
    rows: pd.DataFrame, names: Sequence[str], *, min_count: int
) -> pd.DataFrame:
    if len(names) < 2 or 'rowwise_best_model' not in rows:
        return pd.DataFrame()
    winners = [
        name
        for name in names
        if (rows['rowwise_best_model'] == name).sum() >= int(min_count)
    ]
    if len(winners) < 2:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    numeric_columns = [
        column
        for column in rows.columns
        if column.startswith('mol_')
        or column in {'collision_energy'}
        or any(
            column == f'{name}_{metric}'
            for name in names
            for metric in (
                'candidate_coverage',
                'oracle_binned_cosine',
                'support_gap',
                'scorer_gap',
                'predicted_oos_probability',
                'oos_target_mass',
                'oos_calibration_abs_error',
            )
        )
    ]
    reference = winners[0]
    comparator = winners[1]
    reference_mask = rows['rowwise_best_model'] == reference
    comparator_mask = rows['rowwise_best_model'] == comparator
    for column in numeric_columns:
        values = pd.to_numeric(rows[column], errors='coerce')
        ref = values[reference_mask].dropna()
        cmp = values[comparator_mask].dropna()
        if len(ref) < int(min_count) or len(cmp) < int(min_count):
            continue
        ref_mean = float(ref.mean())
        cmp_mean = float(cmp.mean())
        pooled = float(values[reference_mask | comparator_mask].std(ddof=0))
        records.append(
            {
                'summary_type': 'numeric_feature',
                'feature': column,
                f'{reference}_mean': ref_mean,
                f'{comparator}_mean': cmp_mean,
                f'{comparator}_minus_{reference}': cmp_mean - ref_mean,
                'abs_standardized_difference': abs(cmp_mean - ref_mean) / pooled
                if pooled > 0
                else 0.0,
                f'{reference}_n': int(len(ref)),
                f'{comparator}_n': int(len(cmp)),
            }
        )
    for column in ('instrument_type', 'collision_energy_bin'):
        if column not in rows:
            continue
        counts = pd.crosstab(
            rows[column], rows['rowwise_best_model'], normalize='index'
        )
        totals = rows[column].value_counts()
        for value, row in counts.iterrows():
            if int(totals.get(value, 0)) < int(min_count):
                continue
            record = {
                'summary_type': 'categorical_fraction',
                'feature': column,
                'value': value,
                'n': int(totals[value]),
            }
            for name in names:
                record[f'{name}_fraction'] = float(row.get(name, 0.0))
            records.append(record)
    return pd.DataFrame(records)


def _summarize_groups(
    rows: pd.DataFrame, names: Sequence[str], *, min_count: int
) -> pd.DataFrame:
    group_specs = [('all', []), ('cosine_bin', [f'{names[0]}_cosine_bin'])]
    if 'instrument_type' in rows:
        group_specs.append(('instrument', ['instrument_type']))
    if 'collision_energy_bin' in rows:
        group_specs.append(('collision_energy_bin', ['collision_energy_bin']))
    if {'instrument_type', 'collision_energy_bin'}.issubset(rows.columns):
        group_specs.append(
            (
                'instrument_x_collision_energy_bin',
                ['instrument_type', 'collision_energy_bin'],
            )
        )
    group_specs.append(('dominant_error', [f'{names[0]}_dominant_error']))

    frames = []
    for group_type, columns in group_specs:
        if columns and not all(column in rows for column in columns):
            continue
        grouped = (
            [(('all',), rows)]
            if not columns
            else rows.groupby(columns, dropna=False, observed=False)
        )
        for key, group in grouped:
            if len(group) < int(min_count):
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            record: dict[str, object] = {
                'group_type': group_type,
                'group': 'all'
                if not columns
                else ' | '.join(str(item) for item in key_tuple),
                'n': int(len(group)),
            }
            for name in names:
                for metric in (
                    'cosine',
                    'oracle_binned_cosine',
                    'oracle_tolerance_cosine',
                    'support_gap',
                    'scorer_gap',
                    'oos_calibration_abs_error',
                ):
                    column = f'{name}_{metric}'
                    if column in group:
                        record[f'{name}_{metric}_mean'] = _mean(group[column])
            if 'rowwise_best_model_cosine' in group:
                record['rowwise_best_model_cosine_mean'] = _mean(
                    group['rowwise_best_model_cosine']
                )
                first_cos = f'{names[0]}_cosine'
                if first_cos in group:
                    record['rowwise_best_gain_mean'] = _mean(
                        group['rowwise_best_model_cosine'] - group[first_cos]
                    )
            frames.append(record)
    return pd.DataFrame(frames)


def _complementarity_summary(rows: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    first = names[0]
    records = []
    if 'rowwise_best_model_cosine' in rows and f'{first}_cosine' in rows:
        records.append(
            {
                'model': 'rowwise_best',
                'kind': 'oracle_selection',
                'n': int(rows['rowwise_best_model_cosine'].notna().sum()),
                'cosine_mean': _mean(rows['rowwise_best_model_cosine']),
                'gain_vs_first': _mean(
                    rows['rowwise_best_model_cosine'] - rows[f'{first}_cosine']
                ),
                'cosine_p10': _quantile(rows['rowwise_best_model_cosine'], 0.10),
                'cosine_p50': _quantile(rows['rowwise_best_model_cosine'], 0.50),
                'cosine_p90': _quantile(rows['rowwise_best_model_cosine'], 0.90),
            }
        )
    if 'rowwise_best_model' in rows:
        counts = rows['rowwise_best_model'].value_counts(normalize=True)
        for name in names:
            records.append(
                {
                    'model': f'best_fraction:{name}',
                    'kind': 'oracle_selection_fraction',
                    'n': int((rows['rowwise_best_model'] == name).sum()),
                    'cosine_mean': float(counts.get(name, 0.0)),
                }
            )
    return pd.DataFrame(records)


def _parse_prediction_paths(values: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        paths.extend(part.strip() for part in str(value).split(',') if part.strip())
    if not paths:
        raise SystemExit('At least one prediction CSV is required.')
    return paths


def _prediction_names(paths: Sequence[str], value: str | None) -> list[str]:
    if value:
        names = [part.strip() for part in value.split(',') if part.strip()]
        if len(names) != len(paths):
            raise SystemExit('--prediction-names must match --predictions length.')
        return _unique_names(names)
    return _unique_names([Path(path).stem for path in paths])


def _unique_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        safe = (
            ''.join(ch if ch.isalnum() else '_' for ch in str(name)).strip('_')
            or 'model'
        )
        count = seen.get(safe, 0)
        seen[safe] = count + 1
        out.append(safe if count == 0 else f'{safe}_{count + 1}')
    return out


def _col(rows: pd.DataFrame, name: str, metric: str) -> pd.Series:
    column = f'{name}_{metric}'
    if column not in rows:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors='coerce')


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce')
    return float(values.mean()) if values.notna().any() else float('nan')


def _quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors='coerce')
    return float(values.quantile(q)) if values.notna().any() else float('nan')


def _print_model_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    print('Plateau model summary:')
    display_cols = [
        column
        for column in (
            'model',
            'kind',
            'n',
            'cosine_mean',
            'oracle_binned_mean',
            'support_gap_mean',
            'scorer_gap_mean',
            'oos_abs_error_mean',
            'used_oracle_fraction',
            'gain_vs_first',
        )
        if column in summary.columns
    ]
    print(
        summary[display_cols].to_string(index=False, float_format=lambda x: f'{x:.5f}')
    )


def _print_winner_summary(summary: pd.DataFrame) -> None:
    if summary.empty or 'summary_type' not in summary:
        return
    numeric = summary[summary['summary_type'] == 'numeric_feature'].copy()
    if numeric.empty:
        return
    numeric = numeric.sort_values('abs_standardized_difference', ascending=False).head(
        12
    )
    display_cols = [
        column
        for column in numeric.columns
        if column in {'feature', 'abs_standardized_difference'}
        or column.endswith('_mean')
        or '_minus_' in column
    ]
    print('\nLargest rowwise-winner regime differences:')
    print(
        numeric[display_cols].to_string(index=False, float_format=lambda x: f'{x:.5f}')
    )


def _print_group_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    print('\nLargest low-performance groups:')
    cosine_cols = [
        column for column in summary.columns if column.endswith('_cosine_mean')
    ]
    if not cosine_cols:
        return
    primary = cosine_cols[0]
    cols = [
        column
        for column in ('group_type', 'group', 'n', primary, 'rowwise_best_gain_mean')
        if column in summary.columns
    ]
    print(
        summary.sort_values(primary, na_position='last')
        .head(12)[cols]
        .to_string(index=False, float_format=lambda x: f'{x:.5f}')
    )


if __name__ == '__main__':
    main()
