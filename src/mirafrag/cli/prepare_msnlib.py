from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert FIORA-preprocessed MSnLib library.csv to MiraFrag TSV.'
    )
    parser.add_argument('-i', '--input', required=True, help='FIORA MSnLib library.csv')
    parser.add_argument('-o', '--output', required=True, help='Output MiraFrag TSV')
    parser.add_argument(
        '--instrument-source',
        choices=('fragmentation', 'instrument_type'),
        default='fragmentation',
        help=(
            'Use FIORA fragmentation/instrument column such as HCD, or source '
            'instrument type such as Orbitrap.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input, low_memory=False)
    required = {
        'SMILES',
        'Precursor_type',
        'CE',
        'PEPMASS',
        'peaks',
        'datasplit',
        'NAME',
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise SystemExit(f'MSnLib input is missing required columns: {missing}')

    if args.instrument_source == 'fragmentation':
        instrument = _require_first_available(
            source, ('instrument', 'FRAGMENTATION_METHOD')
        )
    else:
        instrument = _require_first_available(
            source, ('INSTRUMENT_TYPE', 'Instrument_type')
        )

    out = pd.DataFrame(
        {
            'identifier': _unique_identifiers(source),
            'name': source['NAME'].astype(str),
            'smiles': source['SMILES'].astype(str),
            'adduct': source['Precursor_type'].astype(str),
            'precursor_mz': pd.to_numeric(source['PEPMASS'], errors='coerce'),
            'collision_energy': pd.to_numeric(source['CE'], errors='coerce'),
            'instrument_type': instrument.astype(str),
            'peaks': source['peaks'].astype(str),
            'split': source['datasplit'].astype(str).str.strip().str.lower(),
        }
    )
    out = out.dropna(subset=['smiles', 'precursor_mz', 'collision_energy', 'peaks'])
    out = out[out['smiles'].astype(str).str.len() > 0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep='\t', index=False)
    print(f'Wrote {len(out)} MSnLib rows to {output} splits={_split_counts(out)}')


def _unique_identifiers(source: pd.DataFrame) -> pd.Series:
    if 'USI' not in source.columns:
        return pd.Series([f'msnlib:{idx}' for idx in source.index], index=source.index)
    base = source['USI'].astype(str)
    duplicate = base.duplicated(keep=False)
    if not duplicate.any():
        return base
    unique = base.copy()
    unique.loc[duplicate] = [
        f'{identifier}#row={idx}'
        for identifier, idx in zip(base.loc[duplicate], source.index[duplicate])
    ]
    return unique


def _require_first_available(
    source: pd.DataFrame, columns: tuple[str, ...]
) -> pd.Series:
    for column in columns:
        if column in source.columns:
            return source[column]
    raise SystemExit(f'MSnLib input is missing one of columns: {columns}')


def _split_counts(out: pd.DataFrame) -> dict[str, int]:
    return {
        str(split): int(count)
        for split, count in out['split'].value_counts().sort_index().items()
    }


if __name__ == '__main__':
    main()
