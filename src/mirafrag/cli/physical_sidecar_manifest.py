from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from mirafrag.adducts import parse_adduct
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import resolve_device
from mirafrag.data import (
    ADDUCT_ALIASES,
    BinnedSpectrumDataset,
    MetadataConfig,
    _graph_config_cache_settings,
    filter_massspecgym_simulation,
    filter_supported_elements,
    find_column,
    read_table,
    select_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mirafrag-physical-sidecar-manifest',
        description=(
            'Create a cid-physical-features manifest from MassSpecGym rows and '
            'existing MiraFrag graph-cache paths.'
        ),
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='Checkpoint used to infer graph-cache settings.',
    )
    parser.add_argument(
        '-o', '--output', required=True, help='Output manifest CSV path.'
    )
    parser.add_argument(
        '--disk-cache-dir', required=True, help='MiraFrag feature cache root.'
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--massspecgym-filter', action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        '--include-missing-cache', action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        '--graph-relaxation', choices=['rdkit', 'aimnet', 'none'], default=None
    )
    parser.add_argument('--aimnet-relax-model', default=None)
    parser.add_argument('--aimnet-relax-steps', type=int, default=None)
    parser.add_argument('--aimnet-relax-fmax', type=float, default=None)
    parser.add_argument('--aimnet-relax-device', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quiet_rdkit_logs()
    device = resolve_device(args.device) if args.device != 'cpu' else 'cpu'
    model, payload = load_checkpoint(args.model, device=device)
    saved_graph_config = payload.get('graph_config') or {}
    graph_config = infer_graph_config(
        model.encoder,
        relaxation=args.graph_relaxation
        or saved_graph_config.get('relaxation', 'rdkit'),
        aimnet_relax_model=args.aimnet_relax_model
        or saved_graph_config.get('aimnet_relax_model', 'aimnet2'),
        aimnet_relax_steps=args.aimnet_relax_steps
        if args.aimnet_relax_steps is not None
        else int(saved_graph_config.get('aimnet_relax_steps', 50)),
        aimnet_relax_fmax=args.aimnet_relax_fmax
        if args.aimnet_relax_fmax is not None
        else float(saved_graph_config.get('aimnet_relax_fmax', 0.05)),
        aimnet_relax_device=args.aimnet_relax_device
        or saved_graph_config.get('aimnet_relax_device', 'auto'),
    )

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    selected = []
    for split in args.splits:
        if str(split).lower() == 'all':
            selected.append(df)
        else:
            selected.append(select_split(df, split=split, split_col=args.split_col))
    if not selected:
        raise SystemExit('No splits selected.')
    df = (
        selected[0].copy()
        if len(selected) == 1
        else pd.concat(selected, ignore_index=True)
    )
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()
    df, element_stats = filter_supported_elements(
        df, supported_atomic_numbers=graph_config.atomic_numbers
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'Element filter: {element_stats}')
    if df.empty:
        raise SystemExit('No rows left after split and element filtering.')

    metadata_config = MetadataConfig.from_dataframe(df)
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        require_spectrum=False,
        include_fragments=False,
        disk_cache_dir=args.disk_cache_dir,
    )
    adduct_col = find_column(df, ADDUCT_ALIASES, required=False)

    rows = []
    seen_smiles: set[str] = set()
    missing = 0
    for idx, row in df.reset_index(drop=True).iterrows():
        smiles = str(row[dataset.smiles_col])
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        path = dataset._feature_cache_path(
            'graphs',
            smiles,
            {'graph_config': _graph_config_cache_settings(graph_config)},
        )
        if not path.exists():
            missing += 1
            if not args.include_missing_cache:
                continue
        adduct = str(row.get(adduct_col, '')) if adduct_col is not None else ''
        try:
            charge = int(parse_adduct(adduct).charge)
        except Exception:
            charge = 0
        rows.append(
            {
                'input': str(path),
                'smiles': smiles,
                'molecule_id': str(row.get('identifier', row.get('inchikey', idx))),
                'charge': charge,
                'spin': 1,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['input', 'smiles', 'molecule_id', 'charge', 'spin']
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f'Wrote {len(rows)} manifest rows to {output} '
        f'unique_smiles={len(seen_smiles)} missing_cache={missing}'
    )


if __name__ == '__main__':
    main()
