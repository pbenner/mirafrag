from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import resolve_device, validate_checkpoint_bin_config
from mirafrag.data import (
    BinnedSpectrumDataset,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    read_table,
    select_split,
)
from mirafrag.fragments import fragment_config_from_model_config
from mirafrag.model import load_checkpoint
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX
from mirafrag.training import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mirafrag-eval',
        description='Evaluate a MiraFrag checkpoint on a MassSpecGym split.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument('-m', '--model', required=True, help='MiraFrag checkpoint .pt.')
    parser.add_argument('-o', '--output', default=None, help='Optional output CSV.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', default='test')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--cache-graphs',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep decoded graphs/fragments in each worker process.',
    )
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='Optional disk cache for precomputed encoder graphs and fragments.',
    )
    parser.add_argument('--min-intensity', type=float, default=0.001)
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--max-rows', type=int, default=None)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quiet_rdkit_logs()
    device = resolve_device(args.device)
    model, _payload = load_checkpoint(args.model, device=device)
    validate_checkpoint_bin_config(
        model,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )

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
        cache_graphs=args.cache_graphs,
        cache_dir=args.cache_dir,
        include_fragments=True,
        fragment_config=fragment_config_from_model_config(model.config),
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
    )
    print(
        f'n={len(predictions)} '
        f'cosine_mean={summary["cosine_mean"]:.5f} '
        f'sqrt_cosine_mean={summary["sqrt_cosine_mean"]:.5f}'
    )
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f'Wrote predictions to {args.output}')


if __name__ == '__main__':
    main()
