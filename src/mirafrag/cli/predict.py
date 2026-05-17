from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import resolve_device, validate_checkpoint_bin_config
from mirafrag.data import (
    BinnedSpectrumDataset,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_supported_elements,
    read_table,
)
from mirafrag.evaluation import evaluate_model
from mirafrag.fragments import fragment_config_from_model_config
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mirafrag-predict',
        description='Predict MS/MS spectra with a MiraFrag checkpoint.',
    )
    parser.add_argument('-i', '--input', required=True, help='Input CSV/TSV.')
    parser.add_argument('-m', '--model', required=True, help='MiraFrag checkpoint .pt.')
    parser.add_argument('-o', '--output', required=True, help='Output prediction CSV.')
    parser.add_argument('--device', default='auto')
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
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during prediction.',
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
    graph_config = infer_graph_config(model.encoder)
    df, element_stats = filter_supported_elements(
        df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'Prediction element filter: {element_stats}')
    if df.empty:
        raise SystemExit('No rows left after encoder element filtering.')
    ds = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=model.metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        require_spectrum=False,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
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
    predictions, _summary = evaluate_model(
        model,
        loader,
        device=device,
        min_intensity=args.min_intensity,
        top_k=args.top_k,
        show_progress=args.progress,
    )
    predictions.to_csv(args.output, index=False)
    print(f'Wrote predictions to {args.output}')


if __name__ == '__main__':
    main()
