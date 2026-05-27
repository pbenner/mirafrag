from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    apply_fragment_args_to_model_config,
    resolve_device,
    validate_checkpoint_bin_config,
)
from mirafrag.config import MiraFragConfig
from mirafrag.data import (
    BinnedSpectrumDataset,
    MetadataConfig,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    read_table,
    select_split,
)
from mirafrag.encoders import load_foundation_encoder
from mirafrag.fragments import fragment_config_from_model_config
from mirafrag.losses import LOSS_NAMES
from mirafrag.model import MiraFragModel, set_encoder_finetune_strategy
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    num_spectrum_bins,
)
from mirafrag.training import train_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for MiraFrag training.

    The parser covers encoder selection, checkpoint resume, fragment-support settings, losses, optimizer and scheduler controls, cache behavior, and MassSpecGym split selection.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-train',
        description='Fine-tune a foundation encoder with a spectrum prediction head.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument('-o', '--output', required=True, help='Output checkpoint .pt.')
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help=(
            'Optional MiraFrag checkpoint to continue from. The saved spectrum head, '
            'metadata, and foundation weights are reused; --fine-tune-strategy '
            'controls how the encoder is adapted for the new run.'
        ),
    )
    parser.add_argument('--foundation-source', default='off')
    parser.add_argument('--foundation-model', default='medium')
    parser.add_argument('--foundation-path', default=None)
    parser.add_argument(
        '--encoder',
        choices=['mace', 'aimnet'],
        default=None,
        help='Foundation atom encoder.',
    )
    parser.add_argument(
        '--aimnet-model',
        default='aimnet2',
        help='AIMNet registry model name used when --encoder aimnet.',
    )
    parser.add_argument(
        '--aimnet-path',
        default=None,
        help='Optional local AIMNet model path used when --encoder aimnet.',
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help=(
            'Weight decay for trainable encoder parameters. In delta mode this '
            'regularizes only the delta weights; the spectrum head uses no decay.'
        ),
    )
    parser.add_argument(
        '--scheduler',
        choices=['exponential', 'plateau', 'constant', 'cosine', 'none'],
        default='exponential',
        help='Learning-rate scheduler.',
    )
    parser.add_argument(
        '--scheduler-interval',
        choices=['epoch', 'step'],
        default='epoch',
        help='Run step/epoch schedulers once per epoch or once per batch.',
    )
    parser.add_argument(
        '--min-lr-ratio',
        type=float,
        default=0.1,
        help="Final LR as a fraction of each parameter group's initial LR.",
    )
    parser.add_argument(
        '--exponential-gamma',
        type=float,
        default=0.8,
        help='Multiplicative LR decay per scheduler update for --scheduler exponential.',
    )
    parser.add_argument(
        '--plateau-factor',
        type=float,
        default=0.5,
        help='LR multiplier used by --scheduler plateau.',
    )
    parser.add_argument(
        '--plateau-patience',
        type=int,
        default=2,
        help='Validation epochs without improvement before plateau LR decay.',
    )
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--metadata-dim', type=int, default=32)
    parser.add_argument(
        '--max-fragment-tree-depth',
        type=int,
        default=None,
        help='Maximum recursive atom-removal tree depth.',
    )
    parser.add_argument(
        '--max-fragment-broken-bonds',
        type=int,
        default=None,
        help=(
            'Maximum cumulative broken-bond order and hydrogen-transfer budget '
            'for recursive fragmentation.'
        ),
    )
    parser.add_argument(
        '--max-fragments',
        type=int,
        default=None,
        help='Maximum fragment candidates per molecule.',
    )
    parser.add_argument(
        '--max-fragment-edges',
        type=int,
        default=None,
        help='Maximum directed fragment-relationship edges per molecule.',
    )
    parser.add_argument(
        '--fragment-gnn-layers',
        type=int,
        default=2,
        help='Number of message-passing layers over the fragment graph.',
    )
    parser.add_argument(
        '--include-fragment-isotopes',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Expand fragment formula candidates into isotope peak candidates.',
    )
    parser.add_argument(
        '--fragment-isotope-threshold',
        type=float,
        default=None,
        help='Minimum retained isotope probability per fragment formula.',
    )
    parser.add_argument(
        '--max-fragment-isotope-peaks',
        type=int,
        default=None,
        help='Maximum isotope peaks retained per fragment formula.',
    )
    parser.add_argument(
        '--fine-tune-strategy',
        choices=['head', 'delta', 'full'],
        default='head',
        help=(
            'Encoder adaptation strategy: head freezes the encoder, delta trains '
            'additive residual parameters, full trains encoder weights.'
        ),
    )
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument(
        '--loss',
        choices=LOSS_NAMES,
        default='kl',
    )
    parser.add_argument(
        '--kl-weight',
        type=float,
        default=0.7,
        help='KL mixture weight for --loss kl_cosine; 1.0 is pure KL.',
    )
    parser.add_argument(
        '--coverage-weight',
        type=float,
        default=0.1,
        help='Coverage penalty weight for --loss soft_binned_coverage_kl.',
    )
    parser.add_argument(
        '--target-power',
        type=float,
        default=1.0,
        help=(
            'Sharpen target intensities before KL normalization. '
            '1.0 keeps standard intensity-normalized KL.'
        ),
    )
    parser.add_argument(
        '--entropy-weight',
        type=float,
        default=0.0,
        help='Optional entropy penalty on predicted fragment-plus-OOS probabilities.',
    )
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='val')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument(
        '--mass-tolerance',
        type=float,
        default=0.01,
        help=(
            'Absolute Da tolerance for tolerance-based losses. '
            'For --loss soft_projected_kl, sigma is half this value.'
        ),
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
        help='Minimum m/z divisor for relative tolerance matching.',
    )
    parser.add_argument('--train-split-value', default=None)
    parser.add_argument('--val-split-value', default=None)
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply the MassSpecGym simulation-challenge filter.',
    )
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--memory-cache', action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        '--disk-cache-dir',
        default=None,
        help='Optional disk cache for precomputed encoder graphs and fragment candidates.',
    )
    parser.add_argument(
        '--slow-sample-seconds',
        type=float,
        default=0.0,
        help=(
            'Print idx/identifier/SMILES diagnostics for dataset samples taking '
            'at least this many seconds. Use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--trace-samples',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print every dataset sample as it starts loading.',
    )
    parser.add_argument(
        '--dataloader-timeout',
        type=float,
        default=0.0,
        help='Seconds before a DataLoader worker timeout; 0 disables the timeout.',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during training and validation.',
    )
    parser.add_argument(
        '--checkpoint-metric',
        choices=['val_loss', 'train_loss'],
        default='val_loss',
        help='Metric used to decide when to save the best checkpoint.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the end-to-end training command.

    This entry point reads and filters data, creates or restores a model, prepares datasets and caches, builds DataLoaders, and calls the training loop with the resolved configuration.
    """
    args = parse_args()
    quiet_rdkit_logs()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    train_df = select_split(
        df,
        split=args.train_split,
        split_col=args.split_col,
        split_value=args.train_split_value,
    )
    val_df = select_split(
        df,
        split=args.val_split,
        split_col=args.split_col,
        split_value=args.val_split_value,
    )
    if train_df.empty:
        raise SystemExit('No training rows selected.')
    if val_df.empty:
        print(
            'Warning: no validation rows selected; using training stats for checkpointing.'
        )

    fine_tune_strategy = args.fine_tune_strategy
    if args.init_checkpoint:
        model, _payload = load_checkpoint(args.init_checkpoint, device=device)
        _validate_loaded_checkpoint_config(
            model,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
        )
        set_encoder_finetune_strategy(model, fine_tune_strategy)
        _apply_fragment_args_to_model_config(model.config, args)
        metadata_config = model.metadata_config
        print(
            'Loaded init checkpoint '
            f'{args.init_checkpoint} with encoder_type={model.config.encoder_type} '
            f'fine_tune_strategy={fine_tune_strategy}'
        )
        if args.encoder is not None and args.encoder != model.config.encoder_type:
            print(
                f'Warning: checkpoint encoder_type={model.config.encoder_type!r}; '
                f'ignoring requested --encoder {args.encoder!r}.'
            )
    else:
        encoder_type = args.encoder or 'mace'
        encoder = load_foundation_encoder(
            encoder_type=encoder_type,
            foundation_source=args.foundation_source,
            foundation_model=args.foundation_model,
            foundation_path=args.foundation_path,
            aimnet_model=args.aimnet_model,
            aimnet_path=args.aimnet_path,
            device=device,
        )
    graph_source = model.encoder if args.init_checkpoint else encoder
    graph_config = infer_graph_config(graph_source, seed=args.seed)
    train_df, train_element_stats = filter_supported_elements(
        train_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    val_df, val_element_stats = filter_supported_elements(
        val_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        train_element_stats['dropped_invalid_smiles']
        or train_element_stats['dropped_unsupported_elements']
    ):
        print(f'Training element filter: {train_element_stats}')
    if (
        val_element_stats['dropped_invalid_smiles']
        or val_element_stats['dropped_unsupported_elements']
    ):
        print(f'Validation element filter: {val_element_stats}')
    if train_df.empty:
        raise SystemExit('No training rows left after encoder element filtering.')
    if val_df.empty:
        print(
            'Warning: no validation rows left after encoder element filtering; '
            'using training stats for checkpointing.'
        )
    if not args.init_checkpoint:
        metadata_config = MetadataConfig.from_dataframe(
            train_df,
            precursor_mz_max=args.mz_max,
            collision_energy_max=100.0,
        )
        num_bins = num_spectrum_bins(args.mz_max, args.bin_width)
        config = MiraFragConfig(
            num_bins=num_bins,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            metadata_dim=args.metadata_dim,
            max_fragment_tree_depth=_mirafrag_config_value(
                args.max_fragment_tree_depth,
                'max_fragment_tree_depth',
            ),
            max_fragment_broken_bonds=_mirafrag_config_value(
                args.max_fragment_broken_bonds,
                'max_fragment_broken_bonds',
            ),
            max_fragments=_mirafrag_config_value(args.max_fragments, 'max_fragments'),
            max_fragment_edges=_mirafrag_config_value(
                args.max_fragment_edges,
                'max_fragment_edges',
            ),
            include_fragment_isotopes=_mirafrag_config_value(
                args.include_fragment_isotopes,
                'include_fragment_isotopes',
            ),
            fragment_isotope_threshold=_mirafrag_config_value(
                args.fragment_isotope_threshold,
                'fragment_isotope_threshold',
            ),
            max_fragment_isotope_peaks=_mirafrag_config_value(
                args.max_fragment_isotope_peaks,
                'max_fragment_isotope_peaks',
            ),
            fragment_gnn_layers=args.fragment_gnn_layers,
            encoder_type=encoder_type,
            encoder_finetune_strategy=fine_tune_strategy,
            foundation_source=args.foundation_source,
            foundation_model=args.foundation_model,
            foundation_path=args.foundation_path,
            aimnet_model=args.aimnet_model,
            aimnet_path=args.aimnet_path,
        )
        model = MiraFragModel(encoder, metadata_config=metadata_config, config=config)
    fragment_config = fragment_config_from_model_config(model.config)

    train_ds = BinnedSpectrumDataset(
        train_df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_config=fragment_config,
        slow_sample_seconds=args.slow_sample_seconds,
        trace_samples=args.trace_samples,
    )
    val_ds = (
        BinnedSpectrumDataset(
            val_df,
            graph_config=graph_config,
            metadata_config=metadata_config,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
            memory_cache=args.memory_cache,
            disk_cache_dir=args.disk_cache_dir,
            include_fragments=True,
            fragment_config=fragment_config,
            slow_sample_seconds=args.slow_sample_seconds,
            trace_samples=args.trace_samples,
        )
        if not val_df.empty
        else None
    )
    if args.disk_cache_dir is not None:
        prefill_feature_cache(
            train_ds,
            split_name='train',
            chunk_size=args.batch_size,
            num_workers=args.num_workers,
            show_progress=args.progress,
        )
        if val_ds is not None:
            prefill_feature_cache(
                val_ds,
                split_name='val',
                chunk_size=args.batch_size,
                num_workers=args.num_workers,
                show_progress=args.progress,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_spectrum_batch,
        timeout=args.dataloader_timeout,
        **dataloader_performance_kwargs(
            num_workers=args.num_workers,
            device=device,
        ),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_spectrum_batch,
            timeout=args.dataloader_timeout,
            **dataloader_performance_kwargs(
                num_workers=args.num_workers,
                device=device,
            ),
        )
        if val_ds is not None
        else None
    )
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output=args.output,
        loss_name=args.loss,
        train_config=_train_config(args, fine_tune_strategy=fine_tune_strategy),
        show_progress=args.progress,
        scheduler_name=args.scheduler,
        scheduler_interval=args.scheduler_interval,
        min_lr_ratio=args.min_lr_ratio,
        exponential_gamma=args.exponential_gamma,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        evaluate_initial=bool(args.init_checkpoint),
        mass_tolerance=args.mass_tolerance,
        relative_mass_tolerance=args.relative_mass_tolerance,
        mass_tolerance_min_mz=args.mass_tolerance_min_mz,
        kl_weight=args.kl_weight,
        coverage_weight=args.coverage_weight,
        target_power=args.target_power,
        entropy_weight=args.entropy_weight,
        checkpoint_metric=args.checkpoint_metric,
    )


def _validate_loaded_checkpoint_config(
    model: MiraFragModel,
    *,
    mz_max: float,
    bin_width: float,
) -> None:
    """
    Validate runtime bin settings for a resumed checkpoint.

    The wrapper keeps the train CLI symmetric with evaluation and centralizes the checkpoint/bin compatibility check.
    """
    validate_checkpoint_bin_config(model, mz_max=mz_max, bin_width=bin_width)


def _apply_fragment_args_to_model_config(
    config: MiraFragConfig,
    args: argparse.Namespace,
) -> None:
    """
    Apply safe fragment candidate overrides to a loaded model config.

    This excludes head architecture fields so resuming from a checkpoint cannot accidentally change tensor shapes in the saved head.
    """
    apply_fragment_args_to_model_config(config, args)


def _mirafrag_config_value(value, field_name: str):
    """
    Return a CLI value or the MiraFragConfig default for a field.

    The helper is used while constructing new model configs from optional fragment arguments.
    """
    if value is not None:
        return value
    return MiraFragConfig.__dataclass_fields__[field_name].default


def _train_config(
    args: argparse.Namespace,
    *,
    fine_tune_strategy: str,
) -> dict[str, object]:
    """
    Serialize training command arguments for checkpoint metadata.

    The returned dictionary records the raw CLI settings plus the resolved fine-tune strategy for later inspection.
    """
    config = vars(args).copy()
    config['resolved_fine_tune_strategy'] = fine_tune_strategy
    return config


if __name__ == '__main__':
    main()
