from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import torch
from torch.utils.data import DataLoader

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    add_high_ce_fragment_support_args,
    resolve_device,
)
from mirafrag.config import MiraFragConfig
from mirafrag.data import (
    BinnedSpectrumDataset,
    MetadataConfig,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    merge_group_spectra,
    normalize_collision_energy_dataframe,
    read_table,
    select_split,
)
from mirafrag.encoders import load_foundation_encoder
from mirafrag.fragments import fragment_support_profile_from_model_config
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
        choices=['mace', 'aimnet', 'unimol'],
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
    parser.add_argument('--unimol-model-name', default='unimolv1')
    parser.add_argument('--unimol-model-size', default='84m')
    parser.add_argument('--unimol-pretrained-model-path', default=None)
    parser.add_argument('--unimol-pretrained-dict-path', default=None)
    parser.add_argument('--unimol-max-atoms', type=int, default=512)
    parser.add_argument(
        '--unimol-mode',
        choices=['trainable', 'frozen'],
        default='trainable',
        help='Uni-Mol encoder mode: trainable uses the underlying nn.Module; frozen uses UniMolRepr inference.',
    )
    parser.add_argument(
        '--graph-relaxation',
        choices=['rdkit', 'aimnet', 'none'],
        default='rdkit',
        help='Coordinate relaxation method used when building molecular graphs.',
    )
    parser.add_argument(
        '--aimnet-relax-model',
        default=None,
        help='AIMNet model name used for --graph-relaxation aimnet.',
    )
    parser.add_argument(
        '--aimnet-relax-steps',
        type=int,
        default=50,
        help='Maximum ASE optimizer steps for AIMNet graph relaxation.',
    )
    parser.add_argument(
        '--aimnet-relax-fmax',
        type=float,
        default=0.05,
        help='ASE force convergence threshold for AIMNet graph relaxation.',
    )
    parser.add_argument(
        '--aimnet-relax-device',
        default='auto',
        help='Device for AIMNet graph relaxation; auto lets AIMNet choose.',
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--collision-energy-mode',
        choices=['raw', 'normalized'],
        default='raw',
        help=(
            'Use raw CE with model-side robust normalization, or pre-normalize CE '
            'from training statistics and pass it directly to the model.'
        ),
    )
    parser.add_argument(
        '--merge-group-spectra',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Merge replicate spectra inside each selected split by molecule, '
            'adduct, and instrument before training/evaluation.'
        ),
    )
    parser.add_argument(
        '--merge-group-cols',
        default='auto',
        help=(
            'Comma-separated dataframe columns used by --merge-group-spectra; '
            'auto uses SMILES, adduct, and instrument when available.'
        ),
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument(
        '--head-learning-rate',
        type=float,
        default=None,
        help='Optional spectrum-head learning rate; defaults to --learning-rate.',
    )
    parser.add_argument(
        '--encoder-learning-rate',
        type=float,
        default=None,
        help='Optional trainable-encoder learning rate; defaults to --learning-rate.',
    )
    parser.add_argument(
        '--encoder-layer-lr-decay',
        type=float,
        default=1.0,
        help=(
            'Optional AIMNet encoder layerwise learning-rate decay in (0, 1]. '
            'The final AIMNet MLP block uses the encoder learning rate; earlier '
            'blocks and base encoder parameters receive progressively smaller rates.'
        ),
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help=(
            'Legacy trainable-encoder weight decay. Use --encoder-weight-decay '
            'to set it explicitly; the spectrum head is controlled separately '
            'with --head-weight-decay.'
        ),
    )
    parser.add_argument(
        '--head-weight-decay',
        type=float,
        default=0.0,
        help='AdamW weight decay for decayable spectrum-head weight matrices.',
    )
    parser.add_argument(
        '--encoder-weight-decay',
        type=float,
        default=None,
        help='AdamW weight decay for decayable trainable-encoder weight matrices.',
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
        '--fragment-path-layers',
        type=int,
        default=None,
        help=(
            'Number of recursive parent-to-child path propagation steps in the '
            'fragment head; 0 disables the path scorer.'
        ),
    )
    parser.add_argument(
        '--fragment-path-primary',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Use learned root-to-fragment path probabilities as the primary '
            'formula scorer. This keeps multi-break fragments by accumulating '
            'transition scores over the retained fragment DAG.'
        ),
    )
    parser.add_argument(
        '--fragment-action-primary-layers',
        type=int,
        default=None,
        help=(
            'Number of hidden layers in the explicit bond-action formula residual. '
            'This uses cached oriented broken-bond provenance to inject latent '
            'multi-break fragmentation evidence into the primary formula logits.'
        ),
    )
    parser.add_argument(
        '--bond-break-geometry-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Append invariant conformer geometry features to explicit broken-bond '
            'events. When unset, new bond-action branches enable this by default.'
        ),
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
    add_high_ce_fragment_support_args(parser)
    parser.add_argument(
        '--fine-tune-strategy',
        choices=['head', 'full'],
        default='head',
        help='Encoder adaptation strategy: head freezes the encoder, full trains encoder weights.',
    )
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument(
        '--loss',
        choices=LOSS_NAMES,
        default='decoupled_kl',
    )
    parser.add_argument(
        '--kl-weight',
        type=float,
        default=0.7,
        help='KL mixture weight for --loss kl_cosine or decoupled_kl_cosine; 1.0 is pure KL.',
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
        '--prefill-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Precompute missing disk-cache entries before constructing DataLoaders.',
    )
    parser.add_argument(
        '--cache-num-workers',
        type=int,
        default=None,
        help='Worker count for disk-cache prefill. Defaults to --num-workers.',
    )
    parser.add_argument(
        '--cache-chunk-size',
        type=int,
        default=None,
        help='Chunk size for disk-cache prefill. Defaults to 1.',
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
        choices=['val_loss', 'train_loss', 'val_cosine', 'train_cosine'],
        default='val_loss',
        help='Metric used to decide when to save the best checkpoint.',
    )
    parser.add_argument(
        '--swa',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable stochastic weight averaging after --swa-start-epoch.',
    )
    parser.add_argument(
        '--swa-start-epoch',
        type=int,
        default=None,
        help='First epoch included in the SWA average; defaults to epochs // 2.',
    )
    parser.add_argument(
        '--swa-lr',
        type=float,
        default=None,
        help='Optional SWA tail learning rate. If omitted, the main scheduler continues.',
    )
    parser.add_argument(
        '--swa-anneal-epochs',
        type=int,
        default=1,
        help='Epochs used by PyTorch SWALR to anneal to --swa-lr.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the end-to-end training command.

    This entry point reads and filters data, creates or restores a model, prepares datasets and caches, builds DataLoaders, and calls the training loop with the resolved configuration.
    """
    args = parse_args()
    if args.fragment_path_layers is not None and args.fragment_path_layers < 0:
        raise SystemExit('--fragment-path-layers must be nonnegative.')
    if (
        args.fragment_action_primary_layers is not None
        and args.fragment_action_primary_layers < 0
    ):
        raise SystemExit('--fragment-action-primary-layers must be nonnegative.')
    if (
        args.fragment_path_primary is True
        and not args.init_checkpoint
        and not args.fragment_path_layers
    ):
        raise SystemExit('--fragment-path-primary requires --fragment-path-layers > 0.')
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
        model = _maybe_rebuild_fragment_bond_break_model(model, args, device=device)
        _apply_fragment_args_to_model_config(model.config, args)
        set_encoder_finetune_strategy(model, fine_tune_strategy)
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
            unimol_model_name=args.unimol_model_name,
            unimol_model_size=args.unimol_model_size,
            unimol_pretrained_model_path=args.unimol_pretrained_model_path,
            unimol_pretrained_dict_path=args.unimol_pretrained_dict_path,
            unimol_max_atoms=args.unimol_max_atoms,
            unimol_mode=args.unimol_mode,
            device=device,
        )
        metadata_config = MetadataConfig.from_dataframe(
            train_df,
            precursor_mz_max=args.mz_max,
            collision_energy_max=100.0,
            collision_energy_mode=args.collision_energy_mode,
        )
    metadata_ce_mode = str(
        getattr(metadata_config, 'collision_energy_mode', 'raw') or 'raw'
    ).lower()
    if args.init_checkpoint:
        if (
            args.collision_energy_mode == 'normalized'
            and metadata_ce_mode != 'normalized'
        ):
            metadata_config = replace(
                metadata_config,
                collision_energy_mode='normalized',
            )
            model.metadata_config = metadata_config
            metadata_ce_mode = 'normalized'
            print(
                'Collision-energy metadata override: normalized using loaded '
                'checkpoint CE statistics'
            )
        elif args.collision_energy_mode == 'raw' and metadata_ce_mode != 'raw':
            print(
                'Collision-energy metadata: keeping normalized mode from loaded '
                'checkpoint'
            )
    if metadata_ce_mode == 'normalized':
        train_df = normalize_collision_energy_dataframe(
            train_df, metadata_config=metadata_config
        )
        if not val_df.empty:
            val_df = normalize_collision_energy_dataframe(
                val_df, metadata_config=metadata_config
            )
        print('Collision-energy preprocessing: normalized from training metadata stats')
    if args.merge_group_spectra:
        before_train = len(train_df)
        before_val = len(val_df)
        train_df = merge_group_spectra(
            train_df,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
            group_cols=args.merge_group_cols,
        )
        if not val_df.empty:
            val_df = merge_group_spectra(
                val_df,
                mz_max=args.mz_max,
                bin_width=args.bin_width,
                group_cols=args.merge_group_cols,
            )
        print(
            'Merged group spectra: '
            f'train {before_train}->{len(train_df)} val {before_val}->{len(val_df)}'
        )

    graph_source = model.encoder if args.init_checkpoint else encoder
    graph_config = infer_graph_config(
        graph_source,
        seed=args.seed,
        relaxation=args.graph_relaxation,
        aimnet_relax_model=args.aimnet_relax_model or args.aimnet_model,
        aimnet_relax_steps=args.aimnet_relax_steps,
        aimnet_relax_fmax=args.aimnet_relax_fmax,
        aimnet_relax_device=args.aimnet_relax_device,
    )
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
        num_bins = num_spectrum_bins(args.mz_max, args.bin_width)
        fragment_action_primary_layers = _mirafrag_config_value(
            args.fragment_action_primary_layers,
            'fragment_action_primary_layers',
        )
        bond_break_geometry_features = _new_model_bond_break_geometry_features(
            args,
            action_primary_layers=fragment_action_primary_layers,
        )
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
            high_ce_fragment_threshold=args.high_ce_fragment_threshold,
            high_ce_max_fragment_tree_depth=args.high_ce_max_fragment_tree_depth,
            high_ce_max_fragment_broken_bonds=args.high_ce_max_fragment_broken_bonds,
            high_ce_max_fragments=args.high_ce_max_fragments,
            high_ce_max_fragment_edges=args.high_ce_max_fragment_edges,
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
            fragment_path_layers=_mirafrag_config_value(
                args.fragment_path_layers, 'fragment_path_layers'
            ),
            fragment_path_primary=bool(
                _mirafrag_config_value(
                    args.fragment_path_primary, 'fragment_path_primary'
                )
            ),
            fragment_action_primary_layers=fragment_action_primary_layers,
            bond_break_geometry_features=bond_break_geometry_features,
            ce_embedding=_mirafrag_config_value(args.ce_embedding, 'ce_embedding'),
            encoder_type=encoder_type,
            encoder_finetune_strategy=fine_tune_strategy,
            foundation_source=args.foundation_source,
            foundation_model=args.foundation_model,
            foundation_path=args.foundation_path,
            aimnet_model=args.aimnet_model,
            aimnet_path=args.aimnet_path,
            unimol_model_name=args.unimol_model_name,
            unimol_model_size=args.unimol_model_size,
            unimol_pretrained_model_path=args.unimol_pretrained_model_path,
            unimol_pretrained_dict_path=args.unimol_pretrained_dict_path,
            unimol_max_atoms=args.unimol_max_atoms,
            unimol_mode=args.unimol_mode,
        )
        model = MiraFragModel(encoder, metadata_config=metadata_config, config=config)
    fragment_support_profile = fragment_support_profile_from_model_config(model.config)

    train_ds = BinnedSpectrumDataset(
        train_df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        memory_cache=args.memory_cache,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_support_profile=fragment_support_profile,
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
            fragment_support_profile=fragment_support_profile,
            slow_sample_seconds=args.slow_sample_seconds,
            trace_samples=args.trace_samples,
        )
        if not val_df.empty
        else None
    )
    if args.disk_cache_dir is not None and args.prefill_cache:
        cache_num_workers = (
            args.num_workers
            if args.cache_num_workers is None
            else args.cache_num_workers
        )
        cache_chunk_size = 1 if args.cache_chunk_size is None else args.cache_chunk_size
        prefill_feature_cache(
            train_ds,
            split_name='train',
            chunk_size=cache_chunk_size,
            num_workers=cache_num_workers,
            show_progress=args.progress,
        )
        if val_ds is not None:
            prefill_feature_cache(
                val_ds,
                split_name='val',
                chunk_size=cache_chunk_size,
                num_workers=cache_num_workers,
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
        head_lr=args.head_learning_rate,
        encoder_lr=args.encoder_learning_rate,
        head_weight_decay=args.head_weight_decay,
        encoder_weight_decay=args.encoder_weight_decay,
        encoder_layer_lr_decay=args.encoder_layer_lr_decay,
        output=args.output,
        loss_name=args.loss,
        train_config=_train_config(args, fine_tune_strategy=fine_tune_strategy),
        graph_config=graph_config,
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
        verbose_epoch_config=False,
        swa=args.swa,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr,
        swa_anneal_epochs=args.swa_anneal_epochs,
    )


def _config_bond_break_geometry_features(config: MiraFragConfig) -> bool:
    return bool(getattr(config, 'bond_break_geometry_features', False))


def _requested_bond_break_geometry_features(args: argparse.Namespace) -> bool | None:
    value = getattr(args, 'bond_break_geometry_features', None)
    if value is not None:
        return bool(value)
    value = getattr(args, 'fragment_action_geometry_features', None)
    if value is not None:
        return bool(value)
    return None


def _new_model_bond_break_geometry_features(
    args: argparse.Namespace,
    *,
    action_primary_layers: int,
) -> bool:
    requested = _requested_bond_break_geometry_features(args)
    if requested is not None:
        return requested
    return int(action_primary_layers) > 0


def _checkpoint_bond_break_geometry_features(
    args: argparse.Namespace,
    config: MiraFragConfig,
    *,
    target_action_primary_layers: int,
) -> bool:
    requested = _requested_bond_break_geometry_features(args)
    if requested is not None:
        return requested
    if (
        int(target_action_primary_layers) > 0
        and int(getattr(config, 'fragment_action_primary_layers', 0)) <= 0
    ):
        return True
    return _config_bond_break_geometry_features(config)


def _maybe_rebuild_fragment_bond_break_model(
    model: MiraFragModel,
    args: argparse.Namespace,
    *,
    device: str | torch.device,
) -> MiraFragModel:
    requested_path_layers = getattr(args, 'fragment_path_layers', None)
    requested_path_primary = getattr(args, 'fragment_path_primary', None)
    requested_action_primary_layers = getattr(
        args, 'fragment_action_primary_layers', None
    )

    target_path_layers = int(
        getattr(model.config, 'fragment_path_layers', 0)
        if requested_path_layers is None
        else requested_path_layers
    )
    target_path_primary = bool(
        getattr(model.config, 'fragment_path_primary', False)
        if requested_path_primary is None
        else requested_path_primary
    )
    target_action_primary_layers = int(
        getattr(model.config, 'fragment_action_primary_layers', 0)
        if requested_action_primary_layers is None
        else requested_action_primary_layers
    )
    target_geometry = _checkpoint_bond_break_geometry_features(
        args,
        model.config,
        target_action_primary_layers=target_action_primary_layers,
    )

    current = (
        int(getattr(model.config, 'fragment_path_layers', 0)),
        bool(getattr(model.config, 'fragment_path_primary', False)),
        int(getattr(model.config, 'fragment_action_primary_layers', 0)),
        bool(getattr(model.config, 'bond_break_geometry_features', False)),
    )
    target = (
        target_path_layers,
        target_path_primary,
        target_action_primary_layers,
        target_geometry,
    )
    if current == target:
        return model

    state = model.state_dict()
    if current[3] != target[3]:
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith('head.fragment_action_primary_pair_scorer.0.')
        }
    config = replace(
        model.config,
        fragment_path_layers=target_path_layers,
        fragment_path_primary=target_path_primary,
        fragment_action_primary_layers=target_action_primary_layers,
        bond_break_geometry_features=target_geometry,
    )
    rebuilt = MiraFragModel(
        model.encoder,
        metadata_config=model.metadata_config,
        config=config,
    ).to(device)
    incompatible = rebuilt.load_state_dict(state, strict=False)
    allowed_prefixes = (
        'head.fragment_path_',
        'head.fragment_action_primary_',
        'head.formula_count_',
    )
    missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            'Unexpected checkpoint mismatch while changing fragmentation head: '
            f'missing={missing} unexpected={unexpected}'
        )
    changed = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
    if changed:
        print(
            'Loaded checkpoint with '
            f'{changed} fragmentation head parameters initialized from defaults.'
        )
    return rebuilt


def _validate_loaded_checkpoint_config(
    model: MiraFragModel,
    *,
    mz_max: float,
    bin_width: float,
) -> None:
    expected_bins = num_spectrum_bins(mz_max, bin_width)
    if model.config.num_bins != expected_bins:
        raise SystemExit(
            'Checkpoint binning does not match CLI arguments: '
            f'checkpoint num_bins={model.config.num_bins}, requested={expected_bins}. '
            'Use matching --mz-max/--bin-width or train a new model.'
        )


def _apply_fragment_args_to_model_config(
    config: MiraFragConfig, args: argparse.Namespace
) -> None:
    for arg_name, field_name in [
        ('max_fragment_tree_depth', 'max_fragment_tree_depth'),
        ('max_fragment_broken_bonds', 'max_fragment_broken_bonds'),
        ('max_fragments', 'max_fragments'),
        ('max_fragment_edges', 'max_fragment_edges'),
        ('include_fragment_isotopes', 'include_fragment_isotopes'),
        ('fragment_isotope_threshold', 'fragment_isotope_threshold'),
        ('max_fragment_isotope_peaks', 'max_fragment_isotope_peaks'),
        ('high_ce_fragment_threshold', 'high_ce_fragment_threshold'),
        ('high_ce_max_fragment_tree_depth', 'high_ce_max_fragment_tree_depth'),
        ('high_ce_max_fragment_broken_bonds', 'high_ce_max_fragment_broken_bonds'),
        ('high_ce_max_fragments', 'high_ce_max_fragments'),
        ('high_ce_max_fragment_edges', 'high_ce_max_fragment_edges'),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, field_name, value)


def _mirafrag_config_value(value, field_name: str):
    if value is not None:
        return value
    return getattr(MiraFragConfig, field_name)


def _train_config(
    args: argparse.Namespace,
    *,
    fine_tune_strategy: str,
) -> dict[str, Any]:
    return {
        'fine_tune_strategy': fine_tune_strategy,
        'learning_rate': float(args.learning_rate),
        'encoder_learning_rate': (
            None
            if args.encoder_learning_rate is None
            else float(args.encoder_learning_rate)
        ),
        'head_learning_rate': (
            None if args.head_learning_rate is None else float(args.head_learning_rate)
        ),
        'weight_decay': float(args.weight_decay),
        'head_weight_decay': float(args.head_weight_decay),
        'encoder_weight_decay': (
            None
            if args.encoder_weight_decay is None
            else float(args.encoder_weight_decay)
        ),
        'encoder_layer_lr_decay': float(args.encoder_layer_lr_decay),
        'loss': args.loss,
        'kl_weight': float(args.kl_weight),
        'coverage_weight': float(args.coverage_weight),
        'target_power': float(args.target_power),
        'entropy_weight': float(args.entropy_weight),
        'scheduler': args.scheduler,
        'scheduler_interval': args.scheduler_interval,
        'exponential_gamma': float(args.exponential_gamma),
        'min_lr_ratio': float(args.min_lr_ratio),
    }


if __name__ == '__main__':
    main()
