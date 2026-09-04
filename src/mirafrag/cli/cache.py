from __future__ import annotations

import argparse

from mirafrag.cache_fill import fill_feature_cache_unordered
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    add_high_ce_fragment_support_args,
    apply_fragment_args_to_model_config,
    high_ce_fragment_config_from_args,
    resolve_device,
    value_or_default,
)
from mirafrag.config import MiraFragConfig
from mirafrag.data import (
    ADDUCT_ALIASES,
    CE_ALIASES,
    SMILES_ALIASES,
    BinnedSpectrumDataset,
    MetadataConfig,
    filter_massspecgym_simulation,
    filter_supported_elements,
    find_column,
    read_table,
    select_split,
)
from mirafrag.encoders import load_foundation_encoder
from mirafrag.fragments import (
    FragmentConfig,
    FragmentSupportProfile,
    fragment_support_profile_from_model_config,
)
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for cache precomputation.

    The options define input data, encoder selection, fragment support settings, splits, and unordered worker-pool behavior for filling disk feature caches.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-cache',
        description='Precompute MiraFrag graph and fragment feature caches.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument(
        '--disk-cache-dir', required=True, help='Output feature cache dir.'
    )
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help='Optional MiraFrag checkpoint whose encoder config and head settings define the cache.',
    )
    parser.add_argument('--foundation-source', default='off')
    parser.add_argument('--foundation-model', default='medium')
    parser.add_argument('--foundation-path', default=None)
    parser.add_argument(
        '--encoder',
        choices=['mace', 'aimnet', 'unimol'],
        default='mace',
        help='Foundation atom encoder.',
    )
    parser.add_argument('--aimnet-model', default='aimnet2')
    parser.add_argument('--aimnet-path', default=None)
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
    parser.add_argument('--aimnet-relax-steps', type=int, default=50)
    parser.add_argument('--aimnet-relax-fmax', type=float, default=0.05)
    parser.add_argument('--aimnet-relax-device', default='auto')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument('--max-fragment-tree-depth', type=int, default=None)
    parser.add_argument('--max-fragment-broken-bonds', type=int, default=None)
    parser.add_argument('--max-fragments', type=int, default=None)
    parser.add_argument('--max-fragment-edges', type=int, default=None)
    parser.add_argument(
        '--include-fragment-isotopes',
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument('--fragment-isotope-threshold', type=float, default=None)
    parser.add_argument('--max-fragment-isotope-peaks', type=int, default=None)
    parser.add_argument(
        '--fragment-action-primary-layers',
        type=int,
        default=None,
        help=(
            'Include fragment cut-bond provenance in cache entries when positive; '
            'matches training --fragment-action-primary-layers.'
        ),
    )
    parser.add_argument(
        '--bond-break-geometry-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Accept training head geometry setting for cache/config parity. '
            'This does not change cache contents by itself.'
        ),
    )
    parser.add_argument(
        '--fragment-action-geometry-features',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Deprecated alias for --bond-break-geometry-features.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help="Splits to precompute, or 'all' to precompute the filtered table once.",
    )
    add_high_ce_fragment_support_args(parser)
    parser.add_argument('--split-col', default='auto')
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply the MassSpecGym simulation-challenge filter.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=64,
        help='Number of dataset indices handed to each cache worker task.',
    )
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--slow-sample-seconds',
        type=float,
        default=0.0,
        help=(
            'Print idx/identifier/SMILES diagnostics for cache samples taking '
            'at least this many seconds. Use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--trace-samples',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print every dataset sample as it starts loading.',
    )
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars while filling the cache.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the feature-cache precomputation command.

    The command loads either a checkpoint-defined encoder/config or a fresh foundation encoder on CPU by default, filters rows to supported elements, deduplicates cache keys, and fills graph/fragment cache files.
    """
    args = parse_args()
    if (
        args.fragment_action_primary_layers is not None
        and args.fragment_action_primary_layers < 0
    ):
        raise SystemExit('--fragment-action-primary-layers must be nonnegative.')
    quiet_rdkit_logs()
    device = 'cpu' if args.device == 'auto' else resolve_device(args.device)

    if args.init_checkpoint:
        model, _payload = load_checkpoint(args.init_checkpoint, device=device)
        _apply_fragment_args_to_model_config(model.config, args)
        graph_source = model.encoder
        fragment_support_profile = fragment_support_profile_from_model_config(
            model.config
        )
    else:
        encoder = load_foundation_encoder(
            encoder_type=args.encoder,
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
        graph_source = encoder
        fragment_config = _fragment_config_from_args(args)
        fragment_support_profile = FragmentSupportProfile(
            base=fragment_config,
            high_ce_threshold=args.high_ce_fragment_threshold,
            high_ce=high_ce_fragment_config_from_args(fragment_config, args),
        )

    graph_config = infer_graph_config(
        graph_source,
        seed=args.seed,
        relaxation=args.graph_relaxation,
        aimnet_relax_model=args.aimnet_relax_model or args.aimnet_model,
        aimnet_relax_steps=args.aimnet_relax_steps,
        aimnet_relax_fmax=args.aimnet_relax_fmax,
        aimnet_relax_device=args.aimnet_relax_device,
    )
    df = read_table(args.input)
    if args.massspecgym_filter:
        df = filter_massspecgym_simulation(df)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    splits = [str(split) for split in args.splits]
    if splits == ['all'] or 'all' in {split.lower() for split in splits}:
        _precompute_frame(
            df,
            split_name='all',
            graph_config=graph_config,
            fragment_support_profile=fragment_support_profile,
            args=args,
        )
    else:
        for split in splits:
            split_df = select_split(
                df,
                split=split,
                split_col=args.split_col,
            )
            if split_df.empty:
                print(f'Warning: no rows selected for split {split!r}.')
                continue
            _precompute_frame(
                split_df,
                split_name=split,
                graph_config=graph_config,
                fragment_support_profile=fragment_support_profile,
                args=args,
            )


def _precompute_frame(
    df,
    *,
    split_name: str,
    graph_config,
    fragment_support_profile: FragmentSupportProfile,
    args: argparse.Namespace,
) -> None:
    """
    Precompute cache entries for one selected dataframe split.

    Rows are element-filtered and deduplicated by SMILES/adduct before an unordered worker pool materializes dataset items, causing missing graph and fragment cache files to be written.
    """
    df, element_stats = filter_supported_elements(
        df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        element_stats['dropped_invalid_smiles']
        or element_stats['dropped_unsupported_elements']
    ):
        print(f'{split_name} element filter: {element_stats}')
    if df.empty:
        print(
            f'Warning: no rows left for split {split_name!r} after element filtering.'
        )
        return
    input_rows = int(len(df))
    df = _deduplicate_cache_rows(df, fragment_support_profile)
    if len(df) != input_rows:
        print(
            f'{split_name} cache rows deduplicated: '
            f'{input_rows} -> {len(df)} unique SMILES/adduct keys'
        )

    metadata_config = MetadataConfig.from_dataframe(
        df,
        precursor_mz_max=args.mz_max,
        collision_energy_max=100.0,
    )
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata_config,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        require_spectrum=False,
        memory_cache=False,
        disk_cache_dir=args.disk_cache_dir,
        include_fragments=True,
        fragment_support_profile=fragment_support_profile,
        slow_sample_seconds=args.slow_sample_seconds,
        trace_samples=args.trace_samples,
    )
    total = fill_feature_cache_unordered(
        dataset,
        desc=f'cache {split_name}',
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        show_progress=args.progress,
    )
    print(
        f'cached {total} rows for split={split_name} '
        f'disk_cache_dir={args.disk_cache_dir} fragments=True'
    )


def _deduplicate_cache_rows(df, fragment_support_profile: FragmentSupportProfile):
    """
    Drop repeated rows that map to the same cache entries.

    Graph caches depend on SMILES. Fragment caches depend on SMILES, adduct,
    and, when enabled, whether the row uses base or high-CE support.
    """
    smiles_col = find_column(df, SMILES_ALIASES)
    adduct_col = find_column(df, ADDUCT_ALIASES, required=False)
    subset = [smiles_col]
    if adduct_col is not None:
        subset.append(adduct_col)
    work = df
    profile_col = None
    ce_col = find_column(df, CE_ALIASES, required=False)
    if fragment_support_profile.is_enabled() and ce_col is not None:
        profile_col = '_mirafrag_fragment_support_profile'
        work = df.copy()
        work[profile_col] = [
            _support_profile_label(fragment_support_profile, value)
            for value in work[ce_col]
        ]
        subset.append(profile_col)
    out = work.drop_duplicates(subset=subset).reset_index(drop=True)
    if profile_col is not None:
        out = out.drop(columns=[profile_col])
    return out


def _support_profile_label(
    fragment_support_profile: FragmentSupportProfile,
    collision_energy,
) -> str:
    """Return the cache-dedup label for one support profile decision."""
    config = fragment_support_profile.config_for_collision_energy(collision_energy)
    return 'high_ce' if config is fragment_support_profile.high_ce else 'base'


def _apply_fragment_args_to_model_config(
    config: MiraFragConfig,
    args: argparse.Namespace,
) -> None:
    """
    Apply candidate-support overrides to a loaded config.

    This wrapper keeps cache CLI code aligned with the shared fragment-argument policy in ``cli.common``.
    """
    apply_fragment_args_to_model_config(config, args)


def _fragment_config_from_args(args: argparse.Namespace) -> FragmentConfig:
    """
    Build a fragment-generation config from CLI arguments.

    Unset command-line values fall back to :class:`FragmentConfig` defaults, which lets the cache command run without requiring a checkpoint.
    """
    default = FragmentConfig()
    return FragmentConfig(
        max_tree_depth=value_or_default(
            args.max_fragment_tree_depth,
            default.max_tree_depth,
        ),
        max_broken_bonds=value_or_default(
            args.max_fragment_broken_bonds,
            default.max_broken_bonds,
        ),
        max_fragments=value_or_default(args.max_fragments, default.max_fragments),
        max_edges=value_or_default(args.max_fragment_edges, default.max_edges),
        include_isotopes=value_or_default(
            args.include_fragment_isotopes,
            default.include_isotopes,
        ),
        isotope_threshold=value_or_default(
            args.fragment_isotope_threshold,
            default.isotope_threshold,
        ),
        max_isotope_peaks=value_or_default(
            args.max_fragment_isotope_peaks,
            default.max_isotope_peaks,
        ),
        include_bond_breaks=bool((args.fragment_action_primary_layers or 0) > 0),
    )


if __name__ == '__main__':
    main()
