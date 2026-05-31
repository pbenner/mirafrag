from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint
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
from mirafrag.evaluation import probability_mode_from_checkpoint_payload
from mirafrag.fragments import fragment_config_from_model_config
from mirafrag.losses import (
    sparse_binned_cosine_similarity,
    sparse_fragment_only_binned_cosine_similarity,
)
from mirafrag.retrieval import (
    build_retrieval_candidate_rows,
    parse_hit_ks,
    read_retrieval_candidate_table,
    resolve_candidate_mode,
    summarize_retrieval_hits,
)
from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for retrieval-style hit-rate evaluation.
    """
    parser = argparse.ArgumentParser(
        prog='mirafrag-retrieval-eval',
        description='Rank candidate molecules by predicted-vs-query spectrum similarity.',
    )
    parser.add_argument('-i', '--input', default=None, help='MassSpecGym TSV/CSV path.')
    parser.add_argument('-m', '--model', required=True, help='MiraFrag checkpoint .pt.')
    parser.add_argument('-o', '--output', default=None, help='Optional per-query CSV.')
    parser.add_argument(
        '--candidate-input',
        default=None,
        help='Optional explicit candidate table or MassSpecGym retrieval JSON.',
    )
    parser.add_argument(
        '--candidate-mode',
        choices=('auto', 'explicit', 'formula', 'mass'),
        default='auto',
        help='How to construct candidate sets when ranking molecules.',
    )
    parser.add_argument('--max-candidates', type=int, default=256)
    parser.add_argument('--query-chunk-size', type=int, default=64)
    parser.add_argument('--candidate-seed', type=int, default=0)
    parser.add_argument('--hit-ks', default='1,5,10,20')
    parser.add_argument('--score', choices=('cosine', 'sqrt_cosine'), default='cosine')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint-metric', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--split', default='test')
    parser.add_argument('--split-col', default='auto')
    parser.add_argument('--split-value', default=None)
    parser.add_argument('--mz-max', type=float, default=MASS_SPEC_GYM_MZ_MAX)
    parser.add_argument('--bin-width', type=float, default=MASS_SPEC_GYM_BIN_WIDTH)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--score-num-workers',
        type=int,
        default=None,
        help='DataLoader workers for candidate scoring; defaults to 0 with disk cache.',
    )
    parser.add_argument('--cache-chunk-size', type=int, default=1)
    parser.add_argument(
        '--memory-cache',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep decoded graphs/fragments in each worker process.',
    )
    parser.add_argument(
        '--disk-cache-dir',
        default=None,
        help='Optional disk cache for candidate graph and fragment features.',
    )
    parser.add_argument(
        '--prefill-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Fill missing disk-cache entries before ordered model inference.',
    )
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--mass-candidate-tolerance', type=float, default=0.01)
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during retrieval evaluation.',
    )
    parser.add_argument(
        '--massspecgym-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    """
    Evaluate retrieval hit rates by ranking candidate spectra for each query.
    """
    args = parse_args()
    quiet_rdkit_logs()
    hit_ks = parse_hit_ks(args.hit_ks)
    candidate_mode = resolve_candidate_mode(args.candidate_input, args.candidate_mode)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.model, device=device)
    probability_mode = probability_mode_from_checkpoint_payload(payload)
    if probability_mode == 'decoupled':
        print('Retrieval probability mode: decoupled fragment softmax with sigmoid OOS')
    validate_checkpoint_bin_config(
        model,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
    )
    graph_config = infer_graph_config(model.encoder)
    fragment_config = fragment_config_from_model_config(model.config)

    full_df = read_table(args.input)
    if args.massspecgym_filter:
        full_df = filter_massspecgym_simulation(full_df)
    query_df = select_split(
        full_df,
        split=args.split,
        split_col=args.split_col,
        split_value=args.split_value,
    )
    if args.max_rows:
        query_df = query_df.iloc[: args.max_rows].copy()
    if query_df.empty:
        raise SystemExit('No query rows selected for retrieval evaluation.')
    query_df, query_filter_stats = filter_supported_elements(
        query_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if (
        query_filter_stats['dropped_invalid_smiles']
        or query_filter_stats['dropped_unsupported_elements']
    ):
        print(f'Retrieval query element filter: {query_filter_stats}')
    if query_df.empty:
        raise SystemExit('No query rows left after encoder element filtering.')

    if args.candidate_input:
        candidate_pool = read_retrieval_candidate_table(args.candidate_input)
    else:
        candidate_pool = full_df
    if candidate_mode != 'explicit':
        candidate_pool, candidate_filter_stats = filter_supported_elements(
            candidate_pool,
            supported_atomic_numbers=graph_config.atomic_numbers,
        )
        if (
            candidate_filter_stats['dropped_invalid_smiles']
            or candidate_filter_stats['dropped_unsupported_elements']
        ):
            print(f'Retrieval candidate element filter: {candidate_filter_stats}')

    score_num_workers = _resolve_score_num_workers(args)
    print(
        f'retrieval candidate mode={candidate_mode} '
        f'queries={len(query_df)} max_candidates={args.max_candidates} '
        f'hit_ks={",".join(str(k) for k in hit_ks)} score={args.score} '
        f'cache_workers={args.num_workers} score_workers={score_num_workers}'
    )

    per_query_frames = []
    query_chunks = range(0, len(query_df), max(1, int(args.query_chunk_size)))
    chunk_progress = tqdm(
        query_chunks,
        desc='retrieval chunks',
        total=(len(query_df) + max(1, int(args.query_chunk_size)) - 1)
        // max(1, int(args.query_chunk_size)),
        dynamic_ncols=True,
        leave=False,
        disable=not args.progress,
    )
    for chunk_start in chunk_progress:
        chunk = query_df.iloc[chunk_start : chunk_start + args.query_chunk_size].copy()
        candidate_rows = build_retrieval_candidate_rows(
            chunk,
            candidate_pool,
            mode=candidate_mode,
            max_candidates=args.max_candidates,
            seed=args.candidate_seed,
            mass_tolerance=args.mass_candidate_tolerance,
        )
        if candidate_rows.empty:
            continue
        chunk_progress.set_postfix(candidates=len(candidate_rows), refresh=False)
        candidate_rows, row_filter_stats = filter_supported_elements(
            candidate_rows,
            supported_atomic_numbers=graph_config.atomic_numbers,
        )
        if candidate_rows.empty:
            continue
        if (
            row_filter_stats['dropped_invalid_smiles']
            or row_filter_stats['dropped_unsupported_elements']
        ):
            print(f'Retrieval chunk element filter: {row_filter_stats}')
        scores = _score_candidate_rows(
            model,
            candidate_rows,
            graph_config=graph_config,
            fragment_config=fragment_config,
            device=device,
            batch_size=args.batch_size,
            cache_num_workers=args.num_workers,
            score_num_workers=score_num_workers,
            memory_cache=args.memory_cache,
            disk_cache_dir=args.disk_cache_dir,
            prefill_cache=args.prefill_cache,
            cache_chunk_size=args.cache_chunk_size,
            mz_max=args.mz_max,
            bin_width=args.bin_width,
            probability_mode=probability_mode,
            score=args.score,
            show_progress=args.progress,
            split_name=f'{args.split}:{chunk_start}-{chunk_start + len(chunk) - 1}',
        )
        per_query, _summary = summarize_retrieval_hits(
            candidate_rows,
            scores,
            hit_ks=hit_ks,
        )
        per_query_frames.append(per_query)

    if not per_query_frames:
        raise SystemExit('No candidate rows could be evaluated.')
    per_query = pd.concat(per_query_frames, ignore_index=True)
    summary = _summary_from_per_query(per_query, hit_ks=hit_ks)
    print(_format_summary(summary, hit_ks=hit_ks))
    if args.output:
        per_query.to_csv(args.output, index=False)
        print(f'Wrote retrieval metrics to {args.output}')


def _resolve_score_num_workers(args: argparse.Namespace) -> int:
    """
    Choose scoring workers separately from cache-fill workers.

    Retrieval evaluation creates many short-lived candidate DataLoaders. With a
    CUDA checkpoint, multiprocessing uses spawn, and repeatedly starting worker
    pools can exhaust file descriptors. Disk-cache prefill already parallelizes
    expensive feature generation, so the safe default is single-process scoring.
    """
    if args.score_num_workers is not None:
        return max(0, int(args.score_num_workers))
    if args.disk_cache_dir is not None and args.prefill_cache:
        return 0
    return max(0, int(args.num_workers))


def _retrieval_dataloader_kwargs(*, num_workers: int, device) -> dict:
    """
    Return DataLoader kwargs for short-lived retrieval scoring loaders.
    """
    kwargs = dataloader_performance_kwargs(num_workers=num_workers, device=device)
    kwargs.pop('persistent_workers', None)
    return kwargs


def _score_candidate_rows(
    model,
    candidate_rows,
    *,
    graph_config,
    fragment_config,
    device,
    batch_size: int,
    cache_num_workers: int,
    score_num_workers: int,
    memory_cache: bool,
    disk_cache_dir: str | None,
    prefill_cache: bool,
    cache_chunk_size: int,
    mz_max: float,
    bin_width: float,
    probability_mode: str,
    score: str,
    show_progress: bool,
    split_name: str,
) -> list[float]:
    dataset = BinnedSpectrumDataset(
        candidate_rows,
        graph_config=graph_config,
        metadata_config=model.metadata_config,
        mz_max=mz_max,
        bin_width=bin_width,
        require_spectrum=True,
        memory_cache=memory_cache,
        disk_cache_dir=disk_cache_dir,
        include_fragments=True,
        fragment_config=fragment_config,
    )
    if disk_cache_dir is not None and prefill_cache:
        prefill_feature_cache(
            dataset,
            split_name=f'retrieval {split_name}',
            chunk_size=cache_chunk_size,
            num_workers=cache_num_workers,
            show_progress=False,
            print_ready=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=score_num_workers,
        collate_fn=collate_spectrum_batch,
        **_retrieval_dataloader_kwargs(
            num_workers=score_num_workers,
            device=device,
        ),
    )
    model.to(device)
    model.eval()
    scores: list[float] = []
    progress = tqdm(
        loader,
        desc='retrieval score',
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for raw_batch in progress:
        batch = _move(raw_batch, device)
        probs = model.predict_proba(batch)
        values = _retrieval_score(
            probs,
            batch,
            probability_mode=probability_mode,
            score=score,
        )
        scores.extend(float(value) for value in values.detach().cpu())
    return scores


def _retrieval_score(
    pred,
    batch,
    *,
    probability_mode: str,
    score: str,
) -> torch.Tensor:
    sqrt = score == 'sqrt_cosine'
    if probability_mode == 'decoupled':
        return sparse_fragment_only_binned_cosine_similarity(pred, batch, sqrt=sqrt)
    return sparse_binned_cosine_similarity(
        pred,
        batch,
        sqrt=sqrt,
        include_oos=False,
    )


def _summary_from_per_query(per_query, *, hit_ks: tuple[int, ...]) -> dict[str, float]:
    summary = {
        'n_queries': float(len(per_query)),
        'candidate_count_mean': float(per_query['num_candidates'].mean()),
        'mrr': float(per_query['mrr'].mean()),
    }
    for k in hit_ks:
        summary[f'hit_at_{k}'] = float(per_query[f'hit@{k}'].mean())
    return summary


def _format_summary(summary: dict[str, float], *, hit_ks: tuple[int, ...]) -> str:
    parts = [
        f'n_queries={int(summary["n_queries"])}',
        f'candidate_count_mean={summary["candidate_count_mean"]:.2f}',
        f'mrr={summary["mrr"]:.5f}',
    ]
    parts.extend(f'hit@{k}={summary[f"hit_at_{k}"]:.5f}' for k in hit_ks)
    return ' '.join(parts)


def _move(raw_batch, device):
    from mirafrag.data import move_batch_to_device

    return move_batch_to_device(raw_batch, device)


if __name__ == '__main__':
    main()
