from __future__ import annotations

import argparse
import signal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mirafrag.cache_fill import prefill_feature_cache
from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import infer_graph_config, quiet_rdkit_logs
from mirafrag.cli.common import (
    add_high_ce_fragment_support_args,
    apply_fragment_args_to_model_config,
    resolve_device,
    validate_checkpoint_bin_config,
)
from mirafrag.data import (
    ADDUCT_ALIASES,
    CE_ALIASES,
    INSTRUMENT_ALIASES,
    PRECURSOR_ALIASES,
    RAW_COLLISION_ENERGY_COLUMN,
    BinnedSpectrumDataset,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    find_column,
    normalize_collision_energy_dataframe,
    read_table,
    select_split,
)
from mirafrag.evaluation import probability_mode_from_checkpoint_payload
from mirafrag.fragments import fragment_support_profile_from_model_config
from mirafrag.losses import (
    sparse_binned_cosine_similarity,
    sparse_fragment_only_binned_cosine_similarity,
)
from mirafrag.retrieval import (
    build_retrieval_candidate_rows,
    estimate_retrieval_candidate_count,
    parse_hit_ks,
    read_retrieval_candidate_table,
    resolve_candidate_mode,
    summarize_retrieval_hits,
)
from mirafrag.sparse_spectra import (
    SparseSpectrum,
    normalize_sparse_spectrum,
    sparse_cosine,
    sparse_from_peaks,
)
from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    parse_peaks,
)


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
    parser.add_argument(
        '--fast-retrieval',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Score candidate predictions against parsed query spectra outside the '
            'model DataLoader. This avoids duplicating target-spectrum parsing for '
            'every candidate row.'
        ),
    )
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
        '--score-sample-timeout',
        type=float,
        default=300.0,
        help=(
            'Maximum seconds spent building one candidate inside fast retrieval '
            'scoring workers. Non-positive disables the timeout.'
        ),
    )
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
    parser.add_argument(
        '--row-offset',
        type=int,
        default=0,
        help='Skip this many selected query rows before --max-rows is applied.',
    )
    parser.add_argument(
        '--sample-rows',
        type=int,
        default=None,
        help='Randomly sample this many query rows after split/filter selection.',
    )
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--mass-candidate-tolerance', type=float, default=0.01)
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars during retrieval evaluation.',
    )
    add_high_ce_fragment_support_args(parser)
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
    apply_fragment_args_to_model_config(model.config, args)
    graph_config = infer_graph_config(model.encoder)
    fragment_support_profile = fragment_support_profile_from_model_config(model.config)

    full_df = read_table(args.input)
    if args.massspecgym_filter:
        full_df = filter_massspecgym_simulation(full_df)
    query_df = select_split(
        full_df,
        split=args.split,
        split_col=args.split_col,
        split_value=args.split_value,
    )
    if args.row_offset < 0:
        raise SystemExit('--row-offset must be non-negative.')
    if args.row_offset:
        query_df = query_df.iloc[args.row_offset :].copy()
    if args.max_rows:
        query_df = query_df.iloc[: args.max_rows].copy()
    if query_df.empty:
        raise SystemExit('No query rows selected for retrieval evaluation.')
    query_df, query_filter_stats = filter_supported_elements(
        query_df,
        supported_atomic_numbers=graph_config.atomic_numbers,
    )
    if args.sample_rows is not None:
        if args.sample_rows <= 0:
            raise SystemExit('--sample-rows must be positive when provided.')
        if args.sample_rows < len(query_df):
            query_df = query_df.sample(
                n=args.sample_rows,
                random_state=int(args.sample_seed),
            ).sort_index()
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
    fast_prediction_cache: dict[tuple[object, ...], SparseSpectrum | None] = {}
    query_chunk_size = max(1, int(args.query_chunk_size))
    query_chunks = range(0, len(query_df), query_chunk_size)
    candidate_total = estimate_retrieval_candidate_count(
        query_df,
        candidate_pool,
        mode=candidate_mode,
        max_candidates=args.max_candidates,
    )
    candidate_progress = tqdm(
        total=candidate_total,
        desc='retrieval candidates',
        unit='cand',
        dynamic_ncols=True,
        leave=False,
        disable=not args.progress,
    )
    for chunk_index, chunk_start in enumerate(query_chunks, start=1):
        chunk = query_df.iloc[chunk_start : chunk_start + query_chunk_size].copy()
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
        candidate_progress.set_postfix(
            chunk=f'{chunk_index}',
            candidates=len(candidate_rows),
            refresh=False,
        )
        candidate_rows, row_filter_stats = filter_supported_elements(
            candidate_rows,
            supported_atomic_numbers=graph_config.atomic_numbers,
        )
        if candidate_rows.empty:
            continue
        if _checkpoint_uses_normalized_ce(model):
            candidate_rows = normalize_collision_energy_dataframe(
                candidate_rows,
                metadata_config=model.metadata_config,
            )
        if (
            row_filter_stats['dropped_invalid_smiles']
            or row_filter_stats['dropped_unsupported_elements']
        ):
            print(f'Retrieval chunk element filter: {row_filter_stats}')
        score_kwargs = dict(
            graph_config=graph_config,
            fragment_support_profile=fragment_support_profile,
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
            score_sample_timeout=args.score_sample_timeout,
        )
        if args.fast_retrieval:
            scores = _score_candidate_rows_fast(
                model,
                candidate_rows,
                prediction_cache=fast_prediction_cache,
                **score_kwargs,
            )
        else:
            scores = _score_candidate_rows(
                model,
                candidate_rows,
                **score_kwargs,
            )
        per_query, _summary = summarize_retrieval_hits(
            candidate_rows,
            scores,
            hit_ks=hit_ks,
        )
        per_query_frames.append(per_query)
        candidate_progress.update(len(candidate_rows))

    if (
        candidate_progress.total is not None
        and candidate_progress.n < candidate_progress.total
    ):
        candidate_progress.total = candidate_progress.n
        candidate_progress.refresh()
    candidate_progress.close()

    if not per_query_frames:
        raise SystemExit('No candidate rows could be evaluated.')
    per_query = pd.concat(per_query_frames, ignore_index=True)
    summary = _summary_from_per_query(per_query, hit_ks=hit_ks)
    print(_format_summary(summary, hit_ks=hit_ks))
    if args.output:
        per_query.to_csv(args.output, index=False)
        print(f'Wrote retrieval metrics to {args.output}')


def _checkpoint_uses_normalized_ce(model) -> bool:
    return (
        str(
            getattr(model.metadata_config, 'collision_energy_mode', 'raw') or 'raw'
        ).lower()
        == 'normalized'
    )


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
    fragment_support_profile,
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
    score_sample_timeout: float = 0.0,
) -> list[float]:
    del score_sample_timeout
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
        fragment_support_profile=fragment_support_profile,
    )
    num_candidate_rows = len(candidate_rows)
    score_positions = list(range(num_candidate_rows))
    if disk_cache_dir is not None and prefill_cache:
        failures = prefill_feature_cache(
            dataset,
            split_name=f'retrieval {split_name}',
            chunk_size=cache_chunk_size,
            num_workers=cache_num_workers,
            show_progress=show_progress,
            print_ready=False,
            ignore_errors=True,
        )
        if failures:
            failed_positions = {idx for idx, _error in failures}
            tqdm.write(
                f'retrieval {split_name}: assigning -inf to '
                f'{len(failures)}/{len(candidate_rows)} unscoreable candidates'
            )
            for idx, error in failures[:3]:
                tqdm.write(f'  example idx={idx}: {_shorten_error(error)}')
            score_positions = [
                idx for idx in score_positions if idx not in failed_positions
            ]
            if not score_positions:
                return [float('-inf')] * num_candidate_rows
            candidate_rows = candidate_rows.iloc[score_positions].reset_index(drop=True)
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
                fragment_support_profile=fragment_support_profile,
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
    scored_values: list[float] = []
    progress = tqdm(
        loader,
        desc='retrieval score',
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    with torch.no_grad():
        for raw_batch in progress:
            batch = _move(raw_batch, device)
            probs = model.predict_proba(batch)
            values = _retrieval_score(
                probs,
                batch,
                probability_mode=probability_mode,
                score=score,
            )
            scored_values.extend(float(value) for value in values.detach().cpu())
    scores = [float('-inf')] * num_candidate_rows
    if len(scored_values) != len(score_positions):
        raise RuntimeError('Retrieval scoring produced an unexpected number of scores.')
    for position, value in zip(score_positions, scored_values, strict=True):
        scores[position] = value
    return scores


class _SafeRetrievalDataset:
    def __init__(self, dataset, *, timeout_seconds: float = 0.0) -> None:
        self.dataset = dataset
        self.timeout_seconds = max(0.0, float(timeout_seconds))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        try:
            item = self._run_with_timeout(lambda: self.dataset[int(idx)])
            item['_retrieval_position'] = int(idx)
            return item
        except Exception as exc:  # noqa: BLE001 - retrieval should skip bad candidates.
            return {
                '_retrieval_failed': True,
                '_retrieval_position': int(idx),
                '_retrieval_error': str(exc),
            }

    def materialize_feature_cache(self, idx: int) -> None:
        def materialize() -> None:
            materialize_cache = getattr(self.dataset, 'materialize_feature_cache', None)
            if callable(materialize_cache):
                materialize_cache(int(idx))
            else:
                self.dataset[int(idx)]

        self._run_with_timeout(materialize)

    def _run_with_timeout(self, fn):
        previous_handler = None
        timeout_active = self.timeout_seconds > 0.0
        try:
            if timeout_active:
                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _retrieval_timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
            return fn()
        finally:
            if timeout_active:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)


def _retrieval_timeout_handler(_signum, _frame) -> None:
    raise TimeoutError('retrieval candidate feature generation timed out')


def _collate_retrieval_batch(items: list[dict]) -> dict:
    good = [item for item in items if not item.get('_retrieval_failed')]
    failed = [item for item in items if item.get('_retrieval_failed')]
    failed_positions = torch.as_tensor(
        [int(item['_retrieval_position']) for item in failed],
        dtype=torch.long,
    )
    failed_errors = [
        str(item.get('_retrieval_error', 'unknown retrieval error')) for item in failed
    ]
    if not good:
        return {
            '_retrieval_empty': True,
            '_retrieval_positions': torch.empty(0, dtype=torch.long),
            '_retrieval_failed_positions': failed_positions,
            '_retrieval_errors': failed_errors,
        }
    positions = torch.as_tensor(
        [int(item.pop('_retrieval_position')) for item in good],
        dtype=torch.long,
    )
    batch = collate_spectrum_batch(good)
    batch['_retrieval_empty'] = False
    batch['_retrieval_positions'] = positions
    batch['_retrieval_failed_positions'] = failed_positions
    batch['_retrieval_errors'] = failed_errors
    return batch


def _score_candidate_rows_fast(
    model,
    candidate_rows,
    *,
    graph_config,
    fragment_support_profile,
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
    score_sample_timeout: float = 300.0,
    prediction_cache: dict[tuple[object, ...], SparseSpectrum | None] | None = None,
) -> list[float]:
    target_spectra = _target_spectrum_lookup(
        candidate_rows,
        mz_max=mz_max,
        bin_width=bin_width,
    )
    num_candidate_rows = len(candidate_rows)
    keys = [_prediction_cache_key(row) for _idx, row in candidate_rows.iterrows()]
    if prediction_cache is None:
        prediction_cache = {}
    missing_positions = [
        idx for idx, key in enumerate(keys) if key not in prediction_cache
    ]
    if missing_positions:
        missing_rows = candidate_rows.iloc[missing_positions].reset_index(drop=True)
    else:
        missing_rows = candidate_rows.iloc[0:0].copy()
    failed_positions: set[int] = set()

    score_original_positions = list(missing_positions)
    dataset = _SafeRetrievalDataset(
        BinnedSpectrumDataset(
            missing_rows,
            graph_config=graph_config,
            metadata_config=model.metadata_config,
            mz_max=mz_max,
            bin_width=bin_width,
            require_spectrum=False,
            memory_cache=memory_cache,
            disk_cache_dir=disk_cache_dir,
            include_fragments=True,
            fragment_support_profile=fragment_support_profile,
        ),
        timeout_seconds=score_sample_timeout,
    )
    score_positions = list(range(len(missing_rows)))
    if disk_cache_dir is not None and prefill_cache and len(missing_rows) > 0:
        failures = prefill_feature_cache(
            dataset,
            split_name=f'retrieval {split_name}',
            chunk_size=cache_chunk_size,
            num_workers=cache_num_workers,
            show_progress=show_progress,
            print_ready=False,
            ignore_errors=True,
        )
        if failures:
            failed_local_positions = {idx for idx, _error in failures}
            tqdm.write(
                f'retrieval {split_name}: assigning -inf to '
                f'{len(failures)}/{len(missing_rows)} unscoreable uncached candidates'
            )
            for idx, error in failures[:3]:
                tqdm.write(f'  example idx={idx}: {_shorten_error(error)}')
            score_positions = [
                idx for idx in score_positions if idx not in failed_local_positions
            ]
            for local_idx in failed_local_positions:
                original_idx = missing_positions[local_idx]
                prediction_cache[keys[original_idx]] = None
                failed_positions.add(original_idx)
            score_original_positions = [
                missing_positions[idx] for idx in score_positions
            ]
            missing_rows = missing_rows.iloc[score_positions].reset_index(drop=True)
            dataset = _SafeRetrievalDataset(
                BinnedSpectrumDataset(
                    missing_rows,
                    graph_config=graph_config,
                    metadata_config=model.metadata_config,
                    mz_max=mz_max,
                    bin_width=bin_width,
                    require_spectrum=False,
                    memory_cache=memory_cache,
                    disk_cache_dir=disk_cache_dir,
                    include_fragments=True,
                    fragment_support_profile=fragment_support_profile,
                ),
                timeout_seconds=score_sample_timeout,
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=score_num_workers,
        collate_fn=_collate_retrieval_batch,
        **_retrieval_dataloader_kwargs(
            num_workers=score_num_workers,
            device=device,
        ),
    )
    model.to(device)
    model.eval()
    progress = tqdm(
        loader,
        desc='retrieval score fast',
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    scored_count = 0
    with torch.no_grad():
        for raw_batch in progress:
            failed_local_positions = raw_batch.pop('_retrieval_failed_positions')
            failed_errors = raw_batch.pop('_retrieval_errors')
            for local_position, error in zip(
                failed_local_positions.tolist(),
                failed_errors,
                strict=True,
            ):
                original_position = score_original_positions[int(local_position)]
                prediction_cache[keys[original_position]] = None
                failed_positions.add(original_position)
                if len(failed_positions) <= 3:
                    tqdm.write(
                        f'retrieval {split_name}: assigning -inf to candidate '
                        f'{original_position}: {_shorten_error(error)}'
                    )
            local_positions = raw_batch.pop('_retrieval_positions')
            empty_batch = bool(raw_batch.pop('_retrieval_empty'))
            if empty_batch:
                continue
            batch = _move(raw_batch, device)
            probs = model.predict_proba(batch)
            spectra = _prediction_sparse_spectra(
                probs,
                probability_mode=probability_mode,
            )
            for spectrum, local_position in zip(
                spectra,
                local_positions.tolist(),
                strict=True,
            ):
                original_position = score_original_positions[int(local_position)]
                prediction_cache[keys[original_position]] = spectrum
                scored_count += 1

    expected_scored = sum(
        1 for position in score_original_positions if position not in failed_positions
    )
    if scored_count != expected_scored:
        raise RuntimeError(
            'Retrieval scoring produced an unexpected number of spectra.'
        )

    scores = [float('-inf')] * num_candidate_rows
    for position, (_idx, row) in enumerate(candidate_rows.iterrows()):
        if position in failed_positions:
            continue
        spectrum = prediction_cache.get(keys[position])
        if spectrum is None:
            continue
        target = target_spectra[str(row['_retrieval_query_identifier'])]
        scores[position] = sparse_cosine(
            spectrum,
            target,
            sqrt=(score == 'sqrt_cosine'),
        )
    return scores


def _prediction_cache_key(row: pd.Series) -> tuple[object, ...]:
    columns = []
    for aliases in (
        ('smiles', 'SMILES', 'Smiles'),
        PRECURSOR_ALIASES,
        ADDUCT_ALIASES,
        INSTRUMENT_ALIASES,
        CE_ALIASES,
    ):
        column = next(
            (candidate for candidate in aliases if candidate in row.index),
            None,
        )
        if column is not None:
            columns.append(column)
    if RAW_COLLISION_ENERGY_COLUMN in row.index:
        columns.append(RAW_COLLISION_ENERGY_COLUMN)
    key = []
    for column in columns:
        key.append((column, _cacheable_value(row.get(column))))
    return tuple(key)


def _cacheable_value(value) -> object:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_cacheable_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_cacheable_value(item) for item in value.tolist())
    return str(value) if isinstance(value, pd.Timestamp) else value


def _target_spectrum_lookup(
    candidate_rows: pd.DataFrame,
    *,
    mz_max: float,
    bin_width: float,
) -> dict[str, SparseSpectrum]:
    precursor_col = find_column(candidate_rows, PRECURSOR_ALIASES, required=False)
    spectra: dict[str, SparseSpectrum] = {}
    for query_id, group in candidate_rows.groupby(
        '_retrieval_query_identifier',
        sort=False,
    ):
        row = group.iloc[0]
        precursor_mz = row.get(precursor_col) if precursor_col is not None else None
        mzs, intensities = parse_peaks(
            row,
            precursor_mz=precursor_mz,
            exclude_precursor=True,
        )
        spectra[str(query_id)] = sparse_from_peaks(
            mzs,
            intensities,
            mz_max=mz_max,
            bin_width=bin_width,
        )
    return spectra


def _prediction_sparse_spectra(
    pred: dict[str, object],
    *,
    probability_mode: str,
) -> list[SparseSpectrum]:
    if probability_mode == 'decoupled':
        values = _decoupled_fragment_probabilities(pred)
    elif probability_mode == 'joint':
        from mirafrag.probability import fragment_oos_log_probs

        fragment_log_probs, _oos_log_probs = fragment_oos_log_probs(pred)
        values = torch.exp(fragment_log_probs)
    else:
        raise ValueError("probability_mode must be one of: 'joint', 'decoupled'.")

    bins = pred['bins'].detach().long().cpu().numpy()
    batch = pred['batch'].detach().long().cpu().numpy()
    values_np = values.detach().cpu().numpy().astype(np.float32, copy=False)
    rows: list[SparseSpectrum] = []
    for batch_idx in range(int(pred['batch_size'])):
        mask = batch == batch_idx
        rows.append(
            normalize_sparse_spectrum(
                SparseSpectrum(bins=bins[mask], values=values_np[mask])
            )
        )
    return rows


def _decoupled_fragment_probabilities(pred: dict[str, object]) -> torch.Tensor:
    logits = pred['logits']
    batch = pred['batch'].long()
    out = torch.empty_like(logits)
    for batch_idx in range(int(pred['batch_size'])):
        mask = batch == batch_idx
        if bool(mask.any()):
            out[mask] = torch.softmax(logits[mask], dim=0)
    return out


def _shorten_error(error: str, *, max_length: int = 180) -> str:
    """
    Return a one-line, tqdm-friendly error summary.
    """
    text = ' '.join(str(error).split())
    if len(text) <= max_length:
        return text
    return f'{text[: max_length - 3]}...'


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
