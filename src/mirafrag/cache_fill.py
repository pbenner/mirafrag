from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from collections import deque

import torch
from tqdm.auto import tqdm

from mirafrag.chem import quiet_rdkit_logs
from mirafrag.data import BinnedSpectrumDataset

_CACHE_STATE: dict[str, BinnedSpectrumDataset] = {}


def fill_feature_cache_unordered(
    dataset: BinnedSpectrumDataset,
    *,
    desc: str,
    num_workers: int,
    chunk_size: int = 1,
    show_progress: bool = True,
) -> int:
    """
    Fill a dataset-backed feature cache with unordered multiprocessing.

    Each task materializes one dataset row, which computes and stores graph and
    fragment features through :class:`BinnedSpectrumDataset`. Unlike PyTorch's
    ordered ``DataLoader`` iteration, unordered multiprocessing lets workers
    return completed samples immediately and request more work without waiting
    for slower earlier indices. If the multiprocessing result stream stalls,
    the pending rows are reported and the caller fails instead of retrying
    the same expensive work blindly.
    """
    total = len(dataset)
    if total == 0:
        return 0
    chunk_size = max(1, int(chunk_size))
    num_workers = max(0, int(num_workers))
    if num_workers <= 0:
        return _fill_feature_cache_serial(
            dataset,
            desc=desc,
            show_progress=show_progress,
        )

    processed, _failures = _fill_feature_cache_worker_pool(
        dataset,
        desc=desc,
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
        allow_failures=False,
    )
    return processed


def fill_feature_cache_unordered_allow_failures(
    dataset: BinnedSpectrumDataset,
    *,
    desc: str,
    num_workers: int,
    chunk_size: int = 1,
    show_progress: bool = True,
) -> tuple[int, list[tuple[int, str]]]:
    """
    Fill feature caches while collecting samples that cannot be materialized.

    This is intended for retrieval candidate pools, where some decoy molecules
    may be outside the practical RDKit embedding support. Successful rows are
    cached normally; failed rows are returned to the caller so they can be
    scored as impossible instead of aborting the full evaluation.
    """
    total = len(dataset)
    if total == 0:
        return 0, []
    chunk_size = max(1, int(chunk_size))
    num_workers = max(0, int(num_workers))
    if num_workers <= 0:
        return _fill_feature_cache_serial_allow_failures(
            dataset,
            desc=desc,
            show_progress=show_progress,
        )

    return _fill_feature_cache_worker_pool(
        dataset,
        desc=desc,
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
        allow_failures=True,
    )


def prefill_feature_cache(
    dataset: BinnedSpectrumDataset,
    *,
    split_name: str,
    chunk_size: int,
    num_workers: int,
    show_progress: bool,
    print_ready: bool = True,
    ignore_errors: bool = False,
) -> list[tuple[int, str]]:
    """
    Fill missing graph and fragment cache files with optional progress output.

    Training and evaluation use this before constructing ordered DataLoaders so
    expensive cache misses are handled by the unordered worker pool instead of
    blocking model batches behind slow samples. ``print_ready`` can be disabled
    by nested callers such as retrieval evaluation to keep the outer tqdm clean.
    If ``ignore_errors`` is true, failed sample indices and error strings are
    returned instead of raising from the worker pool.
    """
    if len(dataset) == 0:
        return []
    if ignore_errors:
        total, failures = fill_feature_cache_unordered_allow_failures(
            dataset,
            desc=f'cache {split_name}',
            num_workers=num_workers,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
    else:
        total = fill_feature_cache_unordered(
            dataset,
            desc=f'cache {split_name}',
            num_workers=num_workers,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
        failures = []
    if print_ready:
        print(f'cache {split_name} ready rows={total}')
    return failures


def _fill_feature_cache_worker_pool(
    dataset: BinnedSpectrumDataset,
    *,
    desc: str,
    num_workers: int,
    chunk_size: int,
    show_progress: bool,
    allow_failures: bool,
) -> tuple[int, list[tuple[int, str]]]:
    """
    Fill cache entries with one unordered worker pool.

    The parent keeps a bounded window of tasks in flight. Whenever a worker
    returns a result, a new task is submitted immediately, so fast workers do not
    wait behind slow rows. If the in-flight window stops producing results, the
    parent reports long-pending rows periodically instead of aborting by default.
    Timeouts remain available through MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS and
    MIRAFRAG_CACHE_POOL_STALL_SECONDS for debugging stuck workers.
    """
    total = len(dataset)
    context = mp.get_context(_multiprocessing_start_method())
    task_timeout_seconds = _cache_task_timeout_seconds()
    stall_seconds = _cache_pool_stall_seconds(default_seconds=task_timeout_seconds)
    print(
        f'{desc}: cache worker pool start_method={context.get_start_method()} '
        f'workers={num_workers} chunk_size={chunk_size} '
        f'task_timeout={_format_timeout_seconds(task_timeout_seconds)} '
        f'stall_timeout={_format_timeout_seconds(stall_seconds)}',
        file=sys.stderr,
        flush=True,
    )
    poll_seconds = _cache_pool_poll_seconds()
    status_seconds = _cache_pool_status_seconds()
    completed: set[int] = set()
    failures: list[tuple[int, str]] = []
    worker_fn = (
        _cache_dataset_index_chunk_allow_failure
        if allow_failures
        else _cache_dataset_index_chunk
    )
    remaining = deque(range(total))
    max_pending = max(1, int(num_workers) * 2)
    progress = tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    pool = _create_cache_pool(
        context=context,
        dataset=dataset,
        num_workers=num_workers,
    )
    pool_closed = False
    pending: dict[object, tuple[tuple[int, ...], float]] = {}
    try:
        _submit_cache_tasks(
            pool=pool,
            worker_fn=worker_fn,
            remaining=remaining,
            pending=pending,
            chunk_size=chunk_size,
            max_pending=max_pending,
        )
        last_result_time = time.monotonic()
        last_status_time = last_result_time
        while pending:
            now = time.monotonic()
            made_progress = False
            for result, (chunk, start_time) in list(pending.items()):
                if result.ready():
                    try:
                        batch = result.get()
                    except Exception as exc:
                        description = _describe_cache_indices(dataset, chunk)
                        raise RuntimeError(
                            f'{desc}: cache worker failed for {description}.'
                        ) from exc
                    del pending[result]
                    made_progress = True
                    last_result_time = now
                    for row_result in batch:
                        if allow_failures:
                            idx, _elapsed, error = row_result
                            if error is not None and int(idx) not in completed:
                                failures.append((int(idx), str(error)))
                        else:
                            idx, _elapsed = row_result
                        idx = int(idx)
                        if idx not in completed:
                            completed.add(idx)
                            progress.update(1)
                    _submit_cache_tasks(
                        pool=pool,
                        worker_fn=worker_fn,
                        remaining=remaining,
                        pending=pending,
                        chunk_size=chunk_size,
                        max_pending=max_pending,
                    )
                    continue

                if (
                    task_timeout_seconds is not None
                    and now - start_time > task_timeout_seconds
                ):
                    description = _describe_cache_indices(dataset, chunk)
                    if not allow_failures:
                        raise TimeoutError(
                            f'{desc}: cache worker task exceeded '
                            f'{task_timeout_seconds:g}s for {description}.'
                        )
                    message = (
                        'cache worker task exceeded '
                        f'{task_timeout_seconds:g}s for {description}'
                    )
                    print(
                        f'{desc}: skipping timed-out cache task; {message}',
                        file=sys.stderr,
                        flush=True,
                    )
                    _mark_cache_chunk_failed(
                        chunk,
                        message=message,
                        failures=failures,
                        completed=completed,
                        progress=progress,
                    )
                    del pending[result]
                    _requeue_pending_cache_chunks(
                        pending=pending,
                        remaining=remaining,
                        completed=completed,
                    )
                    pending.clear()
                    pool.terminate()
                    pool.join()
                    pool = _create_cache_pool(
                        context=context,
                        dataset=dataset,
                        num_workers=num_workers,
                    )
                    made_progress = True
                    _submit_cache_tasks(
                        pool=pool,
                        worker_fn=worker_fn,
                        remaining=remaining,
                        pending=pending,
                        chunk_size=chunk_size,
                        max_pending=max_pending,
                    )
                    restart_time = time.monotonic()
                    last_result_time = restart_time
                    last_status_time = restart_time
                    break

            if made_progress:
                continue
            if (
                stall_seconds is not None
                and time.monotonic() - last_result_time > stall_seconds
            ):
                pending_description = _describe_pending_cache_tasks(dataset, pending)
                if allow_failures:
                    print(
                        f'{desc}: skipping stalled cache tasks; cache worker pool '
                        f'produced no results for {stall_seconds:g}s. Pending '
                        f'tasks: {pending_description}',
                        file=sys.stderr,
                        flush=True,
                    )
                    _mark_pending_cache_chunks_failed(
                        dataset=dataset,
                        pending=pending,
                        stall_seconds=stall_seconds,
                        max_chunks=num_workers,
                        failures=failures,
                        completed=completed,
                        progress=progress,
                    )
                    _requeue_pending_cache_chunks(
                        pending=pending,
                        remaining=remaining,
                        completed=completed,
                    )
                    pending.clear()
                    pool.terminate()
                    pool.join()
                    pool = _create_cache_pool(
                        context=context,
                        dataset=dataset,
                        num_workers=num_workers,
                    )
                    _submit_cache_tasks(
                        pool=pool,
                        worker_fn=worker_fn,
                        remaining=remaining,
                        pending=pending,
                        chunk_size=chunk_size,
                        max_pending=max_pending,
                    )
                    restart_time = time.monotonic()
                    last_result_time = restart_time
                    last_status_time = restart_time
                    continue
                raise TimeoutError(
                    f'{desc}: cache worker pool produced no results for '
                    f'{stall_seconds:g}s. Pending tasks: {pending_description}'
                )
            if time.monotonic() - last_status_time > status_seconds:
                pending_description = _describe_pending_cache_tasks(dataset, pending)
                print(
                    f'{desc}: cache worker pool waiting; pending tasks: '
                    f'{pending_description}',
                    file=sys.stderr,
                    flush=True,
                )
                last_status_time = time.monotonic()
            time.sleep(poll_seconds)
        pool.close()
        pool_closed = True
    except Exception:
        pool.terminate()
        pool_closed = True
        raise
    finally:
        if not pool_closed:
            pool.terminate()
        pool.join()
        progress.close()

    return len(completed), failures


def _submit_cache_tasks(
    *,
    pool,
    worker_fn,
    remaining: deque[int],
    pending: dict[object, tuple[tuple[int, ...], float]],
    chunk_size: int,
    max_pending: int,
) -> None:
    while remaining and len(pending) < max_pending:
        chunk: list[int] = []
        while remaining and len(chunk) < chunk_size:
            chunk.append(int(remaining.popleft()))
        if not chunk:
            break
        result = pool.apply_async(worker_fn, (tuple(chunk),))
        pending[result] = (tuple(chunk), time.monotonic())


def _create_cache_pool(*, context, dataset: BinnedSpectrumDataset, num_workers: int):
    return context.Pool(
        processes=num_workers,
        initializer=_init_cache_worker,
        initargs=(dataset,),
        maxtasksperchild=_cache_pool_max_tasks_per_child(),
    )


def _mark_cache_chunk_failed(
    chunk: tuple[int, ...],
    *,
    message: str,
    failures: list[tuple[int, str]],
    completed: set[int],
    progress,
) -> None:
    for raw_idx in chunk:
        idx = int(raw_idx)
        if idx in completed:
            continue
        completed.add(idx)
        failures.append((idx, message))
        progress.update(1)


def _mark_pending_cache_chunks_failed(
    *,
    dataset: BinnedSpectrumDataset,
    pending: dict[object, tuple[tuple[int, ...], float]],
    stall_seconds: float,
    max_chunks: int,
    failures: list[tuple[int, str]],
    completed: set[int],
    progress,
) -> None:
    oldest_chunks = sorted(pending.values(), key=lambda item: item[1])[
        : max(1, int(max_chunks))
    ]
    for chunk, _start_time in oldest_chunks:
        description = _describe_cache_indices(dataset, chunk)
        _mark_cache_chunk_failed(
            chunk,
            message=(
                'cache worker pool produced no results for '
                f'{stall_seconds:g}s for {description}'
            ),
            failures=failures,
            completed=completed,
            progress=progress,
        )


def _requeue_pending_cache_chunks(
    *,
    pending: dict[object, tuple[tuple[int, ...], float]],
    remaining: deque[int],
    completed: set[int],
) -> None:
    chunks = [chunk for chunk, _start_time in pending.values()]
    for chunk in reversed(chunks):
        for raw_idx in reversed(chunk):
            idx = int(raw_idx)
            if idx not in completed:
                remaining.appendleft(idx)


def _cache_pool_stall_seconds(*, default_seconds: float | None = None) -> float | None:
    if 'MIRAFRAG_CACHE_POOL_STALL_SECONDS' in os.environ:
        return _optional_cache_timeout_seconds(
            'MIRAFRAG_CACHE_POOL_STALL_SECONDS', default='0'
        )
    return default_seconds


def _cache_task_timeout_seconds() -> float | None:
    return _optional_cache_timeout_seconds(
        'MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS', default='0'
    )


def _cache_pool_status_seconds() -> float:
    return max(
        1.0,
        float(os.environ.get('MIRAFRAG_CACHE_POOL_STATUS_SECONDS', '300')),
    )


def _optional_cache_timeout_seconds(name: str, *, default: str) -> float | None:
    seconds = float(os.environ.get(name, default))
    if seconds <= 0:
        return None
    return max(0.1, seconds)


def _format_timeout_seconds(seconds: float | None) -> str:
    return 'disabled' if seconds is None else f'{seconds:g}s'


def _cache_pool_poll_seconds() -> float:
    return max(
        0.05,
        float(os.environ.get('MIRAFRAG_CACHE_POOL_POLL_SECONDS', '0.25')),
    )


def _cache_pool_max_tasks_per_child() -> int:
    return max(1, int(os.environ.get('MIRAFRAG_CACHE_POOL_MAX_TASKS', '256')))


def _describe_pending_cache_tasks(
    dataset: BinnedSpectrumDataset,
    pending: dict[object, tuple[tuple[int, ...], float]],
    *,
    limit: int = 8,
) -> str:
    now = time.monotonic()
    entries: list[tuple[float, str]] = []
    for chunk, start_time in pending.values():
        elapsed = now - start_time
        entries.append((elapsed, _describe_cache_indices(dataset, chunk)))
    entries.sort(reverse=True, key=lambda item: item[0])
    shown = [
        f'{description} elapsed={elapsed:.1f}s'
        for elapsed, description in entries[:limit]
    ]
    if len(entries) > limit:
        shown.append(f'... {len(entries) - limit} more pending tasks')
    return '; '.join(shown) if shown else 'none'


def _describe_cache_indices(
    dataset: BinnedSpectrumDataset, indices: tuple[int, ...]
) -> str:
    if len(indices) == 1:
        prefix = f'idx={int(indices[0])}'
    else:
        first = int(indices[0])
        last = int(indices[-1])
        prefix = f'indices={first}-{last} n={len(indices)}'

    df = getattr(dataset, 'df', None)
    if df is None or len(indices) != 1:
        return prefix

    idx = int(indices[0])
    try:
        row = df.iloc[idx]
    except Exception:
        return prefix

    fields: list[str] = []
    for column_name, label in (
        ('identifier', 'identifier'),
        (getattr(dataset, 'smiles_col', None), 'smiles'),
        (getattr(dataset, 'adduct_col', None), 'adduct'),
        (getattr(dataset, 'instrument_col', None), 'instrument'),
        (getattr(dataset, 'ce_col', None), 'collision_energy'),
    ):
        if not column_name or column_name not in row.index:
            continue
        value = row.get(column_name)
        if value is None:
            continue
        fields.append(f'{label}={value!r}')
    if not fields:
        return prefix
    return f'{prefix} ' + ' '.join(fields)


def _multiprocessing_start_method() -> str:
    """
    Return a cache-fill start method that avoids unsafe threaded forks.

    Training imports PyTorch/RDKit before cache prefill, so the parent process
    can already be multithreaded. Forking such a process can leave inherited
    locks permanently held in workers. Use forkserver where available, falling
    back to spawn, and allow an environment override for local debugging.
    """
    requested = os.environ.get('MIRAFRAG_CACHE_START_METHOD')
    if requested:
        if requested not in mp.get_all_start_methods():
            raise ValueError(
                'MIRAFRAG_CACHE_START_METHOD must be one of '
                f'{mp.get_all_start_methods()}, got {requested!r}.'
            )
        return requested
    methods = mp.get_all_start_methods()
    if 'forkserver' in methods and not (
        torch.cuda.is_available() and torch.cuda.is_initialized()
    ):
        return 'forkserver'
    return 'spawn'


def _fill_feature_cache_serial_allow_failures(
    dataset: BinnedSpectrumDataset,
    *,
    desc: str,
    show_progress: bool,
) -> tuple[int, list[tuple[int, str]]]:
    progress = tqdm(
        range(len(dataset)),
        desc=desc,
        total=len(dataset),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    processed = 0
    failures: list[tuple[int, str]] = []
    for idx in progress:
        processed += 1
        try:
            _materialize_dataset_features(dataset, int(idx))
        except Exception as exc:  # noqa: BLE001
            failures.append((int(idx), f'{type(exc).__name__}: {exc}'))
    return processed, failures


def _fill_feature_cache_serial(
    dataset: BinnedSpectrumDataset,
    *,
    desc: str,
    show_progress: bool,
) -> int:
    """
    Fill feature caches in the current process.

    This path is used for ``num_workers <= 0`` and is useful for debugging
    because exceptions are raised without multiprocessing wrappers.
    """
    progress = tqdm(
        range(len(dataset)),
        desc=desc,
        total=len(dataset),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    processed = 0
    for idx in progress:
        _materialize_dataset_features(dataset, int(idx))
        processed += 1
    return processed


def _init_cache_worker(dataset: BinnedSpectrumDataset) -> None:
    """
    Store the dataset in a worker-global variable and silence RDKit logs.
    """
    quiet_rdkit_logs()
    _CACHE_STATE['dataset'] = dataset


def _cache_dataset_index(idx: int) -> tuple[int, float]:
    """
    Materialize one dataset row inside a worker and return timing metadata.
    """
    dataset = _CACHE_STATE.get('dataset')
    if dataset is None:
        raise RuntimeError('Cache worker dataset was not initialized.')
    start = time.perf_counter()
    _materialize_dataset_features(dataset, int(idx))
    return int(idx), time.perf_counter() - start


def _cache_dataset_index_chunk(indices: tuple[int, ...]) -> list[tuple[int, float]]:
    return [_cache_dataset_index(int(idx)) for idx in indices]


def _cache_dataset_index_chunk_allow_failure(
    indices: tuple[int, ...],
) -> list[tuple[int, float, str | None]]:
    return [_cache_dataset_index_allow_failure(int(idx)) for idx in indices]


def _materialize_dataset_features(dataset, idx: int) -> None:
    """
    Compute cacheable features without constructing training targets when possible.
    """
    materialize = getattr(dataset, 'materialize_feature_cache', None)
    if callable(materialize):
        materialize(int(idx))
    else:
        dataset[int(idx)]


def _cache_dataset_index_allow_failure(idx: int) -> tuple[int, float, str | None]:
    """
    Materialize one dataset row and return an error string instead of raising.
    """
    start = time.perf_counter()
    try:
        _cache_dataset_index(idx)
    except Exception as exc:  # noqa: BLE001
        return int(idx), time.perf_counter() - start, f'{type(exc).__name__}: {exc}'
    return int(idx), time.perf_counter() - start, None
