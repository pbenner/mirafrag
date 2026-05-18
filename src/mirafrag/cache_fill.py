from __future__ import annotations

import multiprocessing as mp
import time

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
    for slower earlier indices.
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

    context = mp.get_context(_multiprocessing_start_method())
    processed = 0
    with context.Pool(
        processes=num_workers,
        initializer=_init_cache_worker,
        initargs=(dataset,),
    ) as pool:
        progress = tqdm(
            pool.imap_unordered(
                _cache_dataset_index, range(total), chunksize=chunk_size
            ),
            desc=desc,
            total=total,
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for _idx, _elapsed in progress:
            processed += 1
    return processed


def prefill_feature_cache(
    dataset: BinnedSpectrumDataset,
    *,
    split_name: str,
    chunk_size: int,
    num_workers: int,
    show_progress: bool,
) -> None:
    """
    Fill missing graph and fragment cache files with a visible progress bar.

    Training and evaluation use this before constructing ordered DataLoaders so
    expensive cache misses are handled by the unordered worker pool instead of
    blocking model batches behind slow samples.
    """
    if len(dataset) == 0:
        return
    total = fill_feature_cache_unordered(
        dataset,
        desc=f'cache {split_name}',
        num_workers=num_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    print(f'cache {split_name} ready rows={total}')


def _multiprocessing_start_method() -> str:
    """
    Return a cache-fill start method that avoids unsafe CUDA forks.

    Fork is fastest and shares the dataframe copy-on-write when CUDA has not
    been initialized. If CUDA is already initialized, spawn is safer because
    PyTorch does not support reinitializing CUDA state in forked children.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return 'spawn'
    return 'fork'


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
        dataset[int(idx)]
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
    dataset[int(idx)]
    return int(idx), time.perf_counter() - start
