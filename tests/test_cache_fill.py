import time

import pytest

from mirafrag.cache_fill import (
    _cache_pool_stall_seconds,
    _cache_task_timeout_seconds,
    prefill_feature_cache,
)


class SimpleDataset:
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return int(idx)


class FailingDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        if int(idx) == 1:
            raise ValueError('bad sample')
        return int(idx)


class SlowDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        time.sleep(5.0)
        return int(idx)


def test_prefill_feature_cache_collects_failures_in_ignore_error_mode():
    failures = prefill_feature_cache(
        FailingDataset(),
        split_name='test',
        chunk_size=1,
        num_workers=0,
        show_progress=False,
        print_ready=False,
        ignore_errors=True,
    )

    assert failures == [(1, 'ValueError: bad sample')]


def test_prefill_feature_cache_raises_by_default():
    with pytest.raises(ValueError, match='bad sample'):
        prefill_feature_cache(
            FailingDataset(),
            split_name='test',
            chunk_size=1,
            num_workers=0,
            show_progress=False,
            print_ready=False,
        )


def test_prefill_feature_cache_parallel_completes():
    failures = prefill_feature_cache(
        SimpleDataset(),
        split_name='test',
        chunk_size=2,
        num_workers=2,
        show_progress=False,
        print_ready=False,
        ignore_errors=True,
    )

    assert failures == []


def test_prefill_feature_cache_parallel_stall_collects_failures(monkeypatch):
    monkeypatch.setenv('MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS', '0')
    monkeypatch.setenv('MIRAFRAG_CACHE_POOL_STALL_SECONDS', '0.1')
    monkeypatch.setenv('MIRAFRAG_CACHE_POOL_POLL_SECONDS', '0.01')

    failures = prefill_feature_cache(
        SlowDataset(),
        split_name='test',
        chunk_size=1,
        num_workers=2,
        show_progress=False,
        print_ready=False,
        ignore_errors=True,
    )

    assert [idx for idx, _error in failures] == [0, 1, 2]
    assert all('pool produced no results' in error for _idx, error in failures)


def test_cache_worker_timeouts_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv('MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS', raising=False)
    monkeypatch.delenv('MIRAFRAG_CACHE_POOL_STALL_SECONDS', raising=False)

    assert _cache_task_timeout_seconds() is None
    assert _cache_pool_stall_seconds(default_seconds=None) is None


def test_cache_worker_timeouts_can_be_enabled(monkeypatch):
    monkeypatch.setenv('MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS', '10')
    monkeypatch.setenv('MIRAFRAG_CACHE_POOL_STALL_SECONDS', '5')

    assert _cache_task_timeout_seconds() == 10.0
    assert _cache_pool_stall_seconds(default_seconds=None) == 5.0


def test_cache_worker_timeout_zero_disables(monkeypatch):
    monkeypatch.setenv('MIRAFRAG_CACHE_TASK_TIMEOUT_SECONDS', '0')
    monkeypatch.setenv('MIRAFRAG_CACHE_POOL_STALL_SECONDS', '0')

    assert _cache_task_timeout_seconds() is None
    assert _cache_pool_stall_seconds(default_seconds=12.0) is None
