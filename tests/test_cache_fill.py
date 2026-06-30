import pytest

from mirafrag.cache_fill import prefill_feature_cache


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
