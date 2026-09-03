import pandas as pd
import pytest
import torch

from mirafrag.cli.retrieval_eval import (
    _collate_retrieval_batch,
    _prediction_sparse_spectra,
    _SafeRetrievalDataset,
    _target_spectrum_lookup,
)
from mirafrag.losses import sparse_fragment_only_binned_cosine_similarity
from mirafrag.sparse_spectra import sparse_cosine


def test_fast_retrieval_sparse_score_matches_decoupled_torch_score():
    pred = {
        'logits': torch.tensor([0.0, 2.0, 1.0, 1.0, -1.0]),
        'bins': torch.tensor([20, 10, 20, 10, 30]),
        'batch': torch.tensor([0, 0, 0, 1, 1]),
        'batch_size': 2,
        'num_bins': 100,
    }
    batch = {
        'target_mz': torch.tensor([0.105, 0.205, 0.105, 0.305]),
        'target_intensity': torch.tensor([1.0, 2.0, 3.0, 1.0]),
        'target_batch': torch.tensor([0, 0, 1, 1]),
        'bin_width': torch.tensor([0.01, 0.01]),
    }

    expected = sparse_fragment_only_binned_cosine_similarity(pred, batch)
    spectra = _prediction_sparse_spectra(pred, probability_mode='decoupled')
    targets = [
        _target_spectrum_lookup(
            pd.DataFrame(
                [
                    {
                        '_retrieval_query_identifier': 'q',
                        'mzs': mzs,
                        'intensities': intensities,
                    }
                ]
            ),
            mz_max=1.0,
            bin_width=0.01,
        )['q']
        for mzs, intensities in [
            ([0.105, 0.205], [1.0, 2.0]),
            ([0.105, 0.305], [3.0, 1.0]),
        ]
    ]
    observed = torch.tensor(
        [sparse_cosine(spectrum, target) for spectrum, target in zip(spectra, targets)]
    )

    assert torch.allclose(observed, expected, atol=1e-6)


def test_fast_retrieval_target_lookup_excludes_precursor_peak():
    rows = pd.DataFrame(
        [
            {
                '_retrieval_query_identifier': 'q1',
                'precursor_mz': 100.0,
                'mzs': '[50.0, 100.0]',
                'intensities': '[1.0, 10.0]',
            }
        ]
    )

    spectrum = _target_spectrum_lookup(rows, mz_max=200.0, bin_width=1.0)['q1']

    assert spectrum.bins.tolist() == [50]
    assert spectrum.values.tolist() == [1.0]


def test_safe_retrieval_dataset_returns_failed_record_on_exception():
    class BrokenDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            raise ValueError('bad molecule')

    item = _SafeRetrievalDataset(BrokenDataset())[0]

    assert item['_retrieval_failed'] is True
    assert item['_retrieval_position'] == 0
    assert 'bad molecule' in item['_retrieval_error']


def test_collate_retrieval_batch_handles_all_failed_items():
    batch = _collate_retrieval_batch(
        [
            {
                '_retrieval_failed': True,
                '_retrieval_position': 3,
                '_retrieval_error': 'bad molecule',
            }
        ]
    )

    assert batch['_retrieval_empty'] is True
    assert batch['_retrieval_positions'].tolist() == []
    assert batch['_retrieval_failed_positions'].tolist() == [3]
    assert batch['_retrieval_errors'] == ['bad molecule']


def test_safe_retrieval_dataset_materialize_feature_cache_raises_for_prefill():
    class BrokenMaterializeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {}

        def materialize_feature_cache(self, idx):
            raise ValueError('bad cache molecule')

    dataset = _SafeRetrievalDataset(BrokenMaterializeDataset())

    with pytest.raises(ValueError, match='bad cache molecule'):
        dataset.materialize_feature_cache(0)
