import json

import pandas as pd

from mirafrag.cli.ensemble_eval import _parse_weight_grid, run_ensemble_eval
from mirafrag.sparse_spectra import sparse_from_peaks, sparse_jensen_shannon_similarity


def test_ensemble_eval_scores_weight_grid():
    eval_df = pd.DataFrame(
        {
            'identifier': ['v1'],
            'smiles': ['CCO'],
            'mzs': ['[10.5, 20.5]'],
            'intensities': ['[1, 3]'],
            'fold': ['val'],
        }
    )
    good = pd.DataFrame(
        {
            'identifier': ['v1'],
            'pred_peaks': [json.dumps({'mz': [10.5, 20.5], 'intensity': [2.0, 6.0]})],
        }
    )
    bad = pd.DataFrame(
        {
            'identifier': ['v1'],
            'pred_peaks': [json.dumps({'mz': [30.5], 'intensity': [100.0]})],
        }
    )

    rows, summary = run_ensemble_eval(
        eval_df,
        [good, bad],
        prediction_names=['good', 'bad'],
        weight_grid=[(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)],
        mz_max=50,
        bin_width=1.0,
        show_progress=False,
    )

    row = rows.iloc[0]
    assert row['good_cosine'] > 0.999
    assert row['bad_cosine'] == 0.0
    assert row['ensemble_cosine_1_0'] > 0.999
    assert row['ensemble_cosine_0_1'] == 0.0
    assert row['good_jensen_shannon_similarity'] > 0.999
    assert row['bad_jensen_shannon_similarity'] == 0.0
    assert summary.iloc[0]['cosine_mean'] > 0.999
    assert summary.iloc[0]['jensen_shannon_similarity_mean'] > 0.999


def test_parse_weight_grid_auto_for_two_models():
    grid = _parse_weight_grid('auto', n_models=2)

    assert grid[0] == (0.0, 1.0)
    assert grid[-1] == (1.0, 0.0)
    assert (0.5, 0.5) in grid


def test_sparse_jensen_shannon_similarity_scores_distribution_overlap():
    same_a = sparse_from_peaks([10.5, 20.5], [1.0, 3.0], mz_max=50, bin_width=1.0)
    same_b = sparse_from_peaks([10.5, 20.5], [2.0, 6.0], mz_max=50, bin_width=1.0)
    disjoint = sparse_from_peaks([30.5], [1.0], mz_max=50, bin_width=1.0)

    assert sparse_jensen_shannon_similarity(same_a, same_b) > 0.999
    assert sparse_jensen_shannon_similarity(same_a, disjoint) == 0.0
