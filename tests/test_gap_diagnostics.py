import pandas as pd

from mirafrag.cli.gap_diagnostics import (
    build_gap_diagnostics,
    summarize_gap_diagnostics,
)


def test_gap_diagnostics_tracks_train_overlap_and_prediction_metrics():
    train = pd.DataFrame(
        {
            'identifier': ['t1', 't2'],
            'smiles': ['CCO', 'c1ccccc1'],
            'inchikey': ['ethanol', 'benzene'],
            'instrument_type': ['Orbitrap', 'QTOF'],
            'adduct': ['[M+H]+', '[M+H]+'],
            'collision_energy': [20.0, 50.0],
            'mzs': ['[10, 20]', '[50]'],
            'intensities': ['[1, 1]', '[1]'],
            'fold': ['train', 'train'],
        }
    )
    eval_df = pd.DataFrame(
        {
            'identifier': ['v1', 'v2'],
            'smiles': ['CCO', 'CCN'],
            'inchikey': ['ethanol', 'ethylamine'],
            'instrument_type': ['Orbitrap', 'QTOF'],
            'adduct': ['[M+H]+', '[M+H]+'],
            'collision_energy': [30.0, 70.0],
            'mzs': ['[10, 20]', '[50]'],
            'intensities': ['[1, 1]', '[1]'],
            'fold': ['val', 'val'],
        }
    )
    predictions = pd.DataFrame(
        {
            'identifier': ['v1', 'v2'],
            'cosine': [0.9, 0.2],
            'candidate_coverage': [1.0, 0.5],
            'oos_target_mass': [0.0, 0.5],
        }
    )

    diagnostics = build_gap_diagnostics(
        train,
        eval_df,
        predictions=predictions,
        spectrum_neighbor_k=1,
        spectrum_ce_window=15.0,
        mz_max=100.0,
        bin_width=1.0,
        show_progress=False,
    )

    by_id = diagnostics.set_index('identifier')
    assert bool(by_id.loc['v1', 'same_smiles_in_train'])
    assert bool(by_id.loc['v1', 'same_inchikey_in_train'])
    assert by_id.loc['v1', 'same_molecule_min_ce_delta'] == 10.0
    assert not bool(by_id.loc['v2', 'same_smiles_in_train'])
    assert by_id.loc['v1', 'cosine'] == 0.9
    assert by_id.loc['v1', 'nearest_exp_spectrum_cosine'] > 0.999
    assert by_id.loc['v1', 'nearest_exp_fallback'] == 'strict'
    assert by_id.loc['v2', 'nearest_exp_spectrum_cosine'] > 0.999

    summary = summarize_gap_diagnostics(diagnostics, min_count=1)
    assert {'nearest_train_similarity', 'instrument'} <= set(summary['group_type'])
    assert 'cosine_mean' in summary.columns
    assert 'nearest_exp_spectrum_cosine_mean' in summary.columns
