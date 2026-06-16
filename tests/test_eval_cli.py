import pandas as pd

from mirafrag.cli.eval import _attach_metadata, _stratified_summary


def test_stratified_eval_groups_by_instrument_and_collision_energy():
    predictions = pd.DataFrame(
        {
            'identifier': ['a', 'b', 'c', 'd'],
            'cosine': [0.1, 0.3, 0.9, 0.7],
            'sqrt_cosine': [0.2, 0.4, 0.95, 0.8],
            'candidate_coverage': [0.5, 0.6, 0.9, 0.8],
            'oos_target_mass': [0.5, 0.4, 0.1, 0.2],
            'predicted_oos_probability': [0.4, 0.3, 0.1, 0.2],
            'oracle_tolerance_cosine': [0.7, 0.8, 1.0, 0.9],
        }
    )
    metadata = pd.DataFrame(
        {
            'identifier': ['a', 'b', 'c', 'd'],
            'Instrument_type': ['HCD', 'HCD', 'QTOF', 'QTOF'],
            'CollisionEnergy': [10.0, 20.0, 30.0, 40.0],
            'PrecursorMZ': [100.0, 200.0, 300.0, 400.0],
            'adduct': ['[M+H]+'] * 4,
        }
    )

    enriched = _attach_metadata(predictions, metadata)
    summary = _stratified_summary(enriched, ce_bins=2, min_count=1)

    instrument = summary[summary['group_type'] == 'instrument']
    by_group = instrument.set_index('group')
    assert by_group.loc['HCD', 'n'] == 2
    assert by_group.loc['HCD', 'cosine_mean'] == 0.2
    assert by_group.loc['QTOF', 'n'] == 2
    assert by_group.loc['QTOF', 'cosine_mean'] == 0.8

    assert set(summary['group_type']) == {
        'instrument',
        'collision_energy_bin',
        'instrument_x_collision_energy_bin',
    }
    assert 'collision_energy_mean' in summary.columns
