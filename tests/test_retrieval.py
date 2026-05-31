import math

import pandas as pd
import pytest

from mirafrag.retrieval import (
    RETRIEVAL_IS_TRUE,
    RETRIEVAL_QUERY_ID,
    build_retrieval_candidate_rows,
    parse_hit_ks,
    resolve_candidate_mode,
    summarize_retrieval_hits,
)


def _query_df():
    return pd.DataFrame(
        {
            'identifier': ['q1', 'q2'],
            'smiles': ['CCO', 'CCN'],
            'inchikey': ['true1', 'true2'],
            'formula': ['C2H6O', 'C2H7N'],
            'mzs': ['18,31', '18,30'],
            'intensities': ['1,0.5', '1,0.25'],
            'precursor_mz': [47.0, 46.0],
            'adduct': ['[M+H]+', '[M+H]+'],
            'instrument_type': ['HCD', 'HCD'],
            'collision_energy': [20.0, 20.0],
        }
    )


def test_parse_hit_ks_sorts_and_deduplicates():
    assert parse_hit_ks('10,1,5,1') == (1, 5, 10)
    with pytest.raises(ValueError):
        parse_hit_ks('0')


def test_resolve_candidate_mode_uses_explicit_file_when_present():
    assert resolve_candidate_mode('/tmp/candidates.tsv', 'auto') == 'explicit'
    assert resolve_candidate_mode(None, 'auto') == 'formula'
    assert resolve_candidate_mode(None, 'mass') == 'mass'


def test_build_formula_candidate_rows_marks_true_candidate():
    queries = _query_df().iloc[:1]
    pool = pd.DataFrame(
        {
            'smiles': ['CCO', 'CO', 'CCN'],
            'inchikey': ['true1', 'decoy1', 'other'],
            'formula': ['C2H6O', 'C2H6O', 'C2H7N'],
        }
    )

    rows = build_retrieval_candidate_rows(
        queries,
        pool,
        mode='formula',
        max_candidates=8,
    )

    assert len(rows) == 2
    assert set(rows[RETRIEVAL_QUERY_ID]) == {'q1'}
    assert rows[RETRIEVAL_IS_TRUE].sum() == 1
    assert rows.loc[rows[RETRIEVAL_IS_TRUE], 'smiles'].iloc[0] == 'CCO'


def test_build_explicit_candidate_rows_uses_candidate_smiles_and_label():
    queries = _query_df().iloc[:1]
    candidates = pd.DataFrame(
        {
            'query_identifier': ['q1', 'q1'],
            'candidate_smiles': ['CO', 'CCO'],
            'candidate_inchikey': ['decoy1', 'true1'],
            'is_true': [False, True],
        }
    )

    rows = build_retrieval_candidate_rows(
        queries,
        candidates,
        mode='explicit',
        max_candidates=8,
    )

    assert rows['smiles'].tolist() == ['CO', 'CCO']
    assert rows[RETRIEVAL_IS_TRUE].tolist() == [False, True]


def test_summarize_retrieval_hits_reports_topk_and_mrr():
    rows = pd.DataFrame(
        {
            '_retrieval_query_identifier': ['q1', 'q1', 'q2', 'q2'],
            '_retrieval_candidate_identifier': ['true1', 'd1', 'd2', 'true2'],
            '_retrieval_candidate_index': [0, 1, 0, 1],
            '_retrieval_is_true': [True, False, False, True],
        }
    )
    per_query, summary = summarize_retrieval_hits(
        rows,
        [0.9, 0.1, 0.8, 0.7],
        hit_ks=(1, 2),
    )

    assert per_query['true_rank'].tolist() == [1, 2]
    assert summary['hit_at_1'] == 0.5
    assert summary['hit_at_2'] == 1.0
    assert math.isclose(summary['mrr'], 0.75)
