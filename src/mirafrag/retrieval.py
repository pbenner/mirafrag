from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from mirafrag.data import SMILES_ALIASES, find_column

IDENTIFIER_ALIASES = ('identifier', 'query_identifier', 'query_id', 'spectrum_id')
INCHIKEY_ALIASES = ('inchikey', 'InChIKey', 'INCHIKEY')
FORMULA_ALIASES = ('formula', 'molecular_formula', 'Formula')
MASS_ALIASES = ('parent_mass', 'exact_mass', 'monoisotopic_mass')
QUERY_IDENTIFIER_ALIASES = ('query_identifier', 'query_id', 'spectrum_identifier')
QUERY_SMILES_ALIASES = (
    'query_smiles',
    'query_SMILES',
    'query_molecule_smiles',
)
TRUE_ALIASES = ('is_true', 'is_correct', 'label')

RETRIEVAL_QUERY_ID = '_retrieval_query_identifier'
RETRIEVAL_CANDIDATE_ID = '_retrieval_candidate_identifier'
RETRIEVAL_CANDIDATE_INDEX = '_retrieval_candidate_index'
RETRIEVAL_IS_TRUE = '_retrieval_is_true'
RETRIEVAL_SCORE = 'score'


def read_retrieval_candidate_table(path: str | Path) -> pd.DataFrame:
    """
    Read an explicit retrieval candidate table.

    MassSpecGym distributes official retrieval candidates as JSON mappings from
    query SMILES to candidate SMILES lists. Tabular TSV/CSV/JSONL inputs are
    still accepted through the project table reader.
    """
    path = Path(path)
    if path.suffix.lower() == '.json':
        with path.open() as handle:
            return retrieval_candidate_table_from_json(json.load(handle))

    from mirafrag.data import read_table

    return read_table(path)


def retrieval_candidate_table_from_json(payload: Any) -> pd.DataFrame:
    """
    Convert MassSpecGym-style retrieval candidate JSON into an explicit table.

    Supported layouts include ``{query_smiles: [candidate_smiles, ...]}``,
    ``{query_smiles: [{'smiles': ...}, ...]}``, and record lists containing
    query/candidate columns. The returned dataframe has ``query_smiles`` and
    ``candidate_smiles`` columns so it can be consumed by explicit retrieval.
    """
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for query_smiles, candidates in payload.items():
            rows.extend(_json_candidate_rows(str(query_smiles), candidates))
    elif isinstance(payload, list):
        for entry in payload:
            rows.extend(_json_record_rows(entry))
    else:
        raise ValueError('Retrieval candidate JSON must contain an object or a list.')
    if not rows:
        raise ValueError('Retrieval candidate JSON did not contain any candidates.')
    return pd.DataFrame(rows)


def parse_hit_ks(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """
    Parse comma-separated hit-rate cutoffs into a sorted tuple of positive ints.
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',')]
        hit_ks = [int(part) for part in parts if part]
    else:
        hit_ks = [int(part) for part in value]
    if not hit_ks or any(k <= 0 for k in hit_ks):
        raise ValueError('hit_ks must contain at least one positive integer.')
    return tuple(sorted(set(hit_ks)))


def build_retrieval_candidate_rows(
    queries: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    mode: str,
    max_candidates: int = 256,
    seed: int = 0,
    mass_tolerance: float = 0.01,
    ensure_true_candidate: bool = True,
) -> pd.DataFrame:
    """
    Expand query spectra into candidate rows for retrieval-style evaluation.

    Each returned row keeps the query spectrum and metadata, replaces the SMILES
    with one candidate molecule, and adds private retrieval columns used for
    hit-rate aggregation. ``mode='explicit'`` expects one row per candidate with
    a query identifier column; ``mode='formula'`` and ``mode='mass'`` build a
    diagnostic candidate pool from the provided dataframe.
    """
    if mode not in {'explicit', 'formula', 'mass'}:
        raise ValueError("mode must be one of: 'explicit', 'formula', 'mass'.")
    max_candidates = int(max_candidates)
    if max_candidates <= 0:
        raise ValueError('max_candidates must be positive.')
    if queries.empty:
        return pd.DataFrame()
    if mode == 'explicit':
        return _build_explicit_candidate_rows(
            queries,
            candidate_pool,
            max_candidates=max_candidates,
            seed=seed,
            ensure_true_candidate=ensure_true_candidate,
        )
    return _build_pool_candidate_rows(
        queries,
        candidate_pool,
        mode=mode,
        max_candidates=max_candidates,
        seed=seed,
        mass_tolerance=mass_tolerance,
        ensure_true_candidate=ensure_true_candidate,
    )


def estimate_retrieval_candidate_count(
    queries: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    mode: str,
    max_candidates: int = 256,
    ensure_true_candidate: bool = True,
) -> int | None:
    """
    Estimate retrieval candidate rows without constructing the expanded table.

    The estimate is exact for explicit candidate inputs before downstream element
    filtering. Diagnostic formula and mass modes return ``None`` because their
    candidate sets depend on dataframe-wide grouping or numeric tolerances and
    are cheap enough to monitor by processed count only.
    """
    if mode != 'explicit' or queries.empty:
        return None
    max_candidates = int(max_candidates)
    if max_candidates <= 0:
        raise ValueError('max_candidates must be positive.')

    query_id_col = find_column(queries, IDENTIFIER_ALIASES, required=False)
    query_smiles_col = find_column(queries, SMILES_ALIASES)
    query_identity_col = _identity_column(queries)
    candidate_query_id_col = find_column(
        candidate_pool,
        QUERY_IDENTIFIER_ALIASES,
        required=False,
    )
    candidate_query_smiles_col = find_column(
        candidate_pool,
        QUERY_SMILES_ALIASES,
        required=False,
    )
    if candidate_query_id_col is None and candidate_query_smiles_col is None:
        return None
    candidate_smiles_col = find_column(
        candidate_pool,
        ('candidate_smiles', 'smiles', 'SMILES', 'Smiles'),
    )
    candidate_identity_col = find_column(
        candidate_pool,
        ('candidate_inchikey', 'inchikey', 'candidate_identifier', 'identifier'),
        required=False,
    )
    explicit_true_col = find_column(candidate_pool, TRUE_ALIASES, required=False)

    if candidate_query_smiles_col is not None:
        group_col = candidate_query_smiles_col

        def query_group_key(query: pd.Series, _query_position: int) -> str:
            return str(query.get(query_smiles_col))

    else:
        group_col = candidate_query_id_col

        def query_group_key(query: pd.Series, query_position: int) -> str:
            return _query_identifier(query, query_position, query_id_col)

    groups = {
        str(key): group for key, group in candidate_pool.groupby(group_col, sort=False)
    }
    total = 0
    for query_position, (_idx, query) in enumerate(queries.iterrows()):
        candidates = groups.get(
            query_group_key(query, query_position),
            candidate_pool.iloc[0:0],
        )
        count = int(len(candidates))
        if ensure_true_candidate and count > 0:
            query_identity = str(
                query.get(query_identity_col, query.get(query_smiles_col))
            )
            is_true = _candidate_truth_mask(
                candidates,
                query_identity=query_identity,
                query_smiles=str(query.get(query_smiles_col)),
                candidate_smiles_col=candidate_smiles_col,
                candidate_identity_col=candidate_identity_col,
                explicit_true_col=explicit_true_col,
            )
            if not bool(is_true.any()):
                count += 1
        elif ensure_true_candidate:
            count = 1
        total += min(count, max_candidates)
    return total


def summarize_retrieval_hits(
    candidate_rows: pd.DataFrame,
    scores: list[float],
    *,
    hit_ks: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Rank candidates per query and compute hit-rate summary statistics.
    """
    if len(candidate_rows) != len(scores):
        raise ValueError('candidate_rows and scores must have the same length.')
    hit_ks = parse_hit_ks(hit_ks)
    if candidate_rows.empty:
        summary = {'n_queries': 0.0, 'candidate_count_mean': float('nan'), 'mrr': 0.0}
        summary.update({f'hit_at_{k}': 0.0 for k in hit_ks})
        return pd.DataFrame(), summary

    scored = candidate_rows[
        [
            RETRIEVAL_QUERY_ID,
            RETRIEVAL_CANDIDATE_ID,
            RETRIEVAL_CANDIDATE_INDEX,
            RETRIEVAL_IS_TRUE,
        ]
    ].copy()
    scored[RETRIEVAL_SCORE] = [float(score) for score in scores]

    rows: list[dict[str, Any]] = []
    for query_id, group in scored.groupby(RETRIEVAL_QUERY_ID, sort=False):
        ranked = group.sort_values(
            [RETRIEVAL_SCORE, RETRIEVAL_CANDIDATE_INDEX],
            ascending=[False, True],
            kind='mergesort',
        ).reset_index(drop=True)
        true_positions = ranked.index[ranked[RETRIEVAL_IS_TRUE].astype(bool)].tolist()
        true_rank = int(true_positions[0] + 1) if true_positions else None
        best = ranked.iloc[0]
        row: dict[str, Any] = {
            'query_identifier': query_id,
            'num_candidates': int(len(ranked)),
            'true_rank': true_rank,
            'mrr': (1.0 / true_rank) if true_rank is not None else 0.0,
            'best_candidate_identifier': best[RETRIEVAL_CANDIDATE_ID],
            'best_score': float(best[RETRIEVAL_SCORE]),
            'true_score': float(ranked.iloc[true_positions[0]][RETRIEVAL_SCORE])
            if true_positions
            else float('nan'),
        }
        for k in hit_ks:
            row[f'hit@{k}'] = bool(true_rank is not None and true_rank <= k)
        rows.append(row)

    result = pd.DataFrame(rows)
    summary = {
        'n_queries': float(len(result)),
        'candidate_count_mean': float(result['num_candidates'].mean())
        if not result.empty
        else float('nan'),
        'mrr': float(result['mrr'].mean()) if not result.empty else 0.0,
    }
    for k in hit_ks:
        summary[f'hit_at_{k}'] = (
            float(result[f'hit@{k}'].mean()) if not result.empty else 0.0
        )
    return result, summary


def resolve_candidate_mode(candidate_input: str | None, mode: str) -> str:
    """
    Resolve ``auto`` candidate mode from whether an explicit candidate file exists.
    """
    if mode != 'auto':
        return mode
    return 'explicit' if candidate_input else 'formula'


def _build_pool_candidate_rows(
    queries: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    mode: str,
    max_candidates: int,
    seed: int,
    mass_tolerance: float,
    ensure_true_candidate: bool,
) -> pd.DataFrame:
    query_id_col = find_column(queries, IDENTIFIER_ALIASES, required=False)
    query_smiles_col = find_column(queries, SMILES_ALIASES)
    pool_smiles_col = find_column(candidate_pool, SMILES_ALIASES)
    query_identity_col = _identity_column(queries)
    pool_identity_col = _identity_column(candidate_pool)
    pool = _deduplicate_pool(candidate_pool, identity_col=pool_identity_col)
    rows: list[dict[str, Any]] = []

    if mode == 'formula':
        query_group_col = _formula_column(queries)
        pool_group_col = _formula_column(pool)
        groups = dict(tuple(pool.groupby(pool_group_col, dropna=False, sort=False)))
        for query_position, (_idx, query) in enumerate(queries.iterrows()):
            key = query.get(query_group_col)
            candidates = groups.get(key, pool.iloc[0:0])
            rows.extend(
                _candidate_rows_for_query(
                    query,
                    candidates,
                    query_position=query_position,
                    query_id_col=query_id_col,
                    query_smiles_col=query_smiles_col,
                    query_identity_col=query_identity_col,
                    candidate_smiles_col=pool_smiles_col,
                    candidate_identity_col=pool_identity_col,
                    max_candidates=max_candidates,
                    seed=seed,
                    ensure_true_candidate=ensure_true_candidate,
                )
            )
    else:
        query_mass_col = _mass_column(queries)
        pool_mass_col = _mass_column(pool)
        pool_masses = pd.to_numeric(pool[pool_mass_col], errors='coerce')
        for query_position, (_idx, query) in enumerate(queries.iterrows()):
            query_mass = _safe_float(query.get(query_mass_col))
            if math.isnan(query_mass):
                candidates = pool.iloc[0:0]
            else:
                candidates = pool[(pool_masses - query_mass).abs() <= mass_tolerance]
            rows.extend(
                _candidate_rows_for_query(
                    query,
                    candidates,
                    query_position=query_position,
                    query_id_col=query_id_col,
                    query_smiles_col=query_smiles_col,
                    query_identity_col=query_identity_col,
                    candidate_smiles_col=pool_smiles_col,
                    candidate_identity_col=pool_identity_col,
                    max_candidates=max_candidates,
                    seed=seed,
                    ensure_true_candidate=ensure_true_candidate,
                )
            )
    return pd.DataFrame(rows)


def _build_explicit_candidate_rows(
    queries: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    max_candidates: int,
    seed: int,
    ensure_true_candidate: bool,
) -> pd.DataFrame:
    query_id_col = find_column(queries, IDENTIFIER_ALIASES, required=False)
    query_smiles_col = find_column(queries, SMILES_ALIASES)
    query_identity_col = _identity_column(queries)
    candidate_query_id_col = find_column(
        candidate_pool,
        QUERY_IDENTIFIER_ALIASES,
        required=False,
    )
    candidate_query_smiles_col = find_column(
        candidate_pool,
        QUERY_SMILES_ALIASES,
        required=False,
    )
    if candidate_query_id_col is None and candidate_query_smiles_col is None:
        raise ValueError(
            'Explicit retrieval candidates need either a query identifier column '
            f'{QUERY_IDENTIFIER_ALIASES} or a query SMILES column '
            f'{QUERY_SMILES_ALIASES}.'
        )
    candidate_smiles_col = find_column(
        candidate_pool,
        ('candidate_smiles', 'smiles', 'SMILES', 'Smiles'),
    )
    candidate_identity_col = find_column(
        candidate_pool,
        ('candidate_inchikey', 'inchikey', 'candidate_identifier', 'identifier'),
        required=False,
    )
    explicit_true_col = find_column(candidate_pool, TRUE_ALIASES, required=False)
    if candidate_query_smiles_col is not None:
        group_col = candidate_query_smiles_col

        def query_group_key(query: pd.Series, _query_position: int) -> str:
            return str(query.get(query_smiles_col))

    else:
        group_col = candidate_query_id_col

        def query_group_key(query: pd.Series, query_position: int) -> str:
            return _query_identifier(query, query_position, query_id_col)

    groups = {
        str(key): group for key, group in candidate_pool.groupby(group_col, sort=False)
    }
    rows: list[dict[str, Any]] = []
    for query_position, (_idx, query) in enumerate(queries.iterrows()):
        candidates = groups.get(
            query_group_key(query, query_position),
            candidate_pool.iloc[0:0],
        )
        rows.extend(
            _candidate_rows_for_query(
                query,
                candidates,
                query_position=query_position,
                query_id_col=query_id_col,
                query_smiles_col=query_smiles_col,
                query_identity_col=query_identity_col,
                candidate_smiles_col=candidate_smiles_col,
                candidate_identity_col=candidate_identity_col,
                max_candidates=max_candidates,
                seed=seed,
                ensure_true_candidate=ensure_true_candidate,
                explicit_true_col=explicit_true_col,
            )
        )
    return pd.DataFrame(rows)


def _json_candidate_rows(query_smiles: str, candidates: Any) -> list[dict[str, Any]]:
    if isinstance(candidates, dict):
        candidates = _first_present(
            candidates,
            ('candidates', 'candidate_smiles', 'smiles', 'molecules'),
        )
    if isinstance(candidates, str):
        candidates = [candidates]
    if not isinstance(candidates, list):
        raise ValueError(
            'Each retrieval candidate JSON entry must map to a candidate list.'
        )
    rows = []
    for candidate in candidates:
        rows.append(_json_candidate_row(query_smiles, candidate))
    return rows


def _json_record_rows(entry: Any) -> list[dict[str, Any]]:
    if not isinstance(entry, dict):
        raise ValueError('Retrieval candidate JSON list entries must be objects.')
    query_smiles = _first_present(entry, QUERY_SMILES_ALIASES)
    candidates = _first_present(
        entry,
        ('candidates', 'candidate_smiles', 'smiles', 'molecules'),
    )
    if query_smiles is not None and isinstance(candidates, list):
        return _json_candidate_rows(str(query_smiles), candidates)
    candidate_smiles = _candidate_smiles_from_json(entry)
    if query_smiles is None or candidate_smiles is None:
        raise ValueError(
            'Retrieval candidate JSON records need query_smiles and candidate_smiles.'
        )
    row = {
        'query_smiles': str(query_smiles),
        'candidate_smiles': str(candidate_smiles),
    }
    if 'is_true' in entry:
        row['is_true'] = entry['is_true']
    return [row]


def _json_candidate_row(query_smiles: str, candidate: Any) -> dict[str, Any]:
    candidate_smiles = _candidate_smiles_from_json(candidate)
    if candidate_smiles is None:
        raise ValueError('Candidate JSON records need a SMILES value.')
    row = {'query_smiles': query_smiles, 'candidate_smiles': str(candidate_smiles)}
    if isinstance(candidate, dict):
        candidate_id = _first_present(
            candidate,
            ('candidate_identifier', 'identifier', 'inchikey', 'InChIKey'),
        )
        if candidate_id is not None:
            row['candidate_identifier'] = candidate_id
        if 'is_true' in candidate:
            row['is_true'] = candidate['is_true']
    return row


def _candidate_smiles_from_json(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _first_present(
            value,
            ('candidate_smiles', 'smiles', 'SMILES', 'Smiles'),
        )
    return None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _candidate_rows_for_query(
    query: pd.Series,
    candidates: pd.DataFrame,
    *,
    query_position: int,
    query_id_col: str | None,
    query_smiles_col: str,
    query_identity_col: str,
    candidate_smiles_col: str,
    candidate_identity_col: str | None,
    max_candidates: int,
    seed: int,
    ensure_true_candidate: bool,
    explicit_true_col: str | None = None,
) -> list[dict[str, Any]]:
    query_id = _query_identifier(query, query_position, query_id_col)
    query_identity = str(query.get(query_identity_col, query.get(query_smiles_col)))
    candidates = candidates.copy()
    if candidates.empty and ensure_true_candidate:
        candidates = pd.DataFrame([query])
        candidate_smiles_col = query_smiles_col
        candidate_identity_col = query_identity_col
    if candidates.empty:
        return []

    is_true = _candidate_truth_mask(
        candidates,
        query_identity=query_identity,
        query_smiles=str(query.get(query_smiles_col)),
        candidate_smiles_col=candidate_smiles_col,
        candidate_identity_col=candidate_identity_col,
        explicit_true_col=explicit_true_col,
    )
    if ensure_true_candidate and not bool(is_true.any()):
        true_candidate = query.copy()
        true_candidate[candidate_smiles_col] = query.get(query_smiles_col)
        if candidate_identity_col is not None:
            true_candidate[candidate_identity_col] = query_identity
        candidates = pd.concat(
            [candidates, pd.DataFrame([true_candidate])], ignore_index=True
        )
        is_true = pd.concat(
            [is_true, pd.Series([True], index=[len(is_true)])],
            ignore_index=True,
        )
    candidates = candidates.copy()
    candidates['_retrieval_build_is_true'] = is_true.to_numpy(dtype=bool)

    selected = _sample_candidates(
        candidates,
        candidates['_retrieval_build_is_true'],
        max_candidates=max_candidates,
        seed=seed,
        query_id=query_id,
    )
    rows = []
    for candidate_index, (_candidate_idx, candidate) in enumerate(selected.iterrows()):
        row = query.to_dict()
        candidate_smiles = str(
            candidate.get(candidate_smiles_col, query.get(query_smiles_col))
        )
        row[query_smiles_col] = candidate_smiles
        for source_col, target_col in _molecule_copy_columns(candidate, row):
            row[target_col] = candidate[source_col]
        candidate_identifier = _candidate_identifier(
            candidate,
            candidate_identity_col=candidate_identity_col,
            candidate_smiles_col=candidate_smiles_col,
        )
        row[RETRIEVAL_QUERY_ID] = query_id
        row[RETRIEVAL_CANDIDATE_ID] = candidate_identifier
        row[RETRIEVAL_CANDIDATE_INDEX] = candidate_index
        row[RETRIEVAL_IS_TRUE] = bool(candidate.get('_retrieval_build_is_true', False))
        rows.append(row)
    return rows


def _sample_candidates(
    candidates: pd.DataFrame,
    is_true: pd.Series,
    *,
    max_candidates: int,
    seed: int,
    query_id: str,
) -> pd.DataFrame:
    if len(candidates) <= max_candidates:
        return candidates.reset_index(drop=True)
    true_rows = candidates[is_true.to_numpy()].copy()
    false_rows = candidates[~is_true.to_numpy()].copy()
    if len(true_rows) >= max_candidates:
        return true_rows.head(max_candidates).reset_index(drop=True)
    remaining = max_candidates - len(true_rows)
    false_rows = false_rows.sample(
        n=min(remaining, len(false_rows)),
        random_state=_stable_random_state(seed, query_id),
    )
    return pd.concat([true_rows, false_rows], ignore_index=True)


def _candidate_truth_mask(
    candidates: pd.DataFrame,
    *,
    query_identity: str,
    query_smiles: str,
    candidate_smiles_col: str,
    candidate_identity_col: str | None,
    explicit_true_col: str | None,
) -> pd.Series:
    if explicit_true_col is not None and explicit_true_col in candidates:
        return candidates[explicit_true_col].map(_truthy).astype(bool)
    if candidate_identity_col is not None and candidate_identity_col in candidates:
        return candidates[candidate_identity_col].astype(str) == query_identity
    return candidates[candidate_smiles_col].astype(str) == query_smiles


def _deduplicate_pool(pool: pd.DataFrame, *, identity_col: str) -> pd.DataFrame:
    return pool.drop_duplicates(identity_col).reset_index(drop=True)


def _identity_column(df: pd.DataFrame) -> str:
    return find_column(df, INCHIKEY_ALIASES, required=False) or find_column(
        df,
        SMILES_ALIASES,
    )


def _formula_column(df: pd.DataFrame) -> str:
    return find_column(df, FORMULA_ALIASES)


def _mass_column(df: pd.DataFrame) -> str:
    return find_column(df, MASS_ALIASES)


def _query_identifier(
    query: pd.Series,
    query_position: int,
    query_id_col: str | None,
) -> str:
    if query_id_col is not None:
        return str(query.get(query_id_col))
    return str(query_position)


def _candidate_identifier(
    candidate: pd.Series,
    *,
    candidate_identity_col: str | None,
    candidate_smiles_col: str,
) -> str:
    if candidate_identity_col is not None and candidate_identity_col in candidate:
        value = candidate.get(candidate_identity_col)
        if pd.notna(value):
            return str(value)
    value = candidate.get(candidate_smiles_col)
    return str(value) if pd.notna(value) else ''


def _molecule_copy_columns(
    candidate: pd.Series,
    row: dict[str, Any],
) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for aliases in (INCHIKEY_ALIASES, FORMULA_ALIASES):
        source = next((col for col in aliases if col in candidate), None)
        target = next((col for col in aliases if col in row), None)
        if source is not None and target is not None:
            columns.append((source, target))
    return columns


def _stable_random_state(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f'{seed}:{query_id}'.encode()).hexdigest()
    return int(digest[:8], 16)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}
