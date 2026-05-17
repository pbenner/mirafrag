from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _fragment_graph_edges(
    ordered: list[dict[str, Any]],
    *,
    max_edges: int,
    mz_max: float,
    max_h_transfers: int,
    max_isotope_peaks: int,
) -> tuple[list[tuple[int, int]], list[list[float]]]:
    if max_edges <= 0 or len(ordered) <= 1:
        return [], []

    edges: list[tuple[int, int]] = []
    features: list[list[float]] = []
    seen: set[tuple[int, int, str]] = set()
    indices_by_hash: dict[str, list[int]] = {}
    indices_by_formula: dict[tuple[int, int], list[int]] = {}
    indices_by_atomset: dict[int, list[int]] = {}
    indices_by_bin: dict[int, list[int]] = {}
    for idx, item in enumerate(ordered):
        mask = int(item['mask'])
        indices_by_hash.setdefault(str(item['fragment_hash']), []).append(idx)
        indices_by_atomset.setdefault(mask, []).append(idx)
        indices_by_formula.setdefault((mask, int(item['h_shift'])), []).append(idx)
        indices_by_bin.setdefault(int(item['bin']), []).append(idx)

    def add_edge(src: int, dst: int, relation: str) -> bool:
        if src == dst or len(edges) >= max_edges:
            return False
        key = (int(src), int(dst), relation)
        if key in seen:
            return True
        seen.add(key)
        edges.append((int(src), int(dst)))
        features.append(
            _fragment_edge_features(
                ordered[src],
                ordered[dst],
                relation=relation,
                mz_max=mz_max,
                max_h_transfers=max_h_transfers,
                max_isotope_peaks=max_isotope_peaks,
            )
        )
        return len(edges) < max_edges

    for child_idx, child in enumerate(ordered):
        for parent_hash in child.get('parent_hashes', []):
            for parent_idx in indices_by_hash.get(str(parent_hash), [])[:8]:
                if int(ordered[parent_idx]['h_shift']) != int(child['h_shift']):
                    continue
                if not add_edge(parent_idx, child_idx, 'parent_to_child'):
                    return edges, features
                if not add_edge(child_idx, parent_idx, 'child_to_parent'):
                    return edges, features

    for group in indices_by_formula.values():
        sorted_group = sorted(
            group,
            key=lambda idx: (
                int(ordered[idx]['isotope_rank']),
                float(ordered[idx]['mz']),
            ),
        )
        if not _connect_neighbors(sorted_group, 'same_formula', add_edge):
            return edges, features

    for group in indices_by_atomset.values():
        sorted_group = sorted(
            group,
            key=lambda idx: (
                int(ordered[idx]['h_shift']),
                int(ordered[idx]['isotope_rank']),
                float(ordered[idx]['mz']),
            ),
        )
        if not _connect_neighbors(sorted_group, 'same_atomset', add_edge):
            return edges, features

    for group in indices_by_bin.values():
        sorted_group = sorted(
            group,
            key=lambda idx: (
                float(ordered[idx]['score']),
                int(ordered[idx]['cut_count']),
                float(ordered[idx]['mz']),
            ),
        )
        if not _connect_neighbors(sorted_group[:16], 'same_bin', add_edge):
            break

    return edges, features


def _connect_neighbors(
    indices: list[int],
    relation: str,
    add_edge: Callable[[int, int, str], bool],
) -> bool:
    if len(indices) <= 1:
        return True
    for src, dst in zip(indices[:-1], indices[1:]):
        if not add_edge(src, dst, relation):
            return False
        if not add_edge(dst, src, relation):
            return False
    return True


def _fragment_edge_features(
    src: dict[str, Any],
    dst: dict[str, Any],
    *,
    relation: str,
    mz_max: float,
    max_h_transfers: int,
    max_isotope_peaks: int,
) -> list[float]:
    mass_delta = (float(dst['mz']) - float(src['mz'])) / max(float(mz_max), 1.0)
    h_delta = (int(dst['h_shift']) - int(src['h_shift'])) / max(
        float(max_h_transfers),
        1.0,
    )
    isotope_delta = (int(dst['isotope_rank']) - int(src['isotope_rank'])) / max(
        float(max_isotope_peaks - 1),
        1.0,
    )
    return [
        1.0 if relation == 'parent_to_child' else 0.0,
        1.0 if relation == 'child_to_parent' else 0.0,
        1.0 if relation == 'same_formula' else 0.0,
        1.0 if relation == 'same_atomset' else 0.0,
        1.0 if relation == 'same_bin' else 0.0,
        float(mass_delta),
        float(h_delta),
        float(isotope_delta),
    ]
