from __future__ import annotations

from typing import Any

import torch

from mirafrag.fragments.constants import (
    BOND_BREAK_FEATURE_DIM,
    FRAGMENT_EDGE_FEATURE_DIM,
    FRAGMENT_FEATURE_DIM,
    FRAGMENT_FORMULA_DIM,
)


def collate_fragment_candidates(
    fragments: list[dict[str, Any]],
    *,
    node_offsets: list[int],
) -> dict[str, torch.Tensor]:
    """
    Batch variable-size fragment candidate dictionaries.

    Formula atom indices are offset by graph node offsets, peak formula indices are remapped to global formula rows, and fragment-graph edges are converted to one concatenated edge index.
    """
    atom_indices: list[int] = []
    atom_ptr = [0]
    formula_batch: list[int] = []
    fragment_batch: list[int] = []
    fragment_mzs: list[float] = []
    fragment_bins: list[int] = []
    fragment_log_priors: list[float] = []
    fragment_formula_indices: list[int] = []
    fragment_features: list[list[float]] = []
    fragment_formula_counts: list[list[float]] = []
    formula_node_indices: list[int] = []
    formula_h_shifts: list[int] = []
    node_atom_indices: list[int] = []
    node_atom_ptr = [0]
    node_batch: list[int] = []
    node_features: list[list[float]] = []
    node_formula_counts: list[list[float]] = []
    node_edge_sources: list[int] = []
    node_edge_targets: list[int] = []
    node_edge_features: list[list[float]] = []
    has_fragment_nodes = True
    bond_atom_indices: list[list[int]] = []
    bond_ptr = [0]
    bond_features: list[list[float]] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_features: list[list[float]] = []

    for graph_idx, (fragment_set, node_offset) in enumerate(
        zip(fragments, node_offsets)
    ):
        local_atom_indices = fragment_set['atom_indices']
        mzs = fragment_set['mzs']
        bins = fragment_set['bins']
        log_priors = fragment_set['log_priors']
        formula_indices = fragment_set['formula_index']
        features = fragment_set['features']
        formula_counts = fragment_set.get('formula_counts')
        local_formula_node_indices = fragment_set.get('formula_node_index')
        local_formula_h_shifts = fragment_set.get('formula_h_shift')
        local_node_atom_indices = fragment_set.get('node_atom_indices')
        local_node_features = fragment_set.get('node_features')
        local_node_formula_counts = fragment_set.get('node_formula_counts')
        local_node_edges = fragment_set.get('node_edge_index')
        local_node_edge_features = fragment_set.get('node_edge_features')
        local_bond_atom_indices = fragment_set.get('bond_atom_indices')
        local_bond_features = fragment_set.get('bond_features')
        if (
            local_formula_node_indices is None
            or local_formula_h_shifts is None
            or local_node_atom_indices is None
            or local_node_features is None
            or local_node_formula_counts is None
        ):
            has_fragment_nodes = False
        local_node_to_global: dict[int, int] = {}
        if (
            local_node_atom_indices is not None
            and local_node_features is not None
            and local_node_formula_counts is not None
        ):
            for local_node_idx, (local_indices, feature, counts) in enumerate(
                zip(
                    local_node_atom_indices,
                    local_node_features,
                    local_node_formula_counts,
                )
            ):
                if not local_indices:
                    continue
                global_node_idx = len(node_batch)
                local_node_to_global[int(local_node_idx)] = global_node_idx
                node_atom_indices.extend(
                    int(idx) + int(node_offset) for idx in local_indices
                )
                node_atom_ptr.append(len(node_atom_indices))
                node_batch.append(graph_idx)
                node_features.append([float(value) for value in feature])
                node_formula_counts.append([float(value) for value in counts])
        local_to_global: dict[int, int] = {}
        for local_idx, (local_indices, feature) in enumerate(
            zip(local_atom_indices, features)
        ):
            if not local_indices:
                continue
            global_fragment_idx = len(formula_batch)
            local_to_global[int(local_idx)] = global_fragment_idx
            atom_indices.extend(int(idx) + int(node_offset) for idx in local_indices)
            atom_ptr.append(len(atom_indices))
            formula_batch.append(graph_idx)
            if local_formula_node_indices is None:
                local_node_idx = None
            else:
                local_node_idx = int(local_formula_node_indices[local_idx])
            global_node_idx = (
                None
                if local_node_idx is None
                else local_node_to_global.get(local_node_idx)
            )
            if global_node_idx is None:
                global_node_idx = len(node_batch)
                local_node_to_global.setdefault(int(local_idx), global_node_idx)
                node_atom_indices.extend(
                    int(idx) + int(node_offset) for idx in local_indices
                )
                node_atom_ptr.append(len(node_atom_indices))
                node_batch.append(graph_idx)
                node_features.append([float(value) for value in feature])
                if formula_counts is None:
                    node_formula_counts.append([0.0] * FRAGMENT_FORMULA_DIM)
                else:
                    node_formula_counts.append(
                        [float(value) for value in formula_counts[local_idx]]
                    )
            formula_node_indices.append(global_node_idx)
            if local_formula_h_shifts is None:
                formula_h_shifts.append(0)
            else:
                formula_h_shifts.append(int(local_formula_h_shifts[local_idx]))
            fragment_features.append([float(value) for value in feature])
            if formula_counts is None:
                fragment_formula_counts.append([0.0] * FRAGMENT_FORMULA_DIM)
            else:
                fragment_formula_counts.append(
                    [float(value) for value in formula_counts[local_idx]]
                )
            formula_bond_atoms = (
                []
                if local_bond_atom_indices is None
                else local_bond_atom_indices[local_idx]
            )
            formula_bond_features = (
                [] if local_bond_features is None else local_bond_features[local_idx]
            )
            if len(formula_bond_atoms) != len(formula_bond_features):
                raise ValueError(
                    'Fragment bond-break atom/features length mismatch: '
                    f'atoms={len(formula_bond_atoms)} '
                    f'features={len(formula_bond_features)}'
                )
            for pair, bond_feature in zip(formula_bond_atoms, formula_bond_features):
                inside, outside = pair
                bond_atom_indices.append(
                    [int(inside) + int(node_offset), int(outside) + int(node_offset)]
                )
                bond_features.append([float(value) for value in bond_feature])
            bond_ptr.append(len(bond_atom_indices))
        for mz, bin_idx, log_prior, formula_idx in zip(
            mzs,
            bins,
            log_priors,
            formula_indices,
        ):
            global_formula_idx = local_to_global.get(int(formula_idx))
            if global_formula_idx is None:
                continue
            fragment_batch.append(graph_idx)
            fragment_mzs.append(float(mz))
            fragment_bins.append(int(bin_idx))
            fragment_log_priors.append(float(log_prior))
            fragment_formula_indices.append(global_formula_idx)
        if local_node_edges is not None and local_node_edge_features is not None:
            for edge, feature in zip(local_node_edges, local_node_edge_features):
                src = local_node_to_global.get(int(edge[0]))
                dst = local_node_to_global.get(int(edge[1]))
                if src is None or dst is None:
                    continue
                node_edge_sources.append(src)
                node_edge_targets.append(dst)
                node_edge_features.append([float(value) for value in feature])
        for edge, feature in zip(
            fragment_set['edge_index'],
            fragment_set['edge_features'],
        ):
            src = local_to_global.get(int(edge[0]))
            dst = local_to_global.get(int(edge[1]))
            if src is None or dst is None:
                continue
            edge_sources.append(src)
            edge_targets.append(dst)
            edge_features.append([float(value) for value in feature])

    if fragment_features:
        features_tensor = torch.tensor(
            fragment_features, dtype=torch.get_default_dtype()
        )
        formula_counts_tensor = torch.tensor(
            fragment_formula_counts, dtype=torch.get_default_dtype()
        )
    else:
        features_tensor = torch.empty(
            0,
            FRAGMENT_FEATURE_DIM,
            dtype=torch.get_default_dtype(),
        )
        formula_counts_tensor = torch.empty(
            0,
            FRAGMENT_FORMULA_DIM,
            dtype=torch.get_default_dtype(),
        )

    if node_features:
        node_features_tensor = torch.tensor(
            node_features, dtype=torch.get_default_dtype()
        )
        node_formula_counts_tensor = torch.tensor(
            node_formula_counts, dtype=torch.get_default_dtype()
        )
    else:
        node_features_tensor = torch.empty(
            0,
            FRAGMENT_FEATURE_DIM,
            dtype=torch.get_default_dtype(),
        )
        node_formula_counts_tensor = torch.empty(
            0,
            FRAGMENT_FORMULA_DIM,
            dtype=torch.get_default_dtype(),
        )

    if node_edge_sources:
        node_edge_index = torch.tensor(
            [node_edge_sources, node_edge_targets], dtype=torch.long
        )
        node_edge_attr = torch.tensor(
            node_edge_features, dtype=torch.get_default_dtype()
        )
    else:
        node_edge_index = torch.empty(2, 0, dtype=torch.long)
        node_edge_attr = torch.empty(
            0,
            FRAGMENT_EDGE_FEATURE_DIM,
            dtype=torch.get_default_dtype(),
        )

    if bond_atom_indices:
        bond_atom_index = torch.tensor(bond_atom_indices, dtype=torch.long)
        bond_features_tensor = torch.tensor(
            bond_features, dtype=torch.get_default_dtype()
        )
    else:
        bond_atom_index = torch.empty(0, 2, dtype=torch.long)
        bond_features_tensor = torch.empty(
            0,
            BOND_BREAK_FEATURE_DIM,
            dtype=torch.get_default_dtype(),
        )

    if edge_sources:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.get_default_dtype())
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(
            0,
            FRAGMENT_EDGE_FEATURE_DIM,
            dtype=torch.get_default_dtype(),
        )

    return {
        'atom_index': torch.tensor(atom_indices, dtype=torch.long),
        'atom_ptr': torch.tensor(atom_ptr, dtype=torch.long),
        'formula_batch': torch.tensor(formula_batch, dtype=torch.long),
        'has_fragment_nodes': torch.tensor(bool(has_fragment_nodes)),
        'formula_node_index': torch.tensor(formula_node_indices, dtype=torch.long),
        'formula_h_shift': torch.tensor(formula_h_shifts, dtype=torch.long),
        'node_atom_index': torch.tensor(node_atom_indices, dtype=torch.long),
        'node_atom_ptr': torch.tensor(node_atom_ptr, dtype=torch.long),
        'node_batch': torch.tensor(node_batch, dtype=torch.long),
        'node_features': node_features_tensor,
        'node_formula_counts': node_formula_counts_tensor,
        'node_edge_index': node_edge_index,
        'node_edge_attr': node_edge_attr,
        'batch': torch.tensor(fragment_batch, dtype=torch.long),
        'mz': torch.tensor(fragment_mzs, dtype=torch.get_default_dtype()),
        'bin': torch.tensor(fragment_bins, dtype=torch.long),
        'log_prior': torch.tensor(fragment_log_priors, dtype=torch.get_default_dtype()),
        'formula_index': torch.tensor(fragment_formula_indices, dtype=torch.long),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'bond_atom_index': bond_atom_index,
        'bond_ptr': torch.tensor(bond_ptr, dtype=torch.long),
        'bond_features': bond_features_tensor,
        'features': features_tensor,
        'formula_counts': formula_counts_tensor,
    }
