from __future__ import annotations

from typing import Any

import torch

from mirafrag.fragments.constants import FRAGMENT_EDGE_FEATURE_DIM, FRAGMENT_FEATURE_DIM


def collate_fragment_candidates(
    fragments: list[dict[str, Any]],
    *,
    node_offsets: list[int],
) -> dict[str, torch.Tensor]:
    atom_indices: list[int] = []
    atom_ptr = [0]
    formula_batch: list[int] = []
    fragment_batch: list[int] = []
    fragment_mzs: list[float] = []
    fragment_bins: list[int] = []
    fragment_log_priors: list[float] = []
    fragment_formula_indices: list[int] = []
    fragment_features: list[list[float]] = []
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
            fragment_features.append([float(value) for value in feature])
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
    else:
        features_tensor = torch.empty(
            0,
            FRAGMENT_FEATURE_DIM,
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
        'batch': torch.tensor(fragment_batch, dtype=torch.long),
        'mz': torch.tensor(fragment_mzs, dtype=torch.get_default_dtype()),
        'bin': torch.tensor(fragment_bins, dtype=torch.long),
        'log_prior': torch.tensor(fragment_log_priors, dtype=torch.get_default_dtype()),
        'formula_index': torch.tensor(fragment_formula_indices, dtype=torch.long),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'features': features_tensor,
    }
