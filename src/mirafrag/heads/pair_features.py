from __future__ import annotations

import torch


def unique_candidate_pair_index(
    fragments: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return canonical unique atom pairs used by the bond-action branch."""
    pairs = fragments.get('bond_atom_index')
    if pairs is None or not pairs.numel():
        device = pairs.device if pairs is not None else None
        return torch.empty((0, 2), dtype=torch.long, device=device)
    pairs = pairs.long().reshape(-1, 2)
    canonical = torch.stack([pairs.min(dim=1).values, pairs.max(dim=1).values], dim=1)
    return torch.unique(canonical, dim=0, sorted=True)


def align_directional_pair_features(
    pair_features: torch.Tensor,
    pair_edge_index: torch.Tensor,
    candidate_pairs: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Align sparse bidirectional pair features to oriented candidate bonds."""
    if candidate_pairs.ndim != 2 or candidate_pairs.shape[1] != 2:
        raise ValueError('Candidate pair indices must have shape [num_pairs, 2].')
    if pair_edge_index.ndim != 2 or pair_edge_index.shape[1] != 2:
        raise ValueError('Pair edge indices must have shape [num_pairs, 2].')
    if pair_features.ndim != 2 or pair_features.shape[0] != pair_edge_index.shape[0]:
        raise ValueError('Pair features must align row-wise with pair edge indices.')
    if pair_features.shape[1] % 2:
        raise ValueError('Directional pair features must have an even width.')
    if num_nodes < 0:
        raise ValueError('num_nodes must be nonnegative.')
    if candidate_pairs.numel() == 0:
        return pair_features.new_empty((0, int(pair_features.shape[1])))
    if pair_edge_index.numel() == 0:
        raise ValueError('Pair output is missing candidate bond atom pairs.')

    device = pair_features.device
    edges = pair_edge_index.to(device=device).long()
    candidates = candidate_pairs.to(device=device).long()
    if (
        int(edges.min()) < 0
        or int(candidates.min()) < 0
        or int(edges.max()) >= num_nodes
        or int(candidates.max()) >= num_nodes
    ):
        raise ValueError('Pair indices must refer to valid graph atom rows.')
    scale = max(int(num_nodes), 1)
    canonical_edges = torch.stack(
        [edges.min(dim=1).values, edges.max(dim=1).values], dim=1
    )
    edge_keys = canonical_edges[:, 0] * scale + canonical_edges[:, 1]
    sorted_keys, order = torch.sort(edge_keys)
    canonical_candidates = torch.stack(
        [candidates.min(dim=1).values, candidates.max(dim=1).values], dim=1
    )
    candidate_keys = canonical_candidates[:, 0] * scale + canonical_candidates[:, 1]
    positions = torch.searchsorted(sorted_keys, candidate_keys)
    safe_positions = positions.clamp_max(max(int(sorted_keys.numel()) - 1, 0))
    if not bool(sorted_keys[safe_positions].eq(candidate_keys).all()):
        raise ValueError('Pair output is missing a fragment breakpoint atom pair.')

    pair_rows = order[safe_positions]
    aligned = pair_features[pair_rows]
    reverse = candidates[:, 0].ne(edges[pair_rows, 0])
    if bool(reverse.any()):
        half = int(aligned.shape[-1]) // 2
        aligned = aligned.clone()
        aligned[reverse] = torch.cat(
            [aligned[reverse, half:], aligned[reverse, :half]], dim=-1
        )
    return aligned
