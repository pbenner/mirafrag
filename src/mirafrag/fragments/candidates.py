from __future__ import annotations

import math
from typing import Any

from rdkit import Chem

from mirafrag.adducts import (
    Adduct as FragmentAdduct,
)
from mirafrag.adducts import (
    parse_adduct as parse_fragment_adduct,
)
from mirafrag.fragments.config import FragmentConfig
from mirafrag.fragments.engine import _MiraFragFragmentEngine
from mirafrag.fragments.graph import _fragment_graph_edges
from mirafrag.fragments.isotopes import (
    _formula_isotope_peaks,
    _fragment_element_counts,
)
from mirafrag.fragments.mol import (
    _atom_implicit_hydrogens,
    _bond_break_stats,
    _fragment_break_score,
)


def smiles_to_fragment_candidates(
    smiles: str,
    *,
    mz_max: float,
    bin_width: float,
    adduct: str | None = None,
    config: FragmentConfig | None = None,
) -> dict[str, Any]:
    """
    Generate sparse fragment peak candidates for a molecule.

    The generator parses SMILES, builds a recursive atom-removal fragment tree, enumerates hydrogen transfers, applies adduct masses and optional isotope peaks, prunes candidate formulas, and constructs fragment-graph edges.
    """
    config = config or FragmentConfig()
    adduct_spec = parse_fragment_adduct(adduct, default_mass=float(config.proton_mass))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')

    num_bins = int(math.ceil(float(mz_max) / float(bin_width)))
    atom_hs = _atom_implicit_hydrogens(mol)
    bond_stats = _bond_break_stats(mol)
    candidates: dict[tuple[tuple[int, ...], int, int, int], dict[str, Any]] = {}
    _add_recursive_fragments(
        candidates=candidates,
        mol=mol,
        atom_hs=atom_hs,
        bond_stats=bond_stats,
        mz_max=mz_max,
        bin_width=bin_width,
        num_bins=num_bins,
        adduct_spec=adduct_spec,
        config=config,
    )

    formulas, peaks, peak_formula_index = _prune_fragment_candidates(
        candidates.values(),
        max_fragments=max(0, int(config.max_fragments)),
    )
    edge_index, edge_features = _fragment_graph_edges(
        formulas,
        max_edges=max(0, int(config.max_edges)),
        mz_max=float(mz_max),
        max_h_transfers=max(1, int(config.max_broken_bonds)),
        max_isotope_peaks=max(1, int(config.max_isotope_peaks)),
    )
    peak_groups = _formula_peak_groups(peaks)

    return {
        'atom_indices': [item['atom_indices'] for item in formulas],
        'mzs': [item['mz'] for item in peaks],
        'bins': [item['bin'] for item in peaks],
        'log_priors': [item['log_prior'] for item in peaks],
        'formula_index': peak_formula_index,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'features': [
            _formula_features(
                item,
                formula_peaks=peak_groups[(int(item['mask']), int(item['h_shift']))],
                mz_max=float(mz_max),
                max_tree_depth=max(float(config.max_tree_depth), 1.0),
                max_atoms=max(float(mol.GetNumAtoms()), 1.0),
                max_h_transfers=max(float(config.max_broken_bonds), 1.0),
                max_bonds=max(float(len(bond_stats)), 1.0),
                max_isotope_peaks=max(float(config.max_isotope_peaks), 1.0),
            )
            for item in formulas
        ],
    }


def _prune_fragment_candidates(
    candidates: Any,
    *,
    max_fragments: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """
    Select the retained fragment formulas and isotope peaks.

    Candidates are grouped by atom mask and hydrogen shift, ranked by fragmentation score and mass-related tie breakers, and trimmed to the configured formula budget.
    """
    if int(max_fragments) <= 0:
        return [], [], []

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault((int(item['mask']), int(item['h_shift'])), []).append(item)

    selected = sorted(
        grouped.values(),
        key=lambda group: _formula_sort_key(_representative_formula_peak(group)),
    )[:max_fragments]

    formulas: list[dict[str, Any]] = []
    peaks: list[dict[str, Any]] = []
    peak_formula_index: list[int] = []
    for formula_idx, group in enumerate(selected):
        ordered_peaks = sorted(
            group,
            key=lambda item: (int(item['isotope_rank']), float(item['mz'])),
        )
        formula = dict(_representative_formula_peak(ordered_peaks))
        formula['isotope_rank'] = 0
        formula['isotope_prob'] = 1.0
        formula['log_prior'] = 0.0
        formulas.append(formula)
        for peak in ordered_peaks:
            peaks.append(peak)
            peak_formula_index.append(formula_idx)
    return formulas, peaks, peak_formula_index


def _representative_formula_peak(group: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Choose the representative peak for a formula group.
    """
    return min(group, key=lambda item: (int(item['isotope_rank']), float(item['mz'])))


def _formula_sort_key(
    item: dict[str, Any],
) -> tuple[float, int, int, int, float, int, Any]:
    """
    Return the deterministic pruning key for a formula candidate.
    """
    return (
        float(item['score']),
        int(item.get('max_broken', item['num_broken_bonds'])),
        int(item['cut_count']),
        abs(int(item['h_shift'])),
        float(item['mz']),
        len(item['atom_indices']),
        item['atom_indices'],
    )


def _formula_features(
    item: dict[str, Any],
    *,
    formula_peaks: list[dict[str, Any]],
    mz_max: float,
    max_tree_depth: float,
    max_atoms: float,
    max_h_transfers: float,
    max_bonds: float,
    max_isotope_peaks: float,
) -> list[float]:
    """
    Build normalized hand-crafted features for one fragment formula.
    """
    isotope_prob = sum(float(peak['isotope_prob']) for peak in formula_peaks)
    return [
        float(item['mz']) / max(float(mz_max), 1.0),
        float(item['cut_count']) / max(float(max_tree_depth), 1.0),
        float(len(item['atom_indices'])) / max(float(max_atoms), 1.0),
        float(item['h_shift']) / max(float(max_h_transfers), 1.0),
        float(item['score']) / max(float(item['score_max']), 1.0),
        float(item['num_broken_bonds']) / max(float(max_bonds), 1.0),
        float(len(formula_peaks)) / max(float(max_isotope_peaks), 1.0),
        float(isotope_prob),
    ]


def _formula_peak_groups(
    peaks: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """
    Group isotope peak candidates by formula identity.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for peak in peaks:
        groups.setdefault((int(peak['mask']), int(peak['h_shift'])), []).append(peak)
    return groups


def _add_recursive_fragments(
    *,
    candidates: dict[tuple[tuple[int, ...], int, int, int], dict[str, Any]],
    mol: Chem.Mol,
    atom_hs: list[int],
    bond_stats: list[dict[str, Any]],
    mz_max: float,
    bin_width: float,
    num_bins: int,
    adduct_spec: FragmentAdduct,
    config: FragmentConfig,
) -> None:
    """
    Populate raw candidates from the recursive fragment engine.

    Each generated fragment mask is expanded over allowed hydrogen transfers, adduct masses, and isotope peaks before pruning.
    """
    engine = _MiraFragFragmentEngine(
        mol=mol,
        atom_hs=atom_hs,
        max_tree_depth=max(0, int(config.max_tree_depth)),
        max_broken_bonds=max(0, int(config.max_broken_bonds)),
    )
    entries = engine.generate_fragments()
    for fragment_hash, entry in entries:
        if int(entry['tree_depth']) == 0 and not config.include_root:
            continue
        _add_fragment_candidate(
            candidates,
            mol,
            mask=int(entry['frag']),
            fragment_hash=str(fragment_hash),
            parent_hashes=[str(value) for value in entry['parent_hashes']],
            atom_indices=_mask_to_atom_indices(int(entry['frag']), mol.GetNumAtoms()),
            cut_count=int(entry['tree_depth']),
            max_broken=int(entry['max_broken']),
            max_remove_hs=int(entry['max_remove_hs']),
            max_add_hs=int(entry['max_add_hs']),
            atom_hs=atom_hs,
            bond_stats=bond_stats,
            mz_max=mz_max,
            bin_width=bin_width,
            num_bins=num_bins,
            adduct_spec=adduct_spec,
            config=config,
        )


def _mask_to_atom_indices(mask: int, num_atoms: int) -> tuple[int, ...]:
    """
    Convert a bit-mask fragment representation to ordered atom indices.
    """
    return tuple(atom_idx for atom_idx in range(num_atoms) if mask & (1 << atom_idx))


def _add_fragment_candidate(
    candidates: dict[tuple[tuple[int, ...], int, int, int], dict[str, Any]],
    mol: Chem.Mol,
    *,
    mask: int,
    fragment_hash: str,
    parent_hashes: list[str],
    atom_indices: tuple[int, ...],
    cut_count: int,
    max_broken: int,
    max_remove_hs: int,
    max_add_hs: int,
    atom_hs: list[int],
    bond_stats: list[dict[str, Any]],
    mz_max: float,
    bin_width: float,
    num_bins: int,
    adduct_spec: FragmentAdduct,
    config: FragmentConfig,
) -> None:
    """
    Add all valid peak candidates for one fragment mask.

    The function computes break scores, allowed hydrogen shifts, formula masses, isotope priors, m/z bins, and keeps the best duplicate candidate for each atom/hydrogen/bin/isotope key.
    """
    if not atom_indices:
        return
    if len(atom_indices) == mol.GetNumAtoms() and int(cut_count) > 0:
        return

    atom_set = set(atom_indices)
    num_broken_bonds, score = _fragment_break_score(
        atom_set,
        bond_stats,
    )

    score_max = float(sum(item['score'] for item in bond_stats))
    for h_shift in range(-int(max_remove_hs), int(max_add_hs) + 1):
        element_counts = _fragment_element_counts(mol, atom_indices, atom_hs, h_shift)
        if not element_counts:
            continue
        isotope_peaks = _formula_isotope_peaks(
            element_counts,
            include_isotopes=bool(config.include_isotopes),
            threshold=float(config.isotope_threshold),
            max_peaks=max(1, int(config.max_isotope_peaks)),
        )
        for isotope_rank, (neutral_mass, isotope_prob) in enumerate(isotope_peaks):
            mz = adduct_spec.mz(float(neutral_mass))
            bin_idx = int(math.floor(mz / float(bin_width)))
            if bin_idx < 0 or bin_idx >= num_bins or mz > float(mz_max):
                continue

            key = (atom_indices, int(h_shift), int(bin_idx), int(isotope_rank))
            existing = candidates.get(key)
            if existing is not None:
                existing_rank = (
                    float(existing['score']),
                    int(existing.get('max_broken', existing['num_broken_bonds'])),
                    int(existing['cut_count']),
                )
                new_rank = (
                    float(score),
                    int(max_broken),
                    int(cut_count),
                )
                if existing_rank <= new_rank:
                    continue
            safe_prob = max(float(isotope_prob), 1e-12)
            candidates[key] = {
                'atom_indices': atom_indices,
                'mask': int(mask),
                'fragment_hash': str(fragment_hash),
                'parent_hashes': list(parent_hashes),
                'cut_count': int(cut_count),
                'h_shift': int(h_shift),
                'num_broken_bonds': int(num_broken_bonds),
                'max_broken': int(max_broken),
                'score': float(score),
                'score_max': score_max,
                'mz': mz,
                'bin': bin_idx,
                'isotope_rank': int(isotope_rank),
                'isotope_prob': safe_prob,
                'log_prior': math.log(safe_prob),
            }
