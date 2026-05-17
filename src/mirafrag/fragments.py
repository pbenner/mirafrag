from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import torch
from rdkit import Chem

from mirafrag.adducts import parse_adduct_charge_suffix

# Fragment candidate generation is adapted from FraGNNet's BSD-2-Clause
# FragmentEngine and fragment utilities.
# Copyright (c) 2025, Adamo Young and Fei Wang.
ELECTRON_MASS = 0.00054858
PERIODIC_TABLE = Chem.GetPeriodicTable()
HYDROGEN_MASS = float(PERIODIC_TABLE.GetMostCommonIsotopeMass('H'))
PROTON_MASS = HYDROGEN_MASS - ELECTRON_MASS
SODIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('Na')) - ELECTRON_MASS
)
POTASSIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('K')) - ELECTRON_MASS
)
AMMONIUM_ADDUCT_MASS = (
    float(PERIODIC_TABLE.GetMostCommonIsotopeMass('N'))
    + 4.0 * HYDROGEN_MASS
    - ELECTRON_MASS
)
ADDUCT_FORMULA_ALIASES = {
    'ACN': 'C2H3N',
    'FA': 'CH2O2',
    'HFA': 'CH2O2',
    'FORMATE': 'CHO2',
    'HCOO': 'CHO2',
    'HCOOH': 'CH2O2',
    'AC': 'C2H4O2',
    'HAC': 'C2H4O2',
}

_BOND_TYPE_WEIGHT = {
    Chem.BondType.AROMATIC: 2,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.SINGLE: 1,
}
_HETERO_BOND_WEIGHT = {False: 2, True: 1}
FRAGMENT_FEATURE_DIM = 8
FRAGMENT_EDGE_FEATURE_DIM = 8

_ISOTOPE_MASS_NUMBERS = {
    1: (1, 2),
    6: (12, 13),
    7: (14, 15),
    8: (16, 17, 18),
    9: (19,),
    11: (23,),
    15: (31,),
    16: (32, 33, 34, 36),
    17: (35, 37),
    19: (39, 41),
    35: (79, 81),
    53: (127,),
}
_ELEMENT_ISOTOPE_CACHE: dict[str, list[tuple[float, float]]] = {}
_FORMULA_ISOTOPE_CACHE: dict[
    tuple[tuple[tuple[str, int], ...], float, int],
    list[tuple[float, float]],
] = {}


@dataclass(frozen=True)
class FragmentConfig:
    max_tree_depth: int = 3
    max_broken_bonds: int = 6
    max_fragments: int = 2048
    max_edges: int = 8192
    include_isotopes: bool = True
    isotope_threshold: float = 0.001
    max_isotope_peaks: int = 1
    include_root: bool = True
    proton_mass: float = PROTON_MASS


def fragment_config_from_model_config(config: Any) -> FragmentConfig:
    return FragmentConfig(
        max_tree_depth=config.max_fragment_tree_depth,
        max_broken_bonds=config.max_fragment_broken_bonds,
        max_fragments=config.max_fragments,
        max_edges=config.max_fragment_edges,
        include_isotopes=config.include_fragment_isotopes,
        isotope_threshold=config.fragment_isotope_threshold,
        max_isotope_peaks=config.max_fragment_isotope_peaks,
    )


def smiles_to_fragment_candidates(
    smiles: str,
    *,
    mz_max: float,
    bin_width: float,
    adduct: str | None = None,
    config: FragmentConfig | None = None,
) -> dict[str, Any]:
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
    return min(group, key=lambda item: (int(item['isotope_rank']), float(item['mz'])))


def _formula_sort_key(
    item: dict[str, Any],
) -> tuple[float, int, int, int, float, int, Any]:
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
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for peak in peaks:
        groups.setdefault((int(peak['mask']), int(peak['h_shift'])), []).append(peak)
    return groups


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


class _MiraFragFragmentEngine:
    """FraGNNet FragmentEngine tree logic with MiraFrag atom ordering."""

    def __init__(
        self,
        *,
        mol: Chem.Mol,
        atom_hs: list[int],
        max_tree_depth: int,
        max_broken_bonds: int,
    ) -> None:
        self.mol = Chem.Mol(mol)
        try:
            Chem.Kekulize(self.mol, clearAromaticFlags=True)
        except Exception:
            pass
        self.natoms = int(self.mol.GetNumAtoms())
        self.atom_symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.atom_hs = [int(value) for value in atom_hs]
        self.total_hs = int(sum(self.atom_hs))
        self.max_tree_depth = int(max_tree_depth)
        self.max_broken_bonds = int(max_broken_bonds)
        self.bonded_atoms: list[list[int]] = [[] for _ in range(self.natoms)]
        self.bond_to_type: dict[int, int] = {}
        self.bonds: set[int] = set()
        self.bondscore: dict[int, int] = {}
        for bond in self.mol.GetBonds():
            begin = int(bond.GetBeginAtomIdx())
            end = int(bond.GetEndAtomIdx())
            self.bonded_atoms[begin].append(end)
            self.bonded_atoms[end].append(begin)
            bond_bits = (1 << begin) | (1 << end)
            bond_type = int(_BOND_TYPE_WEIGHT.get(bond.GetBondType(), 1))
            hetero_weight = int(
                _HETERO_BOND_WEIGHT[
                    self.atom_symbols[begin] != 'C' or self.atom_symbols[end] != 'C'
                ]
            )
            self.bond_to_type[bond_bits] = bond_type
            self.bondscore[bond_bits] = bond_type * hetero_weight
            self.bonds.add(bond_bits)
        self.frag_to_entry: dict[str, dict[str, Any]] = {}

    def generate_fragments(self) -> list[tuple[str, dict[str, Any]]]:
        current_id = 0
        root_mask = (1 << self.natoms) - 1
        root_hash = self.wl_hash(root_mask)
        root = {
            'frag': root_mask,
            'id': current_id,
            'parents': [],
            'parent_hashes': [],
            'max_broken': 0,
            'tree_depth': 0,
            'score': self.score_fragment(root_mask)[1],
        }
        root.update(self.atom_pass_stats(root_mask, depth=0))
        self.frag_to_entry[root_hash] = root

        current_fragments = [root_hash]
        new_fragments: list[str] = []
        for step in range(self.max_tree_depth):
            for fragment_hash in current_fragments:
                parent = self.frag_to_entry[fragment_hash]
                parent_id = int(parent['id'])
                parent_mask = int(parent['frag'])
                parent_broken = int(parent['max_broken'])
                for atom_idx in range(self.natoms):
                    extended_fragments = self.remove_atom(parent_mask, atom_idx)
                    for fragment in extended_fragments:
                        new_hash = str(fragment['new_hash'])
                        new_mask = int(fragment['new_frag'])
                        removed_bond_order = int(fragment['rm_bond_t'])
                        max_broken = parent_broken + removed_bond_order
                        if max_broken > self.max_broken_bonds:
                            continue

                        old_entry = self.frag_to_entry.get(new_hash)
                        if old_entry is None:
                            current_id += 1
                            new_entry = {
                                'frag': new_mask,
                                'id': current_id,
                                'parents': [parent_id],
                                'parent_hashes': [fragment_hash],
                                'max_broken': max_broken,
                                'tree_depth': step + 1,
                                'score': self.score_fragment(new_mask)[1],
                            }
                            new_entry.update(
                                self.atom_pass_stats(new_mask, depth=max_broken)
                            )
                            self.frag_to_entry[new_hash] = new_entry
                            new_fragments.append(new_hash)
                        elif int(old_entry['max_broken']) == max_broken:
                            old_entry['parents'].append(parent_id)
                            old_entry['parent_hashes'].append(fragment_hash)
            current_fragments = new_fragments
            new_fragments = []

        return sorted(
            self.frag_to_entry.items(),
            key=lambda item: (int(item[1]['tree_depth']), int(item[1]['id'])),
        )

    def score_fragment(self, fragment: int) -> tuple[int, int]:
        score = 0
        breaks = 0
        for bond_bits in self.bonds:
            if 0 < (int(fragment) & bond_bits) < bond_bits:
                score += int(self.bondscore[bond_bits])
                breaks += 1
        return breaks, score

    def atom_pass_stats(self, fragment: int, *, depth: int | None) -> dict[str, int]:
        frag_hs = 0
        for atom_idx in range(self.natoms):
            if int(fragment) & (1 << atom_idx):
                frag_hs += int(self.atom_hs[atom_idx])
        max_remove = min(frag_hs, self.max_broken_bonds)
        max_add = min(self.total_hs - frag_hs, self.max_broken_bonds)
        if depth is not None:
            max_remove = min(int(depth), int(max_remove))
            max_add = min(int(depth), int(max_add))
        return {
            'frag_hs': int(frag_hs),
            'max_remove_hs': int(max_remove),
            'max_add_hs': int(max_add),
        }

    def remove_atom(self, fragment: int, atom_idx: int) -> list[dict[str, Any]]:
        atom_bit = 1 << int(atom_idx)
        if not (atom_bit & int(fragment)):
            return []

        template = int(fragment) ^ atom_bit
        ext_atom_to_bo: dict[int, int] = {}
        for neighbor in self.bonded_atoms[int(atom_idx)]:
            if (1 << neighbor) & template:
                bond_bits = atom_bit | (1 << neighbor)
                ext_atom_to_bo[neighbor] = int(self.bond_to_type[bond_bits])

        if len(ext_atom_to_bo) == 1:
            if template == 0:
                return []
            removed_bond_order = next(iter(ext_atom_to_bo.values()))
            return [
                {
                    'new_frag': template,
                    'new_hash': self.wl_hash(template),
                    'removed_atom': int(atom_idx),
                    'rm_bond_t': removed_bond_order,
                }
            ]

        out: list[dict[str, Any]] = []
        for neighbor, removed_bond_order in sorted(ext_atom_to_bo.items()):
            if any((1 << neighbor) & int(item['new_frag']) for item in out):
                continue
            new_fragment = _extend_fragment(neighbor, self.bonded_atoms, template)
            if new_fragment == 0:
                continue
            out.append(
                {
                    'new_frag': new_fragment,
                    'new_hash': self.wl_hash(new_fragment),
                    'removed_atom': int(atom_idx),
                    'rm_bond_t': int(removed_bond_order),
                }
            )
        return out

    def wl_hash(self, template_fragment: int) -> str:
        cur_hashes = [str(symbol) for symbol in self.atom_symbols]

        def graph_hash(full_hashes: list[str]) -> str:
            counter = Counter(full_hashes)
            counter_str = str(tuple(sorted(counter.items(), key=lambda item: item[0])))
            return _hash_label(counter_str)

        current_graph_hash = graph_hash(cur_hashes)
        iterations = self.natoms
        changed = True
        count = 0
        while count <= iterations and changed:
            new_hashes: list[str] = []
            template_atoms = 0
            for atom_idx in range(self.natoms):
                atom_bit = 1 << atom_idx
                cur_hash = cur_hashes[atom_idx]
                if not (atom_bit & int(template_fragment)):
                    new_hashes.append(cur_hash)
                    continue

                template_atoms += 1
                neighbor_labels = []
                for neighbor in self.bonded_atoms[atom_idx]:
                    neighbor_bit = 1 << neighbor
                    if not (neighbor_bit & int(template_fragment)):
                        continue
                    bond_bits = atom_bit | neighbor_bit
                    neighbor_labels.append(
                        f'{self.bond_to_type[bond_bits]}_{cur_hashes[neighbor]}'
                    )
                new_hashes.append(
                    _hash_label(cur_hash + ''.join(sorted(neighbor_labels)))
                )

            iterations = template_atoms
            next_graph_hash = graph_hash(new_hashes)
            changed = next_graph_hash != current_graph_hash
            current_graph_hash = next_graph_hash
            cur_hashes = new_hashes
            count += 1
        return current_graph_hash


def _extend_fragment(
    atom_idx: int,
    bonded_atoms: list[list[int]],
    template_fragment: int,
) -> int:
    root_bit = 1 << int(atom_idx)
    if not (root_bit & int(template_fragment)):
        return 0
    stack = [int(atom_idx)]
    new_fragment = root_bit
    while stack:
        atom = stack.pop()
        for neighbor in bonded_atoms[atom]:
            atom_bit = 1 << neighbor
            if not (atom_bit & int(template_fragment)) or atom_bit & new_fragment:
                continue
            new_fragment |= atom_bit
            stack.append(neighbor)
    return new_fragment


def _hash_label(label: str, digest_size: int = 32) -> str:
    return blake2b(label.encode('ascii'), digest_size=digest_size).hexdigest()


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
    add_edge,
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


def _fragment_element_counts(
    mol: Chem.Mol,
    atom_indices: tuple[int, ...],
    atom_hs: list[int],
    h_shift: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    hydrogen_count = int(h_shift)
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        hydrogen_count += int(atom_hs[int(atom_idx)])
    if hydrogen_count < 0:
        return {}
    if hydrogen_count > 0:
        counts['H'] = counts.get('H', 0) + hydrogen_count
    return {symbol: count for symbol, count in counts.items() if int(count) > 0}


def _formula_isotope_peaks(
    element_counts: dict[str, int],
    *,
    include_isotopes: bool,
    threshold: float,
    max_peaks: int,
) -> list[tuple[float, float]]:
    if not include_isotopes:
        mass = sum(
            float(count) * float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))
            for symbol, count in element_counts.items()
        )
        return [(mass, 1.0)]

    key = (
        tuple(
            sorted(
                (str(symbol), int(count)) for symbol, count in element_counts.items()
            )
        ),
        float(threshold),
        int(max_peaks),
    )
    cached = _FORMULA_ISOTOPE_CACHE.get(key)
    if cached is not None:
        return cached

    distribution: list[tuple[float, float]] = [(0.0, 1.0)]
    max_states = max(16, int(max_peaks) * 8)
    for symbol, count in sorted(element_counts.items()):
        isotopes = _element_isotopes(symbol)
        for _ in range(int(count)):
            distribution = _convolve_isotope_distribution(
                distribution,
                isotopes,
                threshold=max(float(threshold) * 0.01, 1e-12),
                max_states=max_states,
            )

    peaks = [
        (mass, prob)
        for mass, prob in sorted(distribution, key=lambda item: item[0])
        if float(prob) >= float(threshold)
    ]
    if not peaks:
        peaks = sorted(distribution, key=lambda item: item[1], reverse=True)[:1]
    peaks = sorted(peaks, key=lambda item: item[1], reverse=True)[: int(max_peaks)]
    peaks = sorted(peaks, key=lambda item: item[0])
    total = sum(float(prob) for _mass, prob in peaks)
    if total <= 0.0:
        peaks = [(float(peaks[0][0]), 1.0)]
    else:
        peaks = [(float(mass), float(prob) / total) for mass, prob in peaks]
    _FORMULA_ISOTOPE_CACHE[key] = peaks
    return peaks


def _element_isotopes(symbol: str) -> list[tuple[float, float]]:
    cached = _ELEMENT_ISOTOPE_CACHE.get(symbol)
    if cached is not None:
        return cached

    atomic_number = int(PERIODIC_TABLE.GetAtomicNumber(symbol))
    isotopes: list[tuple[float, float]] = []
    for mass_number in _ISOTOPE_MASS_NUMBERS.get(atomic_number, ()):
        abundance = float(
            PERIODIC_TABLE.GetAbundanceForIsotope(atomic_number, mass_number)
        )
        if abundance <= 0.0:
            continue
        isotopes.append(
            (
                float(PERIODIC_TABLE.GetMassForIsotope(atomic_number, mass_number)),
                abundance / 100.0,
            )
        )
    if not isotopes:
        isotopes = [(float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol)), 1.0)]
    total = sum(prob for _mass, prob in isotopes)
    isotopes = [(mass, prob / total) for mass, prob in isotopes]
    _ELEMENT_ISOTOPE_CACHE[symbol] = isotopes
    return isotopes


def _convolve_isotope_distribution(
    distribution: list[tuple[float, float]],
    isotopes: list[tuple[float, float]],
    *,
    threshold: float,
    max_states: int,
) -> list[tuple[float, float]]:
    merged: dict[float, float] = {}
    for base_mass, base_prob in distribution:
        for isotope_mass, isotope_prob in isotopes:
            prob = float(base_prob) * float(isotope_prob)
            if prob < float(threshold):
                continue
            mass_key = round(float(base_mass) + float(isotope_mass), 8)
            merged[mass_key] = merged.get(mass_key, 0.0) + prob
    if not merged:
        base_mass, base_prob = max(distribution, key=lambda item: item[1])
        isotope_mass, isotope_prob = max(isotopes, key=lambda item: item[1])
        merged[round(float(base_mass) + float(isotope_mass), 8)] = float(
            base_prob
        ) * float(isotope_prob)
    out = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    return [(float(mass), float(prob)) for mass, prob in out[: int(max_states)]]


@dataclass(frozen=True)
class FragmentAdduct:
    molecule_multiplier: int
    mass_delta: float
    charge: int

    def mz(self, neutral_mass: float) -> float:
        charge_abs = max(abs(int(self.charge)), 1)
        return (
            float(self.molecule_multiplier) * float(neutral_mass)
            + float(self.mass_delta)
        ) / float(charge_abs)


def fragment_adduct_mass(
    adduct: str | None, *, default_mass: float = PROTON_MASS
) -> float:
    return parse_fragment_adduct(adduct, default_mass=default_mass).mass_delta


def parse_fragment_adduct(
    adduct: str | None, *, default_mass: float = PROTON_MASS
) -> FragmentAdduct:
    if adduct is None:
        return FragmentAdduct(1, float(default_mass), 1)
    label = str(adduct).strip()
    if not label or label.lower() == 'nan':
        return FragmentAdduct(1, float(default_mass), 1)
    if label == '[M+H]+':
        return FragmentAdduct(1, float(default_mass), 1)
    if not label.startswith('['):
        raise ValueError(f'Unsupported adduct {label!r}; expected bracketed form.')
    try:
        close_idx = label.rindex(']')
    except ValueError as exc:
        raise ValueError(
            f'Unsupported adduct {label!r}; missing closing bracket.'
        ) from exc

    body = label[1:close_idx]
    charge = parse_adduct_charge_suffix(label[close_idx + 1 :])
    if charge == 0:
        raise ValueError(f'Unsupported adduct {label!r}; missing ion charge.')

    pos = 0
    multiplier_text = ''
    while pos < len(body) and body[pos].isdigit():
        multiplier_text += body[pos]
        pos += 1
    if pos >= len(body) or body[pos] != 'M':
        raise ValueError(f'Unsupported adduct {label!r}; expected M in adduct body.')
    molecule_multiplier = int(multiplier_text) if multiplier_text else 1

    neutral_delta = _adduct_expression_mass(body[pos + 1 :])
    mass_delta = neutral_delta - float(charge) * ELECTRON_MASS
    return FragmentAdduct(
        molecule_multiplier=max(1, int(molecule_multiplier)),
        mass_delta=float(mass_delta),
        charge=int(charge),
    )


def _adduct_expression_mass(expression: str) -> float:
    if not expression:
        return 0.0
    total = 0.0
    pos = 0
    while pos < len(expression):
        marker = expression[pos]
        if marker not in '+-':
            raise ValueError(f'Unsupported adduct expression {expression!r}.')
        sign = 1.0 if marker == '+' else -1.0
        pos += 1
        start = pos
        while pos < len(expression) and expression[pos] not in '+-':
            pos += 1
        term = expression[start:pos]
        if not term:
            raise ValueError(f'Unsupported adduct expression {expression!r}.')
        total += sign * _adduct_term_mass(term)
    return total


def _adduct_term_mass(term: str) -> float:
    formula = ADDUCT_FORMULA_ALIASES.get(term.upper(), term)
    pos = 0
    multiplier_text = ''
    while pos < len(formula) and formula[pos].isdigit():
        multiplier_text += formula[pos]
        pos += 1
    multiplier = int(multiplier_text) if multiplier_text else 1
    return float(multiplier) * _formula_mass(formula[pos:])


def _formula_mass(formula: str) -> float:
    if not formula:
        raise ValueError('Empty adduct formula.')
    total = 0.0
    pos = 0
    while pos < len(formula):
        if not formula[pos].isupper():
            raise ValueError(f'Unsupported adduct formula {formula!r}.')
        symbol = formula[pos]
        pos += 1
        if pos < len(formula) and formula[pos].islower():
            symbol += formula[pos]
            pos += 1
        count_text = ''
        while pos < len(formula) and formula[pos].isdigit():
            count_text += formula[pos]
            pos += 1
        count = int(count_text) if count_text else 1
        atomic_number = int(PERIODIC_TABLE.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise ValueError(f'Unsupported adduct element {symbol!r}.')
        total += float(count) * float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))
    return total


def _atom_implicit_hydrogens(mol: Chem.Mol) -> list[int]:
    return [
        int(atom.GetNumImplicitHs()) + int(atom.GetNumExplicitHs())
        for atom in mol.GetAtoms()
    ]


def _bond_break_stats(mol: Chem.Mol) -> list[dict[str, Any]]:
    stats = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        order_weight = int(_BOND_TYPE_WEIGHT.get(bond.GetBondType(), 1))
        begin_symbol = mol.GetAtomWithIdx(begin).GetSymbol()
        end_symbol = mol.GetAtomWithIdx(end).GetSymbol()
        hetero = begin_symbol != 'C' or end_symbol != 'C'
        score = order_weight * int(_HETERO_BOND_WEIGHT[hetero])
        stats.append(
            {
                'begin': begin,
                'end': end,
                'order_weight': order_weight,
                'score': float(score),
            }
        )
    return stats


def _fragment_break_score(
    atom_set: set[int],
    bond_stats: list[dict[str, Any]],
) -> tuple[int, float]:
    num_broken = 0
    score = 0.0
    for bond in bond_stats:
        begin_in = int(bond['begin']) in atom_set
        end_in = int(bond['end']) in atom_set
        if begin_in == end_in:
            continue
        num_broken += 1
        score += float(bond['score'])
    return num_broken, score
