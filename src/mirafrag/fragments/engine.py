from __future__ import annotations

from collections import Counter
from hashlib import blake2b
from typing import Any

from rdkit import Chem

from mirafrag.fragments.constants import _BOND_TYPE_WEIGHT, _HETERO_BOND_WEIGHT


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
