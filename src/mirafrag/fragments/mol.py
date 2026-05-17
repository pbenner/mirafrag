from __future__ import annotations

from typing import Any

from rdkit import Chem

from mirafrag.fragments.constants import _BOND_TYPE_WEIGHT, _HETERO_BOND_WEIGHT


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
