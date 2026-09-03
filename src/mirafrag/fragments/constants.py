from __future__ import annotations

from rdkit import Chem

_BOND_TYPE_WEIGHT = {
    Chem.BondType.AROMATIC: 2,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.SINGLE: 1,
}
_HETERO_BOND_WEIGHT = {False: 2, True: 1}
FRAGMENT_FEATURE_DIM = 8
FRAGMENT_EDGE_FEATURE_DIM = 8
BOND_BREAK_FEATURE_DIM = 3
FRAGMENT_FORMULA_ELEMENTS = (
    'C',
    'H',
    'N',
    'O',
    'P',
    'S',
    'F',
    'Cl',
    'Br',
    'I',
    'B',
    'Si',
    'Se',
)
FRAGMENT_FORMULA_DIM = len(FRAGMENT_FORMULA_ELEMENTS)
