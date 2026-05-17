import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from mirafrag.chem import (
    GraphConfig,
    _has_reasonable_bond_geometry,
    collate_graphs,
    smiles_to_graph,
)


def test_smiles_to_graph_and_collate():
    config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    graph = smiles_to_graph('CCO', config)
    assert graph['positions'].shape[1] == 3
    assert graph['node_attrs'].shape[1] == 3
    assert graph['edge_index'].shape[0] == 2

    batch = collate_graphs([graph, graph])
    assert batch['ptr'].tolist() == [
        0,
        graph['positions'].shape[0],
        graph['positions'].shape[0] * 2,
    ]
    assert batch['batch'].dtype == torch.long


def test_smiles_to_graph_accepts_reasonable_sulfur_hydrogen_bond():
    config = GraphConfig(atomic_numbers=(1, 6, 7, 8, 16), cutoff=5.0, seed=17)
    graph = smiles_to_graph('C(CS)C(C(=O)O)N', config)
    assert graph['positions'].shape[0] > 0


def test_bond_geometry_validation_rejects_collapsed_hydrogen_bond():
    config = GraphConfig(atomic_numbers=(1, 6), cutoff=5.0)
    mol = Chem.AddHs(Chem.MolFromSmiles('C'))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for atom_idx in range(1, mol.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, Point3D(1.09, 0.0, 0.0))
    mol.AddConformer(conformer)

    assert _has_reasonable_bond_geometry(mol, config)

    mol.GetConformer().SetAtomPosition(1, Point3D(0.3, 0.0, 0.0))

    assert not _has_reasonable_bond_geometry(mol, config)
