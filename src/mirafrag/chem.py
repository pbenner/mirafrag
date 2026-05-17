from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem

TensorDict = dict[str, torch.Tensor]
PERIODIC_TABLE = Chem.GetPeriodicTable()


def quiet_rdkit_logs() -> None:
    """Suppress RDKit C++ logs that bypass tqdm's terminal handling."""
    RDLogger.DisableLog('rdApp.*')


@dataclass(frozen=True)
class GraphConfig:
    """
    Configuration for deterministic SMILES-to-3D-graph conversion.

    The config records encoder-supported atomic numbers, the radial cutoff used for edges, RDKit embedding behavior, geometry optimization options, and validation controls. It is part of the feature-cache key because changing it changes graph tensors.
    """

    atomic_numbers: tuple[int, ...]
    cutoff: float
    seed: int = 17
    add_hydrogens: bool = True
    optimize: bool = True
    max_embed_attempts: int = 50
    embed_timeout_seconds: int = 10
    fallback_to_2d: bool = True
    warn_2d_fallback: bool = False
    validate_bond_geometry: bool = True


def atomic_number_index(atomic_numbers: list[int] | tuple[int, ...]) -> dict[int, int]:
    """
    Map atomic numbers to one-hot column indices.

    The returned dictionary is used to build ``node_attrs`` tensors matching the selected foundation encoder's supported element ordering.
    """
    return {int(z): i for i, z in enumerate(atomic_numbers)}


def infer_graph_config(
    encoder,
    *,
    seed: int = 17,
    add_hydrogens: bool = True,
    optimize: bool = True,
) -> GraphConfig:
    """
    Build a :class:`GraphConfig` from a foundation encoder.

    Encoders are expected to expose ``atomic_numbers`` and ``r_max`` buffers or tensors. Those values define element filtering and graph edge construction for all downstream datasets.
    """
    atomic_numbers = tuple(int(z) for z in encoder.atomic_numbers.detach().cpu())
    cutoff = float(encoder.r_max.detach().cpu().item())
    return GraphConfig(
        atomic_numbers=atomic_numbers,
        cutoff=cutoff,
        seed=seed,
        add_hydrogens=add_hydrogens,
        optimize=optimize,
    )


def _embed_molecule(smiles: str, config: GraphConfig) -> Chem.Mol:
    """
    Generate a conformer-bearing RDKit molecule from SMILES.

    The routine adds hydrogens when configured, tries several ETKDG parameter variants, optionally relaxes coordinates, validates bonded distances, and may fall back to 2D coordinates for robustness.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    if config.add_hydrogens:
        mol = Chem.AddHs(mol)

    for params in _embedding_parameter_variants(config):
        candidate = Chem.Mol(mol)
        status = AllChem.EmbedMolecule(candidate, params)
        if status == 0:
            embedded = Chem.Mol(candidate)
            candidate = _optimize_molecule(candidate, config)
            if _has_reasonable_bond_geometry(candidate, config):
                return candidate
            if _has_reasonable_bond_geometry(embedded, config):
                return embedded

    if config.fallback_to_2d:
        candidate = Chem.Mol(mol)
        status = AllChem.Compute2DCoords(candidate)
        if status >= 0 and candidate.GetNumConformers() > 0:
            if config.warn_2d_fallback:
                warnings.warn(
                    f'RDKit 3D conformer generation failed for SMILES {smiles!r}; '
                    'falling back to 2D coordinates.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            if _has_reasonable_bond_geometry(candidate, config):
                return candidate

    raise ValueError(f'RDKit conformer generation failed for SMILES: {smiles!r}')


def _embedding_parameter_variants(config: GraphConfig):
    """
    Yield increasingly permissive RDKit embedding settings.

    The first variants preserve normal stereochemical and torsional assumptions. Later variants relax constraints to handle difficult molecules that otherwise block dataset loading.
    """
    variants = [
        {},
        {'useRandomCoords': True},
        {'enforceChirality': False},
        {'useRandomCoords': True, 'enforceChirality': False},
        {
            'useRandomCoords': True,
            'enforceChirality': False,
            'useBasicKnowledge': False,
            'useExpTorsionAnglePrefs': False,
            'ignoreSmoothingFailures': True,
        },
    ]
    for offset, options in enumerate(variants):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(config.seed) + offset
        _set_embed_param(params, 'timeout', int(config.embed_timeout_seconds))
        _set_embed_param(params, 'maxAttempts', int(config.max_embed_attempts))
        for name, value in options.items():
            _set_embed_param(params, name, value)
        yield params


def _set_embed_param(params, name: str, value) -> None:
    """
    Set an RDKit embedding parameter when available.

    RDKit exposes slightly different parameter attributes across versions. This helper keeps the code version-tolerant by ignoring missing attributes.
    """
    try:
        setattr(params, name, value)
    except AttributeError:
        pass


def _optimize_molecule(mol: Chem.Mol, config: GraphConfig) -> Chem.Mol:
    """
    Relax an embedded molecule with classical force fields.

    UFF is tried first because it has broad element coverage; MMFF is used as a fallback when all parameters are available. The input molecule is returned even when optimization fails.
    """
    if not config.optimize:
        return mol
    try:
        with rdBase.BlockLogs():
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        return mol
    except Exception:
        pass
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            with rdBase.BlockLogs():
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            return mol
    except Exception:
        pass
    return mol


def _has_reasonable_bond_geometry(mol: Chem.Mol, config: GraphConfig) -> bool:
    """
    Check whether bonded atoms have plausible distances.

    Bond lengths are compared against covalent-radius sums with broad tolerances. This rejects collapsed or exploded conformers before they are passed to a neural encoder.
    """
    if not config.validate_bond_geometry:
        return True
    if mol.GetNumConformers() == 0:
        return False

    conformer = mol.GetConformer()
    for bond in mol.GetBonds():
        begin_idx = int(bond.GetBeginAtomIdx())
        end_idx = int(bond.GetEndAtomIdx())
        begin = conformer.GetAtomPosition(begin_idx)
        end = conformer.GetAtomPosition(end_idx)
        distance = float(begin.Distance(end))

        begin_atom = mol.GetAtomWithIdx(begin_idx)
        end_atom = mol.GetAtomWithIdx(end_idx)
        begin_z = int(begin_atom.GetAtomicNum())
        end_z = int(end_atom.GetAtomicNum())

        radius_sum = float(PERIODIC_TABLE.GetRcovalent(begin_z)) + float(
            PERIODIC_TABLE.GetRcovalent(end_z)
        )
        if radius_sum <= 0.0:
            continue
        if distance < 0.55 * radius_sum or distance > 1.65 * radius_sum:
            return False

    return True


def _directed_edges(positions: np.ndarray, cutoff: float) -> torch.Tensor:
    """
    Create directed atom-pair edges under a distance cutoff.

    The output has shape ``(2, E)`` and includes both directions for every atom pair within the encoder cutoff, excluding self edges.
    """
    coords = torch.as_tensor(positions, dtype=torch.get_default_dtype())
    distances = torch.cdist(coords, coords)
    mask = (distances <= float(cutoff)) & ~torch.eye(
        coords.shape[0], dtype=torch.bool, device=coords.device
    )
    edge_index = mask.nonzero(as_tuple=False).T.contiguous()
    return edge_index.to(dtype=torch.long)


def smiles_to_graph(smiles: str, config: GraphConfig) -> TensorDict:
    """
    Convert a SMILES string into a foundation-encoder graph.

    The returned dictionary contains positions, atomic numbers, one-hot node attributes, directed edges, zero periodic shifts, cell, and related tensors expected by the MACE and AIMNet adapters.
    """
    mol = _embed_molecule(smiles, config)
    conformer = mol.GetConformer()
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)

    z_to_idx = atomic_number_index(config.atomic_numbers)
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    missing = sorted({z for z in atom_numbers if z not in z_to_idx})
    if missing:
        raise ValueError(
            'Foundation encoder does not support atomic numbers '
            f'{missing}; supported={list(config.atomic_numbers)}'
        )

    node_attrs = torch.zeros(
        len(atom_numbers),
        len(config.atomic_numbers),
        dtype=torch.get_default_dtype(),
    )
    for atom_idx, z in enumerate(atom_numbers):
        node_attrs[atom_idx, z_to_idx[z]] = 1.0

    edge_index = _directed_edges(positions, config.cutoff)
    num_edges = int(edge_index.shape[1])
    return {
        'positions': torch.as_tensor(positions, dtype=torch.get_default_dtype()),
        'atomic_numbers': torch.as_tensor(atom_numbers, dtype=torch.long),
        'node_attrs': node_attrs,
        'edge_index': edge_index,
        'shifts': torch.zeros(num_edges, 3, dtype=torch.get_default_dtype()),
        'unit_shifts': torch.zeros(num_edges, 3, dtype=torch.get_default_dtype()),
        'cell': torch.zeros(1, 3, 3, dtype=torch.get_default_dtype()),
    }


def collate_graphs(graphs: list[Mapping[str, torch.Tensor]]) -> TensorDict:
    """
    Batch per-molecule graph dictionaries into one graph.

    Node tensors are concatenated, edge indices are offset, and ``batch`` plus ``ptr`` tensors are added so model code can recover molecule membership.
    """
    if not graphs:
        raise ValueError('Cannot collate an empty graph list.')

    positions = []
    atomic_numbers = []
    node_attrs = []
    edge_indices = []
    shifts = []
    unit_shifts = []
    batch = []
    ptr = [0]
    node_offset = 0

    for graph_idx, graph in enumerate(graphs):
        n_nodes = int(graph['positions'].shape[0])
        positions.append(graph['positions'])
        atomic_numbers.append(graph['atomic_numbers'])
        node_attrs.append(graph['node_attrs'])
        batch.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        if graph['edge_index'].numel() > 0:
            edge_indices.append(graph['edge_index'] + node_offset)
            shifts.append(graph['shifts'])
            unit_shifts.append(graph['unit_shifts'])
        node_offset += n_nodes
        ptr.append(node_offset)

    device = positions[0].device
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        shift_tensor = torch.cat(shifts, dim=0)
        unit_shift_tensor = torch.cat(unit_shifts, dim=0)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        shift_tensor = torch.empty(0, 3, dtype=torch.get_default_dtype(), device=device)
        unit_shift_tensor = torch.empty(
            0, 3, dtype=torch.get_default_dtype(), device=device
        )

    return {
        'positions': torch.cat(positions, dim=0),
        'atomic_numbers': torch.cat(atomic_numbers, dim=0),
        'node_attrs': torch.cat(node_attrs, dim=0),
        'edge_index': edge_index,
        'shifts': shift_tensor,
        'unit_shifts': unit_shift_tensor,
        'cell': torch.zeros(len(graphs), 3, 3, dtype=positions[0].dtype, device=device),
        'batch': torch.cat(batch, dim=0).to(device),
        'ptr': torch.tensor(ptr, dtype=torch.long, device=device),
        'head': torch.zeros(len(graphs), dtype=torch.long, device=device),
    }


def move_graph_to_device(graph: TensorDict, device: torch.device | str) -> TensorDict:
    """
    Move every tensor in a graph dictionary to a device.

    This small helper is useful for standalone graph workflows outside the full dataset batch mover.
    """
    return {key: value.to(device) for key, value in graph.items()}
