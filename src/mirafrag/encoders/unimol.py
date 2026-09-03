from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import nn

UNIMOL_ATOMIC_NUMBERS = tuple(range(1, 119))
UNIMOL_R_MAX = 5.0
UNIMOL_MODES = ('trainable', 'frozen')


def load_unimol_encoder(
    *,
    model_name: str = 'unimolv1',
    model_size: str = '84m',
    pretrained_model_path: str | None = None,
    pretrained_dict_path: str | None = None,
    max_atoms: int = 512,
    mode: str = 'trainable',
    device: str | torch.device = 'cpu',
) -> nn.Module:
    """Build a Uni-Mol atom-representation encoder."""
    return UniMolNodeEncoder(
        model_name=model_name,
        model_size=model_size,
        pretrained_model_path=pretrained_model_path,
        pretrained_dict_path=pretrained_dict_path,
        max_atoms=max_atoms,
        mode=mode,
        device=device,
    )


class UniMolNodeEncoder(nn.Module):
    """Expose Uni-Mol atom-level representations through MiraFrag's encoder API.

    The default ``trainable`` mode calls the underlying Uni-Mol ``nn.Module``
    directly, so ``fine-tune-strategy=full`` can update Uni-Mol weights. The
    ``frozen`` mode keeps the old ``UniMolRepr`` inference wrapper for cheap
    feature-extractor experiments. Coordinates come from MiraFrag's cached
    graph, not from Uni-Mol's conformer generator, and hydrogens are retained
    so atom indices remain aligned with fragment candidates.
    """

    uses_molecular_charge = False

    def __init__(
        self,
        *,
        model_name: str = 'unimolv1',
        model_size: str = '84m',
        pretrained_model_path: str | None = None,
        pretrained_dict_path: str | None = None,
        max_atoms: int = 512,
        mode: str = 'trainable',
        device: str | torch.device = 'cpu',
    ) -> None:
        super().__init__()
        if max_atoms <= 0:
            raise ValueError('max_atoms must be positive.')
        mode = str(mode).lower()
        if mode not in UNIMOL_MODES:
            raise ValueError(
                f'Unknown Uni-Mol mode {mode!r}; expected one of: '
                f'{", ".join(UNIMOL_MODES)}.'
            )
        self.model_name = str(model_name).lower()
        if mode == 'frozen' and self.model_name == 'unimolv2':
            raise ValueError(
                'MiraFrag supports Uni-Mol2 through trainable mode only. '
                'The frozen UniMolRepr Uni-Mol2 path drops hydrogens and cannot '
                'preserve MiraFrag fragment atom alignment.'
            )
        self.model_size = str(model_size)
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_dict_path = pretrained_dict_path
        self.max_atoms = int(max_atoms)
        self.mode = mode
        self.uses_smiles = self.mode == 'trainable' and self.model_name == 'unimolv2'
        self._device_name = _unimol_device_name(device)
        self._repr = None
        self.unimol_model: nn.Module | None = None
        self.register_buffer(
            'atomic_numbers',
            torch.tensor(UNIMOL_ATOMIC_NUMBERS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'r_max',
            torch.tensor(float(UNIMOL_R_MAX), dtype=torch.float32),
            persistent=False,
        )
        if self.mode == 'frozen':
            self.eval()

    def train(self, mode: bool = True) -> UniMolNodeEncoder:
        super().train(mode)
        if self.mode == 'frozen':
            super().train(False)
        return self

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        *,
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_node_feats: bool = True,
        molecular_charge: torch.Tensor | None = None,
        smiles: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del compute_force, compute_virials, compute_stress, molecular_charge
        if not compute_node_feats:
            raise ValueError('Uni-Mol encoder is only used for node features.')
        atomic_numbers = graph['atomic_numbers'].detach().cpu().long()
        positions = graph['positions'].detach().cpu().float()
        ptr = graph.get('ptr')
        if ptr is None:
            ptr_cpu = torch.tensor([0, int(atomic_numbers.numel())], dtype=torch.long)
        else:
            ptr_cpu = ptr.detach().cpu().long()

        if self.mode == 'frozen':
            outputs = self._encode_batch_frozen(atomic_numbers, positions, ptr_cpu)
            if outputs:
                node_feats = torch.cat(outputs, dim=0)
            else:
                node_feats = torch.empty(0, 0, dtype=torch.float32)
            return {'node_feats': node_feats.to(device=graph['positions'].device)}

        outputs = self._encode_batch_trainable(
            atomic_numbers,
            positions,
            ptr_cpu,
            training=training,
            smiles=smiles,
        )
        if outputs:
            node_feats = torch.cat(outputs, dim=0)
        else:
            node_feats = torch.empty(
                0,
                0,
                dtype=torch.float32,
                device=graph['positions'].device,
            )
        return {'node_feats': node_feats.to(device=graph['positions'].device)}

    def _input_features(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        ptr: torch.Tensor,
    ) -> tuple[list[dict[str, np.ndarray]], list[int]]:
        try:
            from unimol_tools.data.conformer import coords2unimol
        except ImportError as exc:
            raise ImportError(
                'Uni-Mol support requires unimol_tools. Run uv sync --extra '
                'unimol before ENCODER=unimol.'
            ) from exc

        dictionary = self._dictionary()
        features: list[dict[str, np.ndarray]] = []
        expected_lengths: list[int] = []
        for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True):
            molecule_numbers = atomic_numbers[start:end]
            molecule_positions = positions[start:end]
            if molecule_numbers.numel() > self.max_atoms:
                raise ValueError(
                    f'Uni-Mol max_atoms={self.max_atoms} is smaller than a molecule '
                    f'with {int(molecule_numbers.numel())} atoms. Increase '
                    '--unimol-max-atoms; silent Uni-Mol atom cropping would break '
                    'fragment atom indexing.'
                )
            expected_lengths.append(int(molecule_numbers.numel()))
            atoms = [_atomic_symbol(int(z)) for z in molecule_numbers.tolist()]
            coordinates = molecule_positions.numpy().astype(np.float32, copy=False)
            encoded = coords2unimol(
                atoms,
                coordinates,
                dictionary,
                max_atoms=self.max_atoms,
                remove_hs=False,
                data_type='molecule',
            )
            features.append(encoded)
        return features, expected_lengths

    def _encode_batch_trainable(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        ptr: torch.Tensor,
        *,
        training: bool,
        smiles: list[str] | None = None,
    ) -> list[torch.Tensor]:
        if self.model_name == 'unimolv2':
            return self._encode_batch_trainable_v2(
                atomic_numbers,
                positions,
                ptr,
                training=training,
                smiles=smiles,
            )
        features, expected_lengths = self._input_features(
            atomic_numbers, positions, ptr
        )
        if not features:
            return []
        model = self._unimol_model()
        model.train(training)
        samples = [(feature, None) for feature in features]
        batch, _labels = model.batch_collate_fn(samples)
        device = next(model.parameters()).device
        net_input = {
            key: value.to(device=device) if hasattr(value, 'to') else value
            for key, value in batch.items()
        }
        if self.model_name == 'unimolv1':
            return self._unimolv1_atomic_reprs(model, net_input, expected_lengths)
        result = model(
            **net_input,
            return_repr=True,
            return_atomic_reprs=True,
        )
        atomic_reprs = result.get('atomic_reprs') if isinstance(result, dict) else None
        return _validate_atomic_reprs(atomic_reprs, expected_lengths)

    def _input_features_v2(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        ptr: torch.Tensor,
        smiles: list[str] | None,
    ) -> tuple[list[dict[str, np.ndarray]], list[int], list[torch.Tensor]]:
        try:
            from rdkit import Chem, Geometry
            from unimol_tools.data.conformer import mol2unimolv2
        except ImportError as exc:
            raise ImportError(
                'Trainable Uni-Mol2 support requires rdkit and unimol_tools. '
                'Run uv sync --extra unimol before ENCODER=unimol '
                'UNIMOL_MODEL_NAME=unimolv2.'
            ) from exc

        molecule_count = max(int(ptr.numel()) - 1, 0)
        if smiles is None or len(smiles) != molecule_count:
            raise ValueError(
                'Trainable Uni-Mol2 requires one SMILES string per graph molecule.'
            )

        features: list[dict[str, np.ndarray]] = []
        expected_lengths: list[int] = []
        full_to_heavy_indices: list[torch.Tensor] = []
        for mol_idx, (start, end) in enumerate(
            zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True)
        ):
            molecule_numbers = atomic_numbers[start:end]
            molecule_positions = positions[start:end]
            heavy_mask = molecule_numbers.ne(1)
            heavy_indices = torch.nonzero(heavy_mask, as_tuple=False).flatten()
            heavy_count = int(heavy_indices.numel())
            if heavy_count <= 0:
                raise ValueError(
                    'Uni-Mol2 cannot encode a molecule with no heavy atoms.'
                )
            if heavy_count > self.max_atoms:
                raise ValueError(
                    f'Uni-Mol2 max_atoms={self.max_atoms} is smaller than a molecule '
                    f'with {heavy_count} heavy atoms. Increase --unimol-max-atoms; '
                    'silent Uni-Mol atom cropping would break fragment atom indexing.'
                )

            mol = Chem.MolFromSmiles(str(smiles[mol_idx]))
            if mol is None:
                raise ValueError(f'Invalid SMILES for Uni-Mol2: {smiles[mol_idx]!r}')
            if mol.GetNumAtoms() != heavy_count:
                raise ValueError(
                    'Uni-Mol2 heavy-atom count does not match MiraFrag graph atom '
                    f'order for SMILES {smiles[mol_idx]!r}: '
                    f'smiles_atoms={mol.GetNumAtoms()} graph_heavy_atoms={heavy_count}.'
                )

            conf = Chem.Conformer(heavy_count)
            heavy_positions = molecule_positions[heavy_indices]
            for atom_idx, coord in enumerate(heavy_positions.tolist()):
                conf.SetAtomPosition(
                    atom_idx,
                    Geometry.Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
                )
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            features.append(mol2unimolv2(mol, max_atoms=self.max_atoms, remove_hs=True))
            expected_lengths.append(heavy_count)
            full_to_heavy_indices.append(
                _full_to_heavy_index(
                    molecule_numbers, molecule_positions, heavy_indices
                )
            )
        return features, expected_lengths, full_to_heavy_indices

    def _encode_batch_trainable_v2(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        ptr: torch.Tensor,
        *,
        training: bool,
        smiles: list[str] | None,
    ) -> list[torch.Tensor]:
        features, expected_lengths, full_to_heavy_indices = self._input_features_v2(
            atomic_numbers, positions, ptr, smiles
        )
        if not features:
            return []
        model = self._unimol_model()
        model.train(training)
        samples = [(feature, None) for feature in features]
        batch, _labels = model.batch_collate_fn(samples)
        device = next(model.parameters()).device
        net_input = {
            key: value.to(device=device) if hasattr(value, 'to') else value
            for key, value in batch.items()
        }
        result = model(
            **net_input,
            return_repr=True,
            return_atomic_reprs=True,
        )
        atomic_reprs = result.get('atomic_reprs') if isinstance(result, dict) else None
        heavy_outputs = _validate_atomic_reprs(atomic_reprs, expected_lengths)
        return [
            heavy.index_select(0, mapping.to(device=heavy.device))
            for heavy, mapping in zip(heavy_outputs, full_to_heavy_indices, strict=True)
        ]

    def _unimolv1_atomic_reprs(
        self,
        model: nn.Module,
        net_input: dict[str, torch.Tensor],
        expected_lengths: list[int],
    ) -> list[torch.Tensor]:
        src_tokens = net_input['src_tokens']
        src_distance = net_input['src_distance']
        src_edge_type = net_input['src_edge_type']
        padding_mask = src_tokens.eq(model.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = model.embed_tokens(src_tokens)
        n_node = src_distance.size(-1)
        gbf_feature = model.gbf(src_distance, src_edge_type)
        graph_attn_bias = model.gbf_proj(gbf_feature)
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        encoder_rep, *_unused = model.encoder(
            x,
            padding_mask=padding_mask,
            attn_mask=graph_attn_bias,
        )
        outputs: list[torch.Tensor] = []
        for row, expected_length in enumerate(expected_lengths):
            # Uni-Mol v1 input is [CLS], atoms..., [SEP]. Slice atoms directly;
            # the package helper filters special IDs inconsistently for all-H dicts.
            tensor = encoder_rep[row, 1 : expected_length + 1, :]
            outputs.append(tensor)
        return _validate_atomic_reprs(outputs, expected_lengths)

    def _encode_batch_frozen(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        ptr: torch.Tensor,
    ) -> list[torch.Tensor]:
        atom_lists: list[list[str]] = []
        coordinate_lists: list[np.ndarray] = []
        expected_lengths: list[int] = []
        for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist(), strict=True):
            molecule_numbers = atomic_numbers[start:end]
            molecule_positions = positions[start:end]
            if molecule_numbers.numel() > self.max_atoms:
                raise ValueError(
                    f'Uni-Mol max_atoms={self.max_atoms} is smaller than a molecule '
                    f'with {int(molecule_numbers.numel())} atoms. Increase '
                    '--unimol-max-atoms; silent Uni-Mol atom cropping would break '
                    'fragment atom indexing.'
                )
            expected_lengths.append(int(molecule_numbers.numel()))
            atom_lists.append(
                [_atomic_symbol(int(z)) for z in molecule_numbers.tolist()]
            )
            coordinate_lists.append(
                molecule_positions.numpy().astype(np.float32, copy=False)
            )

        data = {'atoms': atom_lists, 'coordinates': coordinate_lists}
        result = self._unimol_repr().get_repr(data, return_atomic_reprs=True)
        atomic_reprs = result.get('atomic_reprs') if isinstance(result, dict) else None
        return _validate_atomic_reprs(atomic_reprs, expected_lengths)

    def _dictionary(self):
        if self.mode == 'trainable':
            return self._unimol_model().dictionary
        return self._unimol_repr().model.dictionary

    def _unimol_model(self) -> nn.Module:
        if self.unimol_model is None:
            try:
                from unimol_tools.models import UniMolModel
                from unimol_tools.models.unimolv2 import UniMolV2Model
            except ImportError as exc:
                raise ImportError(
                    'Trainable Uni-Mol support requires unimol_tools. Run uv sync '
                    '--extra unimol before ENCODER=unimol.'
                ) from exc
            if self.model_name == 'unimolv1':
                self.unimol_model = UniMolModel(
                    output_dim=1,
                    data_type='molecule',
                    remove_hs=False,
                    pretrained_model_path=self.pretrained_model_path,
                    pretrained_dict_path=self.pretrained_dict_path,
                ).to(torch.device(self._device_name))
            elif self.model_name == 'unimolv2':
                if self.pretrained_dict_path is not None:
                    raise ValueError(
                        'Uni-Mol2 does not use --unimol-pretrained-dict-path.'
                    )
                self.unimol_model = UniMolV2Model(
                    output_dim=1,
                    model_size=self.model_size,
                    pretrained_model_path=self.pretrained_model_path,
                ).to(torch.device(self._device_name))
            else:
                raise ValueError(
                    'Trainable MiraFrag Uni-Mol supports model_name unimolv1 or '
                    f'unimolv2, got {self.model_name!r}.'
                )
        model = self.unimol_model
        assert model is not None
        return model

    def _unimol_repr(self):
        if self._repr is None:
            try:
                from unimol_tools import UniMolRepr
            except ImportError as exc:
                raise ImportError(
                    'Uni-Mol support requires unimol_tools. Run uv sync --extra '
                    'unimol before ENCODER=unimol.'
                ) from exc
            self._repr = UniMolRepr(
                data_type='molecule',
                batch_size=1,
                remove_hs=False,
                model_name=self.model_name,
                model_size=self.model_size,
                use_cuda=self._device_name.startswith('cuda'),
                pretrained_model_path=self.pretrained_model_path,
                pretrained_dict_path=self.pretrained_dict_path,
                max_atoms=self.max_atoms,
            )
        return self._repr


def _full_to_heavy_index(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    heavy_indices: torch.Tensor,
) -> torch.Tensor:
    full_length = int(atomic_numbers.numel())
    heavy_count = int(heavy_indices.numel())
    if heavy_count <= 0:
        raise ValueError('Cannot align Uni-Mol2 atoms without heavy atoms.')
    mapping = torch.empty(full_length, dtype=torch.long)
    for heavy_row, atom_idx in enumerate(heavy_indices.tolist()):
        mapping[int(atom_idx)] = int(heavy_row)
    hydrogen_indices = torch.nonzero(atomic_numbers.eq(1), as_tuple=False).flatten()
    if hydrogen_indices.numel() > 0:
        heavy_positions = positions[heavy_indices].float()
        hydrogen_positions = positions[hydrogen_indices].float()
        nearest = torch.cdist(hydrogen_positions, heavy_positions).argmin(dim=1).long()
        mapping[hydrogen_indices] = nearest
    return mapping


def _validate_atomic_reprs(
    atomic_reprs: object,
    expected_lengths: list[int],
) -> list[torch.Tensor]:
    if atomic_reprs is None or len(atomic_reprs) != len(expected_lengths):
        raise RuntimeError(
            'Uni-Mol did not return one atomic representation array per molecule.'
        )

    outputs: list[torch.Tensor] = []
    for features, expected_length in zip(atomic_reprs, expected_lengths, strict=True):
        if isinstance(features, torch.Tensor):
            tensor = features.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] != expected_length:
            raise RuntimeError(
                'Uni-Mol atomic representation shape does not match MiraFrag '
                'atom order: '
                f'features={tuple(tensor.shape)} atoms={expected_length}.'
            )
        outputs.append(tensor)
    return outputs


def _unimol_device_name(device: str | torch.device) -> str:
    value = str(device)
    if value == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return value


@lru_cache(maxsize=128)
def _atomic_symbol(atomic_number: int) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - rdkit is a core dependency
        raise ImportError('RDKit is required for Uni-Mol atom symbols.') from exc
    symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number))
    if not symbol:
        raise ValueError(f'Unknown atomic number: {atomic_number}.')
    return symbol
