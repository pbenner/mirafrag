from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from mirafrag.adducts import parse_adduct_charge
from mirafrag.data import MetadataConfig
from mirafrag.delta import TorchDeltaFineTuneWrapper
from mirafrag.fragments import FRAGMENT_EDGE_FEATURE_DIM
from mirafrag.probability import fragment_oos_log_probs


@dataclass
class MiraFragConfig:
    num_bins: int
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    metadata_dim: int = 32
    max_fragment_tree_depth: int = 3
    max_fragment_broken_bonds: int = 6
    max_fragments: int = 2048
    max_fragment_edges: int = 8192
    include_fragment_isotopes: bool = True
    fragment_isotope_threshold: float = 0.001
    max_fragment_isotope_peaks: int = 1
    fragment_gnn_layers: int = 2
    encoder_type: str = 'mace'
    encoder_finetune_strategy: str = 'head'
    foundation_source: str = 'off'
    foundation_model: str | None = 'medium'
    foundation_path: str | None = None
    aimnet_model: str | None = 'aimnet2'
    aimnet_path: str | None = None


AIMNET2_ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)


def load_mace_encoder(
    *,
    source: str = 'off',
    model: str | None = 'medium',
    model_path: str | None = None,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    try:
        from mace_model import (
            convert_torch_model,
            download_foundation_model,
            load_serialized_torch_model,
        )
    except ImportError as exc:
        raise ImportError(
            'MACE support requires mace-model. Install it with '
            '`uv sync --extra mace` or run Make targets with ENCODER=mace.'
        ) from exc

    if model_path:
        torch_model, config = load_serialized_torch_model(Path(model_path))
        result = convert_torch_model(
            torch_model,
            backend='torch',
            device=str(device),
            config=config,
        )
    else:
        result = download_foundation_model(
            backend='torch',
            source=source,
            model=model,
            device=str(device),
        )
    mace_model = result.model
    mace_model = mace_model.to(device)
    mace_model.eval()
    return mace_model


def _adduct_charge_lookup(metadata_config: MetadataConfig) -> list[float]:
    size = max(metadata_config.adduct_to_idx.values(), default=-1) + 2
    charges = [0.0] * max(size, 1)
    for adduct, idx in metadata_config.adduct_to_idx.items():
        if 0 <= int(idx) < len(charges):
            charges[int(idx)] = float(parse_adduct_charge(adduct))
    return charges


def load_aimnet_encoder(
    *,
    model: str | None = 'aimnet2',
    model_path: str | None = None,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    return AimnetNodeEncoder(
        model=model_path or model or 'aimnet2',
        device=device,
    )


def load_foundation_encoder(
    *,
    encoder_type: str = 'mace',
    foundation_source: str = 'off',
    foundation_model: str | None = 'medium',
    foundation_path: str | None = None,
    aimnet_model: str | None = 'aimnet2',
    aimnet_path: str | None = None,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    if encoder_type == 'mace':
        return load_mace_encoder(
            source=foundation_source,
            model=foundation_model,
            model_path=foundation_path,
            device=device,
        )
    if encoder_type == 'aimnet':
        return load_aimnet_encoder(
            model=aimnet_model,
            model_path=aimnet_path,
            device=device,
        )
    raise ValueError(
        f'Unknown encoder_type {encoder_type!r}; expected one of: mace, aimnet.'
    )


class AimnetNodeEncoder(nn.Module):
    """Expose AIMNet2 hidden atom features through MiraFrag's encoder interface."""

    uses_molecular_charge = True

    def __init__(
        self,
        *,
        model: str = 'aimnet2',
        device: str | torch.device = 'cpu',
    ) -> None:
        super().__init__()
        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as exc:
            raise ImportError(
                'AIMNet support requires the aimnet package. Install the local '
                'aimnetcentral checkout or run `uv sync --extra aimnet`.'
            ) from exc

        self.model_name = str(model)
        self.calculator = AIMNet2Calculator(
            self.model_name,
            device=str(device),
            train=True,
        )
        self.model = self.calculator.model
        metadata = self.calculator.metadata or {}
        atomic_numbers = tuple(
            int(z) for z in metadata.get('implemented_species', AIMNET2_ATOMIC_NUMBERS)
        )
        if not atomic_numbers:
            atomic_numbers = AIMNET2_ATOMIC_NUMBERS
        self.register_buffer(
            'atomic_numbers',
            torch.tensor(atomic_numbers, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'r_max',
            torch.tensor(float(self.calculator.cutoff), dtype=torch.float32),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True) -> AimnetNodeEncoder:
        super().train(mode)
        self.model.train(mode)
        if hasattr(self.calculator, '_train'):
            self.calculator._train = bool(mode)
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
    ) -> dict[str, torch.Tensor]:
        del compute_force, compute_virials, compute_stress
        if bool(training) != self.training:
            self.train(bool(training))
        if not compute_node_feats:
            raise ValueError('AIMNet encoder is only used for node features.')

        device = self._device()
        self.calculator.device = str(device)
        positions = graph['positions'].to(device=device)
        atomic_numbers = graph['atomic_numbers'].to(device=device, dtype=torch.long)
        batch = graph.get('batch')
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
            batch_size = 1
        else:
            batch = batch.to(device=device, dtype=torch.long)
            batch_size = (
                int(graph['ptr'].numel() - 1)
                if 'ptr' in graph
                else int(batch.max().item()) + 1
            )

        if molecular_charge is None:
            charge = positions.new_zeros(batch_size)
        else:
            charge = molecular_charge.to(device=device, dtype=positions.dtype)
            if charge.numel() != batch_size:
                raise ValueError(
                    'AIMNet molecular_charge must have one value per molecule; '
                    f'got {charge.numel()} values for batch_size={batch_size}.'
                )
            charge = charge.reshape(batch_size)

        data = {
            'coord': positions,
            'numbers': atomic_numbers,
            'charge': charge,
            'mol_idx': batch,
        }
        prepared = self.calculator.prepare_input(data)
        if isinstance(self.model, torch.jit.ScriptModule):
            with torch.jit.optimized_execution(False):  # type: ignore[attr-defined]
                out = self.model(prepared)
        else:
            out = self.model(prepared)
        if 'aim' not in out:
            raise RuntimeError(
                'AIMNet model did not return hidden atom features `aim`.'
            )
        return {'node_feats': out['aim'][: positions.shape[0]]}

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.atomic_numbers.device


class MiraFragModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        metadata_config: MetadataConfig,
        config: MiraFragConfig,
    ) -> None:
        super().__init__()
        self.metadata_config = metadata_config
        self.config = config
        self._sync_finetune_strategy_config()
        self.register_buffer(
            'adduct_charge_by_idx',
            torch.tensor(
                _adduct_charge_lookup(metadata_config),
                dtype=torch.get_default_dtype(),
            ),
            persistent=False,
        )
        self.encoder = self._prepare_encoder(encoder)

        self.adduct_embedding = nn.Embedding(
            metadata_config.num_adducts, config.metadata_dim
        )
        self.instrument_embedding = nn.Embedding(
            metadata_config.num_instruments, config.metadata_dim
        )
        self.head = FragmentSpectrumHead(self.config)

    def _prepare_encoder(self, encoder: nn.Module) -> nn.Module:
        strategy = self._encoder_finetune_strategy()
        if strategy == 'head':
            for param in encoder.parameters():
                param.requires_grad_(False)
            return encoder
        if strategy == 'delta':
            return TorchDeltaFineTuneWrapper(encoder)
        if strategy == 'full':
            for param in encoder.parameters():
                param.requires_grad_(True)
            return encoder
        raise ValueError(
            'Unknown encoder_finetune_strategy '
            f'{strategy!r}; expected one of: head, delta, full.'
        )

    def _sync_finetune_strategy_config(self) -> None:
        legacy_strategy = getattr(self.config, 'mace_finetune_strategy', None)
        if legacy_strategy is not None:
            self.config.encoder_finetune_strategy = str(legacy_strategy)
            delattr(self.config, 'mace_finetune_strategy')

    def _encoder_finetune_strategy(self) -> str:
        strategy = getattr(self.config, 'encoder_finetune_strategy', None)
        return str(strategy or 'head')

    def train(self, mode: bool = True) -> MiraFragModel:
        super().train(mode)
        if self._encoder_finetune_strategy() == 'head':
            self.encoder.eval()
        return self

    def metadata_features(self, batch: dict[str, Any]) -> torch.Tensor:
        precursor_mz = batch['precursor_mz'].float().unsqueeze(-1)
        precursor_mz = precursor_mz / max(self.metadata_config.precursor_mz_max, 1.0)
        collision_energy = self._normalized_collision_energy(batch)
        adduct = self.adduct_embedding(batch['adduct'].long())
        instrument = self.instrument_embedding(batch['instrument_type'].long())
        return torch.cat([precursor_mz, collision_energy, adduct, instrument], dim=-1)

    def _normalized_collision_energy(self, batch: dict[str, Any]) -> torch.Tensor:
        collision_energy = batch['collision_energy'].float()
        instrument = batch['instrument_type'].long()
        global_center = self._metadata_float('collision_energy_center', 0.0)
        global_scale = max(
            self._metadata_float(
                'collision_energy_scale',
                self._metadata_float('collision_energy_max', 100.0),
            ),
            1e-6,
        )
        center = torch.full_like(
            collision_energy,
            global_center,
        )
        scale = torch.full_like(
            collision_energy,
            global_scale,
        )

        stats_by_instrument = getattr(
            self.metadata_config,
            'collision_energy_by_instrument',
            {},
        )
        for (
            instrument_name,
            instrument_idx,
        ) in self.metadata_config.instrument_to_idx.items():
            stats = stats_by_instrument.get(instrument_name)
            if not stats:
                continue
            mask = instrument == int(instrument_idx)
            center_value = float(stats.get('center', global_center))
            scale_value = max(float(stats.get('scale', global_scale)), 1e-6)
            center = torch.where(mask, torch.full_like(center, center_value), center)
            scale = torch.where(mask, torch.full_like(scale, scale_value), scale)
        return ((collision_energy - center) / scale).unsqueeze(-1)

    def _metadata_float(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.metadata_config, name, default))
        except Exception:
            return float(default)

    def _molecular_charge(self, batch: dict[str, Any]) -> torch.Tensor:
        if 'adduct_charge' in batch:
            return batch['adduct_charge'].to(
                device=self.adduct_charge_by_idx.device,
                dtype=self.adduct_charge_by_idx.dtype,
            )
        adduct = batch['adduct'].long().to(device=self.adduct_charge_by_idx.device)
        adduct = adduct.clamp(0, self.adduct_charge_by_idx.numel() - 1)
        return self.adduct_charge_by_idx[adduct]

    def _encoder_dtype(self) -> torch.dtype:
        try:
            return next(self.encoder.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def _cast_graph_for_encoder(
        self, graph: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        dtype = self._encoder_dtype()
        return {
            key: value.to(dtype=dtype) if value.is_floating_point() else value
            for key, value in graph.items()
        }

    def encode_node_features(
        self,
        graph: dict[str, torch.Tensor],
        *,
        molecular_charge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._encode_node_features(
            graph,
            molecular_charge=molecular_charge,
        )

    def _encode_node_features(
        self,
        graph: dict[str, torch.Tensor],
        *,
        molecular_charge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph = self._cast_graph_for_encoder(graph)
        trainable_encoder = (
            self.training
            and torch.is_grad_enabled()
            and self._encoder_finetune_strategy() in {'delta', 'full'}
        )
        context = torch.enable_grad() if trainable_encoder else torch.no_grad()
        kwargs: dict[str, Any] = {}
        if self._encoder_uses_molecular_charge():
            if molecular_charge is None:
                raise ValueError(
                    'molecular_charge is required when encoding with a '
                    'charge-aware foundation encoder.'
                )
            kwargs['molecular_charge'] = molecular_charge
        with context:
            out = self.encoder(
                graph,
                training=self.training and trainable_encoder,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_node_feats=True,
                **kwargs,
            )
        return out['node_feats'].float()

    def _encoder_uses_molecular_charge(self) -> bool:
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        return bool(getattr(encoder, 'uses_molecular_charge', False))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        metadata_features = self.metadata_features(batch)
        if 'fragments' not in batch:
            raise ValueError("MiraFrag requires batch['fragments'].")
        node_feats = self._encode_node_features(
            batch['graph'],
            molecular_charge=self._molecular_charge(batch),
        )
        return self.head(node_feats, batch['fragments'], metadata_features)

    def predict_proba(self, batch: dict[str, Any]) -> dict[str, Any]:
        pred = self.forward(batch)
        pred = pred.copy()
        fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
        pred['log_probs'] = fragment_log_probs
        pred['oos_log_probs'] = oos_log_probs
        return pred


def save_checkpoint(
    path: str | Path,
    model: MiraFragModel,
    *,
    train_config: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model,
        'mirafrag_config': asdict(model.config),
        'metadata_config': model.metadata_config.to_dict(),
        'train_config': train_config or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = 'cpu',
) -> tuple[MiraFragModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = payload['model'].to(device)
    _repair_model_compat(model)
    payload['mirafrag_config'] = asdict(model.config)
    _repair_mace_cuequivariance_config(model.encoder)
    model.eval()
    return model, payload


def set_encoder_finetune_strategy(model: MiraFragModel, strategy: str) -> None:
    """Switch a loaded model between head, delta, and full encoder adaptation."""
    if strategy not in {'head', 'delta', 'full'}:
        raise ValueError(
            f'Unknown encoder_finetune_strategy {strategy!r}; '
            'expected one of: head, delta, full.'
        )

    encoder = model.encoder
    if strategy == 'delta':
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            for param in encoder.base_module.parameters():
                param.requires_grad_(False)
            for param in encoder.delta_parameters():
                param.requires_grad_(True)
        else:
            model.encoder = TorchDeltaFineTuneWrapper(encoder)
        _repair_mace_cuequivariance_config(model.encoder)
    else:
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.merge_deltas_()
            model.encoder = encoder
        train_encoder = strategy == 'full'
        for param in model.encoder.parameters():
            param.requires_grad_(train_encoder)
        _repair_mace_cuequivariance_config(model.encoder)

    model.config.encoder_finetune_strategy = strategy


def _repair_model_compat(model: MiraFragModel) -> None:
    if 'encoder' not in model._modules and 'mace' in model._modules:
        model.encoder = model._modules.pop('mace')
    if not hasattr(model.config, 'encoder_type'):
        model.config.encoder_type = 'mace'
    defaults = MiraFragConfig(num_bins=int(model.config.num_bins))
    for field in fields(MiraFragConfig):
        if not hasattr(model.config, field.name):
            setattr(model.config, field.name, getattr(defaults, field.name))
    legacy_strategy = getattr(model.config, 'mace_finetune_strategy', None)
    if legacy_strategy is not None:
        model.config.encoder_finetune_strategy = str(legacy_strategy)
        delattr(model.config, 'mace_finetune_strategy')
    if not hasattr(model, 'adduct_charge_by_idx'):
        model.register_buffer(
            'adduct_charge_by_idx',
            torch.tensor(
                _adduct_charge_lookup(model.metadata_config),
                dtype=torch.get_default_dtype(),
            ),
            persistent=False,
        )


def _repair_mace_cuequivariance_config(mace: nn.Module) -> None:
    """Restore cuequivariance flags that are not stable across pickling.

    Some MACE torch product blocks use cuequivariance symmetric contractions.
    After torch.save/torch.load, those blocks can retain the cuequivariance
    contraction module while losing the config flag that converts one-hot
    element attributes into 1D element indices. Without that flag, evaluation
    passes a 2D node-attribute matrix where cuequivariance expects a 1D index.
    """
    for module in mace.modules():
        if not hasattr(module, 'symmetric_contractions'):
            continue
        if not hasattr(module, 'cueq_config'):
            continue
        if getattr(module, 'cueq_config', None) is not None:
            continue
        symmetric_contractions = getattr(module, 'symmetric_contractions')
        module_name = type(symmetric_contractions).__module__
        if not module_name.startswith('cuequivariance_torch.'):
            continue
        module.cueq_config = SimpleNamespace(
            enabled=True,
            optimize_all=False,
            optimize_symmetric=True,
            layout_str='mul_ir',
        )


class FragmentSpectrumHead(nn.Module):
    def __init__(self, config: MiraFragConfig) -> None:
        super().__init__()
        self.num_bins = int(config.num_bins)
        self.fragment_gnn_layers = nn.ModuleList(
            [
                FragmentGraphMessageLayer(
                    hidden_dim=config.hidden_dim,
                    edge_dim=FRAGMENT_EDGE_FEATURE_DIM,
                    dropout=config.dropout,
                )
                for _ in range(max(0, int(config.fragment_gnn_layers)))
            ]
        )
        self.fragment_encoder = nn.Sequential(
            nn.LazyLinear(config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        layers: list[nn.Module] = []
        for _ in range(max(0, config.num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.scorer = nn.Sequential(*layers)
        self.oos_scorer = self._make_oos_scorer(config)

    @staticmethod
    def _make_oos_scorer(config: MiraFragConfig) -> nn.Module:
        metadata_feature_dim = 2 + 2 * int(config.metadata_dim)
        return nn.Sequential(
            nn.Linear(metadata_feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        fragments: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
    ) -> dict[str, Any]:
        fragment_batch = fragments['batch'].to(device=node_feats.device).long()
        batch_size = int(metadata_features.shape[0])
        if fragment_batch.numel() == 0:
            return {
                'kind': 'sparse',
                'logits': node_feats.new_empty(0),
                'oos_logits': self.oos_scorer(metadata_features).squeeze(-1),
                'mzs': node_feats.new_empty(0),
                'bins': torch.empty(0, dtype=torch.long, device=node_feats.device),
                'log_prior': node_feats.new_empty(0),
                'formula_index': torch.empty(
                    0,
                    dtype=torch.long,
                    device=node_feats.device,
                ),
                'edge_index': torch.empty(
                    2, 0, dtype=torch.long, device=node_feats.device
                ),
                'edge_attr': node_feats.new_empty(0, FRAGMENT_EDGE_FEATURE_DIM),
                'formula_batch': torch.empty(
                    0, dtype=torch.long, device=node_feats.device
                ),
                'batch': torch.empty(0, dtype=torch.long, device=node_feats.device),
                'batch_size': batch_size,
                'num_bins': self.num_bins,
            }

        formula_batch = fragments['formula_batch'].to(device=node_feats.device).long()
        fragment_atom_features = self._pool_fragment_atoms(
            node_feats,
            fragments['atom_index'].to(device=node_feats.device).long(),
            fragments['atom_ptr'].to(device=node_feats.device).long(),
        )
        features = torch.cat(
            [
                fragment_atom_features,
                fragments['features'].to(
                    device=node_feats.device,
                    dtype=node_feats.dtype,
                ),
                metadata_features[formula_batch],
            ],
            dim=-1,
        )
        formula_features = self.fragment_encoder(features)
        edge_index = fragments['edge_index'].to(device=node_feats.device)
        edge_attr = fragments['edge_attr'].to(
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        for layer in self.fragment_gnn_layers:
            formula_features = layer(formula_features, edge_index, edge_attr)
        formula_index = fragments['formula_index'].to(device=node_feats.device).long()
        formula_logits = self.scorer(formula_features).squeeze(-1)
        logits = formula_logits[formula_index]
        log_prior = fragments['log_prior'].to(
            device=node_feats.device,
            dtype=logits.dtype,
        )
        logits = logits + log_prior
        return {
            'kind': 'sparse',
            'logits': logits,
            'oos_logits': self.oos_scorer(metadata_features).squeeze(-1),
            'mzs': fragments['mz'].to(device=node_feats.device, dtype=node_feats.dtype),
            'bins': fragments['bin']
            .to(device=node_feats.device)
            .long()
            .clamp(0, self.num_bins - 1),
            'log_prior': log_prior,
            'formula_index': formula_index,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'formula_batch': formula_batch,
            'batch': fragment_batch,
            'batch_size': batch_size,
            'num_bins': self.num_bins,
        }

    @staticmethod
    def _pool_fragment_atoms(
        node_feats: torch.Tensor,
        atom_index: torch.Tensor,
        atom_ptr: torch.Tensor,
    ) -> torch.Tensor:
        num_fragments = max(int(atom_ptr.numel()) - 1, 0)
        pooled = node_feats.new_zeros(num_fragments, node_feats.shape[-1])
        if num_fragments == 0:
            return pooled

        counts = (atom_ptr[1:] - atom_ptr[:-1]).clamp_min(1)
        fragment_idx = torch.repeat_interleave(
            torch.arange(num_fragments, device=node_feats.device),
            counts.to(device=node_feats.device),
        )
        pooled.index_add_(0, fragment_idx, node_feats[atom_index])
        return pooled / counts.to(
            dtype=node_feats.dtype, device=node_feats.device
        ).unsqueeze(-1)


class FragmentGraphMessageLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.source = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_features

        src = edge_index[0].long()
        dst = edge_index[1].long()
        messages = self.source(node_features[src]) * torch.sigmoid(
            self.edge_gate(edge_attr)
        )
        aggregated = node_features.new_zeros(node_features.shape)
        aggregated.index_add_(0, dst, messages)
        degree = node_features.new_zeros(node_features.shape[0], 1)
        degree.index_add_(0, dst, node_features.new_ones(dst.shape[0], 1))
        aggregated = aggregated / degree.clamp_min(1.0)

        hidden = self.message_norm(node_features + self.dropout(aggregated))
        return self.update_norm(hidden + self.dropout(self.update(hidden)))
