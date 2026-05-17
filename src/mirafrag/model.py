from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mirafrag.adducts import parse_adduct
from mirafrag.config import MiraFragConfig
from mirafrag.data import MetadataConfig
from mirafrag.delta import TorchDeltaFineTuneWrapper
from mirafrag.encoders.mace import repair_mace_cuequivariance_config
from mirafrag.heads.fragment import FragmentSpectrumHead
from mirafrag.probability import fragment_oos_log_probs


def _adduct_charge_lookup(metadata_config: MetadataConfig) -> list[float]:
    """
    Build a lookup table from metadata adduct index to ion charge.
    """
    size = max(metadata_config.adduct_to_idx.values(), default=-1) + 2
    charges = [0.0] * max(size, 1)
    for adduct, idx in metadata_config.adduct_to_idx.items():
        if 0 <= int(idx) < len(charges):
            charges[int(idx)] = float(parse_adduct(adduct).charge)
    return charges


class MiraFragModel(nn.Module):
    """
    Foundation-encoder model for sparse MS/MS spectrum prediction.

    The model wraps a MACE or AIMNet atom encoder, builds precursor metadata features, and applies a candidate-based fragment spectrum head. Encoder adaptation is controlled by head-only, delta, or full fine-tuning strategy.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        metadata_config: MetadataConfig,
        config: MiraFragConfig,
    ) -> None:
        """
        Initialize encoder, metadata embeddings, adduct-charge lookup, and fragment head.
        """
        super().__init__()
        self.metadata_config = metadata_config
        self.config = config
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
        """
        Apply the configured fine-tuning strategy to the foundation encoder.
        """
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

    def _encoder_finetune_strategy(self) -> str:
        """
        Return the active encoder fine-tuning strategy with a safe default.
        """
        return str(self.config.encoder_finetune_strategy or 'head')

    def train(self, mode: bool = True) -> MiraFragModel:
        """
        Set module training mode while keeping frozen encoders in eval mode.

        Head-only fine-tuning should not update encoder state such as dropout or normalization behavior, so the encoder is forced back to evaluation mode when frozen.
        """
        super().train(mode)
        if self._encoder_finetune_strategy() == 'head':
            self.encoder.eval()
        return self

    def metadata_features(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Build dense precursor metadata features for a batch.

        The vector concatenates scaled precursor m/z, normalized collision energy, adduct embedding, and instrument embedding.
        """
        precursor_mz = batch['precursor_mz'].float().unsqueeze(-1)
        precursor_mz = precursor_mz / max(self.metadata_config.precursor_mz_max, 1.0)
        collision_energy = self._normalized_collision_energy(batch)
        adduct = self.adduct_embedding(batch['adduct'].long())
        instrument = self.instrument_embedding(batch['instrument_type'].long())
        return torch.cat([precursor_mz, collision_energy, adduct, instrument], dim=-1)

    def _normalized_collision_energy(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Normalize collision energy using global and instrument-specific robust statistics.
        """
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
        center = torch.full_like(collision_energy, global_center)
        scale = torch.full_like(collision_energy, global_scale)
        stats_by_instrument = self.metadata_config.collision_energy_by_instrument
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
        """
        Read a float attribute from metadata config with a robust fallback.
        """
        try:
            return float(getattr(self.metadata_config, name, default))
        except Exception:
            return float(default)

    def _molecular_charge(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Return one molecular ion charge per batch item.

        The value comes from the collated ``adduct_charge`` tensor when available, otherwise it is looked up from the adduct categorical index.
        """
        if 'adduct_charge' in batch:
            return batch['adduct_charge'].to(
                device=self.adduct_charge_by_idx.device,
                dtype=self.adduct_charge_by_idx.dtype,
            )
        adduct = batch['adduct'].long().to(device=self.adduct_charge_by_idx.device)
        adduct = adduct.clamp(0, self.adduct_charge_by_idx.numel() - 1)
        return self.adduct_charge_by_idx[adduct]

    def _encoder_dtype(self) -> torch.dtype:
        """
        Return the floating dtype used by encoder parameters.
        """
        try:
            return next(self.encoder.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def _cast_graph_for_encoder(
        self, graph: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Cast floating graph tensors to the encoder dtype while preserving integer tensors.
        """
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
        """
        Public wrapper for obtaining encoder node features from a graph.
        """
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
        """
        Run the foundation encoder with correct grad and charge handling.

        Frozen encoders are evaluated under ``no_grad``. Delta and full fine-tuning keep gradients enabled during training, and charge-aware encoders receive molecular charges.
        """
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
        """
        Return whether the wrapped encoder expects molecular charge inputs.
        """
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        return bool(getattr(encoder, 'uses_molecular_charge', False))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Predict sparse fragment spectrum logits for a collated batch.

        The batch must include graph tensors, metadata tensors, and fragment candidate tensors. The return value is a sparse prediction dictionary consumed by losses and evaluation.
        """
        metadata_features = self.metadata_features(batch)
        if 'fragments' not in batch:
            raise ValueError("MiraFrag requires batch['fragments'].")
        node_feats = self._encode_node_features(
            batch['graph'],
            molecular_charge=self._molecular_charge(batch),
        )
        return self.head(node_feats, batch['fragments'], metadata_features)

    def predict_proba(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Return sparse prediction log-probabilities including OOS probability.
        """
        pred = self.forward(batch)
        pred = pred.copy()
        fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
        pred['log_probs'] = fragment_log_probs
        pred['oos_log_probs'] = oos_log_probs
        return pred


def set_encoder_finetune_strategy(model: MiraFragModel, strategy: str) -> None:
    """
    Switch the encoder adaptation strategy of an existing model.

    Delta mode wraps the encoder in additive delta parameters, full mode trains base weights, and head mode freezes encoder weights. Existing delta wrappers are merged when leaving delta mode.
    """
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
        repair_mace_cuequivariance_config(model.encoder)
    else:
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.merge_deltas_()
            model.encoder = encoder
        train_encoder = strategy == 'full'
        for param in model.encoder.parameters():
            param.requires_grad_(train_encoder)
        repair_mace_cuequivariance_config(model.encoder)

    model.config.encoder_finetune_strategy = strategy
