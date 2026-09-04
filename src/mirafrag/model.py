from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mirafrag.adducts import parse_adduct
from mirafrag.config import MiraFragConfig
from mirafrag.data import MetadataConfig
from mirafrag.heads.fragment import FragmentSpectrumHead
from mirafrag.probability import fragment_oos_log_probs

ENCODER_FINE_TUNE_STRATEGIES = ('head', 'full')
ENCODER_FINE_TUNE_STRATEGY_NAMES = ', '.join(ENCODER_FINE_TUNE_STRATEGIES)


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

    The model wraps a molecular atom encoder, builds precursor metadata features, and applies a candidate-based fragment spectrum head. Encoder adaptation is controlled by head-only or full fine-tuning.
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

    @staticmethod
    def _molecule_node_summary(
        node_feats: torch.Tensor,
        graph_batch: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Mean-pool atom features into one vector per candidate molecule."""
        if graph_batch is None:
            if batch_size != 1:
                raise ValueError('graph batch indices are required for batched graphs.')
            return node_feats.mean(dim=0, keepdim=True)
        graph_batch = graph_batch.to(device=node_feats.device, dtype=torch.long)
        pooled = node_feats.new_zeros((batch_size, int(node_feats.shape[-1])))
        counts = node_feats.new_zeros((batch_size, 1))
        pooled.index_add_(0, graph_batch, node_feats)
        counts.index_add_(
            0,
            graph_batch,
            torch.ones(
                (node_feats.shape[0], 1),
                device=node_feats.device,
                dtype=node_feats.dtype,
            ),
        )
        return pooled / counts.clamp_min(1.0)

    def _prepare_encoder(self, encoder: nn.Module) -> nn.Module:
        """
        Apply the configured fine-tuning strategy to the foundation encoder.
        """
        strategy = self._encoder_finetune_strategy()
        if strategy == 'head':
            for param in encoder.parameters():
                param.requires_grad_(False)
            return encoder
        if strategy == 'full':
            for param in encoder.parameters():
                param.requires_grad_(True)
            return encoder
        raise ValueError(
            'Unknown encoder_finetune_strategy '
            f'{strategy!r}; expected one of: '
            f'{ENCODER_FINE_TUNE_STRATEGY_NAMES}.'
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

        Merged spectra may carry multiple collision-energy values. When present,
        each value is normalized with the sample instrument statistics and then
        mean-pooled back to one metadata feature per sample. The scalar
        collision_energy path is kept for non-merged callers and old tests.
        """
        collision_energy = batch['collision_energy'].float()
        batch_size = int(collision_energy.shape[0])
        if 'collision_energy_values' in batch and 'collision_energy_batch' in batch:
            ce_values = batch['collision_energy_values'].float()
            ce_batch = batch['collision_energy_batch'].long()
            if ce_values.numel() > 0 and ce_batch.numel() == ce_values.numel():
                instrument = batch['instrument_type'].long()[ce_batch]
                normalized = self._normalize_collision_energy_values(
                    ce_values,
                    instrument,
                )
                pooled = collision_energy.new_zeros(batch_size)
                counts = collision_energy.new_zeros(batch_size)
                safe_batch = ce_batch.clamp(0, max(batch_size - 1, 0))
                pooled.index_add_(0, safe_batch, normalized)
                counts.index_add_(0, safe_batch, torch.ones_like(normalized))
                return (pooled / counts.clamp_min(1.0)).unsqueeze(-1)
        instrument = batch['instrument_type'].long()
        return self._normalize_collision_energy_values(
            collision_energy,
            instrument,
        ).unsqueeze(-1)

    def _normalize_collision_energy_values(
        self,
        collision_energy: torch.Tensor,
        instrument: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize one flat CE tensor, preserving normalized-mode values."""
        mode = str(
            getattr(self.metadata_config, 'collision_energy_mode', 'raw') or 'raw'
        ).lower()
        if mode == 'normalized':
            return collision_energy
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
        return (collision_energy - center) / scale

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
        smiles: list[str] | None = None,
        metadata_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Public wrapper for obtaining encoder node features from a graph.
        """
        out = self._encode_node_outputs(
            graph,
            molecular_charge=molecular_charge,
            smiles=smiles,
            metadata_features=metadata_features,
        )
        return out['node_feats']

    def _encode_node_features(
        self,
        graph: dict[str, torch.Tensor],
        *,
        molecular_charge: torch.Tensor | None = None,
        smiles: list[str] | None = None,
        metadata_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Backward-compatible private wrapper for encoder node features.
        """
        return self._encode_node_outputs(
            graph,
            molecular_charge=molecular_charge,
            smiles=smiles,
            metadata_features=metadata_features,
        )['node_feats']

    def _encode_node_outputs(
        self,
        graph: dict[str, torch.Tensor],
        *,
        molecular_charge: torch.Tensor | None = None,
        smiles: list[str] | None = None,
        metadata_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run the foundation encoder with correct grad and charge handling.

        Frozen encoders are evaluated under ``no_grad``. Full fine-tuning keeps gradients enabled during training, and charge-aware encoders receive molecular charges.
        """
        graph = self._cast_graph_for_encoder(graph)
        trainable_encoder = (
            self.training
            and torch.is_grad_enabled()
            and self._encoder_finetune_strategy() == 'full'
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
        if self._encoder_uses_smiles():
            if smiles is None:
                raise ValueError(
                    'smiles are required when encoding with a SMILES-aware '
                    'foundation encoder.'
                )
            kwargs['smiles'] = smiles
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
        result = dict(out)
        node_feats = out['node_feats'].float()
        result['node_feats'] = node_feats
        return result

    def _encoder_uses_molecular_charge(self) -> bool:
        """
        Return whether the wrapped encoder expects molecular charge inputs.
        """
        return bool(getattr(self.encoder, 'uses_molecular_charge', False))

    def _encoder_uses_smiles(self) -> bool:
        """
        Return whether the wrapped encoder expects batch SMILES strings.
        """
        return bool(getattr(self.encoder, 'uses_smiles', False))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Predict sparse fragment spectrum logits for a collated batch.

        The batch must include graph tensors, metadata tensors, and fragment candidate tensors. The return value is a sparse prediction dictionary consumed by losses and evaluation.
        """
        metadata_features = self.metadata_features(batch)
        if 'fragments' not in batch:
            raise ValueError("MiraFrag requires batch['fragments'].")
        encoder_out = self._encode_node_outputs(
            batch['graph'],
            molecular_charge=self._molecular_charge(batch),
            smiles=batch.get('smiles'),
            metadata_features=metadata_features,
        )
        node_feats = encoder_out['node_feats']
        pred = self.head(
            node_feats,
            batch['fragments'],
            metadata_features,
            graph_batch=batch['graph'].get('batch'),
            graph=batch['graph'],
        )
        return pred

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
    """Switch an existing model between head-only and full encoder fine-tuning."""
    if strategy not in ENCODER_FINE_TUNE_STRATEGIES:
        raise ValueError(
            f'Unknown encoder_finetune_strategy {strategy!r}; '
            f'expected one of: {", ".join(ENCODER_FINE_TUNE_STRATEGIES)}.'
        )
    train_encoder = strategy == 'full'
    for param in model.encoder.parameters():
        param.requires_grad_(train_encoder)
    model.config.encoder_finetune_strategy = strategy
