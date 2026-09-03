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

ENCODER_FINE_TUNE_STRATEGIES = ('head', 'delta', 'full', 'aimnet-final')
AIMNET_FINAL_PARAMETER_PREFIXES = ('model.mlps.2.6.', 'mlps.2.6.')
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


class EncoderBondAdapter(nn.Module):
    """
    Residual molecular-edge adapter for encoder node features.

    The adapter uses the existing collated molecular radius graph and 3D distances
    to run a small supervised message-passing block on top of foundation encoder
    features. It is zero-gated at initialization so adding it to a checkpoint is
    initially prediction-preserving.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_radial: int = 8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError('feature_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if num_radial <= 0:
            raise ValueError('num_radial must be positive.')
        if dropout < 0:
            raise ValueError('dropout must be nonnegative.')
        self.feature_dim = int(feature_dim)
        self.num_radial = int(num_radial)
        self.node_in = nn.LazyLinear(self.feature_dim)
        self.edge_mlp = nn.Sequential(
            nn.LazyLinear(self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, self.feature_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.feature_dim, self.feature_dim),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out: nn.Linear | None = None
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        node_feats: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        edge_index = graph.get('edge_index')
        positions = graph.get('positions')
        if edge_index is None or positions is None or edge_index.numel() == 0:
            return node_feats
        edge_index = edge_index.to(device=node_feats.device, dtype=torch.long)
        positions = positions.to(device=node_feats.device, dtype=node_feats.dtype)
        src = edge_index[0]
        dst = edge_index[1]
        h = self.node_in(node_feats)
        radial = self._distance_basis(positions[src], positions[dst])
        edge_features = torch.cat([h[src], h[dst], radial], dim=-1)
        edge_gate = self.edge_mlp(edge_features)
        degree = node_feats.new_zeros(node_feats.shape[0], 1)
        degree.index_add_(0, dst, node_feats.new_ones(dst.shape[0], 1))
        norm = degree.clamp_min(1.0).reciprocal()
        for layer in self.layers:
            messages = edge_gate * h[src]
            aggregated = h.new_zeros(h.shape)
            aggregated.index_add_(0, dst, messages)
            h = h + layer(aggregated * norm)
        return node_feats + self.residual_scale * self._output_projection(
            h, node_feats.shape[-1]
        )

    def _output_projection(
        self,
        hidden: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        if self.out is None:
            self.out = nn.Linear(self.feature_dim, int(output_dim)).to(
                device=hidden.device, dtype=hidden.dtype
            )
        return self.out(hidden)

    def _distance_basis(
        self,
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        distances = (source_positions - target_positions).norm(dim=-1)
        max_distance = distances.detach().max().clamp_min(1.0)
        centers = torch.linspace(
            0.0,
            1.0,
            self.num_radial,
            dtype=distances.dtype,
            device=distances.device,
        )
        scaled = (distances / max_distance).unsqueeze(-1)
        gamma = float(max(self.num_radial - 1, 1)) ** 2
        return torch.exp(-gamma * (scaled - centers) ** 2)


class LazyZeroOutputProjection(nn.Module):
    """
    Lazily created zero-initialized projection with checkpoint reload support.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.linear: nn.Linear | None = None

    def forward(self, hidden: torch.Tensor, output_dim: int) -> torch.Tensor:
        if self.linear is None:
            self.linear = nn.Linear(self.feature_dim, int(output_dim)).to(
                device=hidden.device, dtype=hidden.dtype
            )
            nn.init.zeros_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        return self.linear(hidden)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        weight = state_dict.get(prefix + 'linear.weight')
        if self.linear is None and weight is not None:
            self.linear = nn.Linear(self.feature_dim, int(weight.shape[0])).to(
                device=weight.device, dtype=weight.dtype
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MetadataConditionedEncoderAdapter(nn.Module):
    """
    Residual metadata-conditioned message passing over encoder atom states.

    The adapter lets collision energy, adduct, and instrument metadata modulate
    atom features before fragment scoring. Its output projection is initialized
    to zero, so enabling it on a checkpoint preserves predictions until the new
    branch is trained.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_radial: int = 8,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError('feature_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if dropout < 0:
            raise ValueError('dropout must be nonnegative.')
        if num_radial <= 0:
            raise ValueError('num_radial must be positive.')
        self.feature_dim = int(feature_dim)
        self.num_radial = int(num_radial)
        self.node_in = nn.LazyLinear(self.feature_dim)
        self.metadata_in = nn.LazyLinear(self.feature_dim)
        self.edge_mlp = nn.Sequential(
            nn.LazyLinear(self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, self.feature_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.feature_dim, self.feature_dim),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out = LazyZeroOutputProjection(self.feature_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        graph: dict[str, torch.Tensor],
        metadata_features: torch.Tensor,
    ) -> torch.Tensor:
        graph_batch = graph.get('batch')
        if graph_batch is None:
            if metadata_features.shape[0] != 1:
                raise ValueError(
                    'graph batch indices are required for metadata-conditioned '
                    'encoder adaptation with batched metadata.'
                )
            graph_batch = torch.zeros(
                node_feats.shape[0], dtype=torch.long, device=node_feats.device
            )
        else:
            graph_batch = graph_batch.to(device=node_feats.device, dtype=torch.long)
        metadata_features = metadata_features.to(
            device=node_feats.device, dtype=node_feats.dtype
        )
        if graph_batch.numel() != node_feats.shape[0]:
            raise ValueError('graph batch indices must align with node_feats.')
        if (
            graph_batch.numel() > 0
            and int(graph_batch.max().item()) >= metadata_features.shape[0]
        ):
            raise ValueError('graph batch index exceeds metadata batch size.')

        node_metadata = metadata_features[graph_batch]
        h = self.node_in(node_feats) + self.metadata_in(node_metadata)
        edge_index = graph.get('edge_index')
        positions = graph.get('positions')
        if edge_index is None or positions is None or edge_index.numel() == 0:
            hidden = h
        else:
            edge_index = edge_index.to(device=node_feats.device, dtype=torch.long)
            positions = positions.to(device=node_feats.device, dtype=node_feats.dtype)
            src = edge_index[0]
            dst = edge_index[1]
            radial = self._distance_basis(positions[src], positions[dst])
            edge_features = torch.cat(
                [h[src], h[dst], node_metadata[dst], radial], dim=-1
            )
            edge_gate = self.edge_mlp(edge_features)
            degree = node_feats.new_zeros(node_feats.shape[0], 1)
            degree.index_add_(0, dst, node_feats.new_ones(dst.shape[0], 1))
            norm = degree.clamp_min(1.0).reciprocal()
            hidden = h
            for layer in self.layers:
                messages = edge_gate * hidden[src]
                aggregated = hidden.new_zeros(hidden.shape)
                aggregated.index_add_(0, dst, messages)
                hidden = hidden + layer(aggregated * norm)
        return node_feats + self._output_projection(hidden, node_feats.shape[-1])

    def _output_projection(
        self,
        hidden: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        return self.out(hidden, output_dim)

    def _distance_basis(
        self,
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        distances = (source_positions - target_positions).norm(dim=-1)
        max_distance = distances.detach().max().clamp_min(1.0)
        centers = torch.linspace(
            0.0,
            1.0,
            self.num_radial,
            dtype=distances.dtype,
            device=distances.device,
        )
        scaled = (distances / max_distance).unsqueeze(-1)
        gamma = float(max(self.num_radial - 1, 1)) ** 2
        return torch.exp(-gamma * (scaled - centers) ** 2)


class AimnetMultipassAdapter(nn.Module):
    """
    Zero-gated adapter that fuses intermediate AIMNet atom states into node features.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError('feature_dim must be positive.')
        if dropout < 0:
            raise ValueError('dropout must be nonnegative.')
        self.feature_dim = int(feature_dim)
        self.input_dropout = nn.Dropout(float(dropout))
        self.encoder = nn.Sequential(
            nn.LazyLinear(self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
        )
        self.out: nn.Linear | None = (
            nn.Linear(self.feature_dim, int(output_dim))
            if output_dim is not None
            else None
        )
        self._reset_output_projection()

    def forward(
        self,
        node_feats: torch.Tensor,
        aux_node_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        features = aux_node_features.get('aimnet_multipass_features')
        if features is None:
            raise ValueError(
                'AIMNet multipass features are enabled, but the encoder did not '
                'return `aimnet_multipass_features`.'
            )
        features = features.to(device=node_feats.device, dtype=node_feats.dtype)
        if features.ndim != 2 or features.shape[0] != node_feats.shape[0]:
            raise ValueError(
                'AIMNet multipass features must have shape [num_atoms, dim] and '
                'align with node_feats.'
            )
        hidden = self.encoder(self.input_dropout(features))
        return node_feats + self._output_projection(hidden, node_feats.shape[-1])

    def _output_projection(
        self,
        hidden: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        if self.out is None:
            self.out = nn.Linear(self.feature_dim, int(output_dim)).to(
                device=hidden.device, dtype=hidden.dtype
            )
            self._reset_output_projection()
        return self.out(hidden)

    def _reset_output_projection(self) -> None:
        """
        Initialize the adapter as an exact no-op while keeping output weights trainable.
        """
        if self.out is None:
            return
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)


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
        self._configure_encoder_auxiliary_outputs()
        self.aimnet_multipass_adapter = self._build_aimnet_multipass_adapter()
        self.encoder_bond_adapter = self._build_encoder_bond_adapter()
        self.encoder_metadata_adapter = self._build_encoder_metadata_adapter()
        self.retrieval_calibration_head = self._build_retrieval_calibration_head()
        self.adduct_embedding = nn.Embedding(
            metadata_config.num_adducts, config.metadata_dim
        )
        self.instrument_embedding = nn.Embedding(
            metadata_config.num_instruments, config.metadata_dim
        )
        self.metadata_ce_interaction = self._build_metadata_ce_interaction()
        self.head = FragmentSpectrumHead(self.config)

    def _configure_encoder_auxiliary_outputs(self) -> None:
        """
        Request optional auxiliary tensors from encoders that can provide them.
        """
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        if hasattr(encoder, 'export_multipass_features'):
            setattr(
                encoder,
                'export_multipass_features',
                bool(getattr(self.config, 'aimnet_multipass_features', False)),
            )

    def _build_aimnet_multipass_adapter(self) -> AimnetMultipassAdapter | None:
        """
        Build the optional adapter over AIMNet intermediate atom states.
        """
        if not bool(getattr(self.config, 'aimnet_multipass_features', False)):
            return None
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        output_dim = getattr(encoder, 'node_feature_dim', None)
        if output_dim is None:
            raise ValueError(
                'aimnet_multipass_features requires the encoder to expose '
                'node_feature_dim so the adapter output projection can be '
                'registered before checkpoint loading.'
            )
        return AimnetMultipassAdapter(
            feature_dim=int(self.config.hidden_dim),
            output_dim=int(output_dim),
            dropout=float(self.config.dropout),
        )

    def _build_encoder_bond_adapter(self) -> EncoderBondAdapter | None:
        """
        Build the optional post-encoder molecular bond/radius-graph adapter.
        """
        layers = int(getattr(self.config, 'encoder_bond_adapter_layers', 0) or 0)
        if layers <= 0:
            return None
        return EncoderBondAdapter(
            feature_dim=int(
                getattr(self.config, 'encoder_bond_adapter_feature_dim', 128) or 128
            ),
            num_layers=layers,
            dropout=float(
                getattr(self.config, 'encoder_bond_adapter_dropout', 0.0) or 0.0
            ),
        )

    def _build_encoder_metadata_adapter(
        self,
    ) -> MetadataConditionedEncoderAdapter | None:
        """
        Build the optional CE/adduct/instrument-conditioned node adapter.
        """
        layers = int(getattr(self.config, 'encoder_metadata_adapter_layers', 0) or 0)
        if layers <= 0:
            return None
        return MetadataConditionedEncoderAdapter(
            feature_dim=int(
                getattr(self.config, 'encoder_metadata_adapter_feature_dim', 128) or 128
            ),
            num_layers=layers,
            dropout=float(
                getattr(self.config, 'encoder_metadata_adapter_dropout', 0.0) or 0.0
            ),
        )

    def _build_retrieval_calibration_head(self) -> nn.Module | None:
        """
        Build an optional candidate-level retrieval calibration head.

        The final projection starts at zero, so enabling the head is initially a
        no-op for retrieval scores and checkpoint evaluation.
        """
        if not bool(getattr(self.config, 'retrieval_calibration_head', False)):
            return None
        hidden_dim = int(self.config.hidden_dim)
        head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(hidden_dim, 1),
        )
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError('retrieval calibration final layer must be Linear.')
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        return head

    def _build_metadata_ce_interaction(self) -> nn.Linear | None:
        """
        Build a zero-initialized CE-by-instrument residual for metadata features.

        The output is added to the instrument embedding instead of widening the
        metadata vector, which keeps old spectrum-head weights shape-compatible.
        """
        if not bool(getattr(self.config, 'metadata_ce_interaction', False)):
            return None
        layer = nn.Linear(int(self.config.metadata_dim), int(self.config.metadata_dim))
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

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
        if strategy == 'delta':
            return TorchDeltaFineTuneWrapper(encoder)
        if strategy == 'full':
            for param in encoder.parameters():
                param.requires_grad_(True)
            return encoder
        if strategy == 'aimnet-final':
            _set_aimnet_final_requires_grad(encoder)
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
        if self.metadata_ce_interaction is not None:
            instrument = instrument + self.metadata_ce_interaction(
                instrument * collision_energy
            )
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

        Frozen encoders are evaluated under ``no_grad``. Delta, full, and aimnet-final fine-tuning keep gradients enabled during training, and charge-aware encoders receive molecular charges.
        """
        graph = self._cast_graph_for_encoder(graph)
        trainable_encoder = (
            self.training
            and torch.is_grad_enabled()
            and self._encoder_finetune_strategy() in {'delta', 'full', 'aimnet-final'}
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
        for key in ('aimnet_charge_features', 'aimnet_multipass_features'):
            if key in result:
                result[key] = result[key].float()
        if self.aimnet_multipass_adapter is not None:
            node_feats = self.aimnet_multipass_adapter(node_feats, result).float()
        if self.encoder_bond_adapter is not None:
            node_feats = self.encoder_bond_adapter(node_feats, graph).float()
        if self.encoder_metadata_adapter is not None:
            if metadata_features is None:
                raise ValueError(
                    'metadata_features are required when encoder_metadata_adapter_layers > 0.'
                )
            node_feats = self.encoder_metadata_adapter(
                node_feats, graph, metadata_features
            ).float()
        result['node_feats'] = node_feats
        return result

    def _encoder_uses_molecular_charge(self) -> bool:
        """
        Return whether the wrapped encoder expects molecular charge inputs.
        """
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        return bool(getattr(encoder, 'uses_molecular_charge', False))

    def _encoder_uses_smiles(self) -> bool:
        """
        Return whether the wrapped encoder expects batch SMILES strings.
        """
        encoder = self.encoder
        if isinstance(encoder, TorchDeltaFineTuneWrapper):
            encoder = encoder.base_module
        return bool(getattr(encoder, 'uses_smiles', False))

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
        aux_node_features = {
            key: value for key, value in encoder_out.items() if key != 'node_feats'
        }
        pred = self.head(
            node_feats,
            batch['fragments'],
            metadata_features,
            graph_batch=batch['graph'].get('batch'),
            aux_node_features=aux_node_features,
            molecule_descriptors=batch.get('molecule_descriptors'),
            graph=batch['graph'],
        )
        if self.retrieval_calibration_head is not None:
            pred = pred.copy()
            pred['retrieval_logit'] = self.retrieval_calibration_head(
                torch.cat(
                    [
                        metadata_features,
                        self._molecule_node_summary(
                            node_feats,
                            batch['graph'].get('batch'),
                            batch_size=int(pred['batch_size']),
                        ),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
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


def _set_aimnet_final_requires_grad(encoder: nn.Module) -> None:
    """
    Train only AIMNet2's final atom-feature projection used as MiraFrag node features.
    """
    trainable_names: list[str] = []
    for name, param in encoder.named_parameters():
        trainable = name.startswith(AIMNET_FINAL_PARAMETER_PREFIXES)
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(
            'encoder_finetune_strategy=aimnet-final requires an AIMNet2-like '
            'encoder exposing parameters under model.mlps.2.6.*.'
        )


def set_encoder_finetune_strategy(model: MiraFragModel, strategy: str) -> None:
    """
    Switch the encoder adaptation strategy of an existing model.

    Delta mode wraps the encoder in additive delta parameters, full mode trains base weights, aimnet-final trains only AIMNet2 model.mlps.2.6, and head mode freezes encoder weights. Existing delta wrappers are merged when leaving delta mode.
    """
    if strategy not in ENCODER_FINE_TUNE_STRATEGIES:
        raise ValueError(
            f'Unknown encoder_finetune_strategy {strategy!r}; '
            f'expected one of: {", ".join(ENCODER_FINE_TUNE_STRATEGIES)}.'
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
        if strategy == 'full':
            for param in model.encoder.parameters():
                param.requires_grad_(True)
        elif strategy == 'aimnet-final':
            _set_aimnet_final_requires_grad(model.encoder)
        else:
            for param in model.encoder.parameters():
                param.requires_grad_(False)
        repair_mace_cuequivariance_config(model.encoder)

    model.config.encoder_finetune_strategy = strategy
    model._configure_encoder_auxiliary_outputs()
