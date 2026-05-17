from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mirafrag.config import mirafrag_config_from_dict
from mirafrag.data import MetadataConfig
from mirafrag.encoders import load_foundation_encoder
from mirafrag.encoders.mace import repair_mace_cuequivariance_config
from mirafrag.model import MiraFragModel

CHECKPOINT_FORMAT = 'mirafrag-state-v1'


def save_checkpoint(
    path: str | Path,
    model: MiraFragModel,
    *,
    train_config: dict[str, Any] | None = None,
) -> None:
    """
    Write a strict MiraFrag state checkpoint.

    The checkpoint stores a state dict, serializable model config, metadata config, and training configuration. It intentionally does not pickle the model object, so loading is explicit about the foundation encoder and package code in use.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'checkpoint_format': CHECKPOINT_FORMAT,
        'model_state_dict': model.state_dict(),
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
    """
    Load a MiraFrag checkpoint and reconstruct the model.

    The loader validates the checkpoint format, rebuilds the configured foundation encoder, instantiates :class:`MiraFragModel`, and loads the saved state dict. The returned model is moved to ``device`` and placed in evaluation mode.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    _validate_checkpoint_payload(payload)
    model = _load_state_checkpoint_model(payload, device=device)
    payload['mirafrag_config'] = asdict(model.config)
    repair_mace_cuequivariance_config(model.encoder)
    model.eval()
    return model, payload


def _validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    """
    Validate the top-level checkpoint schema.

    This catches old pickle-style checkpoints or partially written files before any model construction starts. A clear error is raised when required keys or the expected format marker are missing.
    """
    required = {
        'checkpoint_format',
        'model_state_dict',
        'mirafrag_config',
        'metadata_config',
        'train_config',
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f'Invalid MiraFrag checkpoint; missing keys: {sorted(missing)}'
        )
    if payload['checkpoint_format'] != CHECKPOINT_FORMAT:
        raise ValueError(
            'Unsupported MiraFrag checkpoint format '
            f'{payload["checkpoint_format"]!r}; expected {CHECKPOINT_FORMAT!r}.'
        )


def _load_state_checkpoint_model(
    payload: dict[str, Any],
    *,
    device: str | torch.device,
) -> MiraFragModel:
    """
    Instantiate a model from a validated checkpoint payload.

    The function rebuilds the saved MiraFrag and metadata configs, loads the requested encoder family, and restores all model parameters from the state dict.
    """
    config = mirafrag_config_from_dict(payload['mirafrag_config'])
    metadata_config = MetadataConfig.from_dict(payload['metadata_config'])
    encoder = load_foundation_encoder(
        encoder_type=config.encoder_type,
        foundation_source=config.foundation_source,
        foundation_model=config.foundation_model,
        foundation_path=config.foundation_path,
        aimnet_model=config.aimnet_model,
        aimnet_path=config.aimnet_path,
        device=device,
    )
    model = MiraFragModel(encoder, metadata_config=metadata_config, config=config).to(
        device
    )
    model.load_state_dict(payload['model_state_dict'])
    return model
