from __future__ import annotations

import torch
from torch import nn

from mirafrag.encoders.aimnet import AimnetNodeEncoder, load_aimnet_encoder
from mirafrag.encoders.mace import load_mace_encoder, repair_mace_cuequivariance_config


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


__all__ = [
    'AimnetNodeEncoder',
    'load_aimnet_encoder',
    'load_foundation_encoder',
    'load_mace_encoder',
    'repair_mace_cuequivariance_config',
]
