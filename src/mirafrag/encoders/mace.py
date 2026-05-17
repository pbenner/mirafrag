from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


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
    mace_model = result.model.to(device)
    repair_mace_cuequivariance_config(mace_model)
    mace_model.eval()
    return mace_model


def repair_mace_cuequivariance_config(module: nn.Module) -> None:
    for child in module.modules():
        if not hasattr(child, 'symmetric_contractions'):
            continue
        if not hasattr(child, 'cueq_config'):
            continue
        if getattr(child, 'cueq_config', None) is not None:
            continue
        symmetric_contractions = getattr(child, 'symmetric_contractions')
        module_name = type(symmetric_contractions).__module__
        if not module_name.startswith('cuequivariance_torch.'):
            continue
        child.cueq_config = SimpleNamespace(
            enabled=True,
            optimize_all=False,
            optimize_symmetric=True,
            layout_str='mul_ir',
        )
