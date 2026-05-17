from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.parameter import UninitializedParameter

from mirafrag.model import MiraFragModel


def _optimizer_param_groups(
    model: MiraFragModel,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """
    Build AdamW parameter groups for head and trainable encoder parameters.

    The spectrum head always has zero weight decay. Encoder or delta parameters use the configured weight decay, and uninitialized lazy parameters are skipped until materialized.
    """
    encoder_params = [
        param
        for param in model.encoder.parameters()
        if param.requires_grad and not _is_uninitialized_parameter(param)
    ]
    encoder_param_ids = {id(param) for param in encoder_params}
    head_params = [
        param
        for param in model.parameters()
        if (
            param.requires_grad
            and id(param) not in encoder_param_ids
            and not _is_uninitialized_parameter(param)
        )
    ]

    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append(
            {
                'params': head_params,
                'lr': float(lr),
                'weight_decay': 0.0,
                'name': 'head',
            }
        )
    if encoder_params:
        groups.append(
            {
                'params': encoder_params,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': 'encoder',
            }
        )
    if not groups:
        raise ValueError('No trainable parameters found.')
    return groups


def _print_optimizer_groups(optimizer: torch.optim.Optimizer) -> None:
    """
    Print parameter counts and hyperparameters for optimizer groups.
    """
    parts = []
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get('name', f'group{idx}'))
        num_params = sum(
            int(param.numel())
            for param in group['params']
            if not _is_uninitialized_parameter(param)
        )
        parts.append(
            f'{name}: params={num_params} '
            f'lr={float(group["lr"]):.2e} '
            f'weight_decay={float(group["weight_decay"]):.2e}'
        )
    print('optimizer groups: ' + '; '.join(parts))


def _is_uninitialized_parameter(param: torch.nn.Parameter) -> bool:
    """
    Return whether a parameter belongs to an unmaterialized lazy module.
    """
    return isinstance(param, UninitializedParameter)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    min_lr_ratio: float,
    exponential_gamma: float = 0.8,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Create the configured learning-rate scheduler.

    Supported policies are constant, exponential, cosine, plateau, and none. Exponential and cosine are implemented with a minimum learning-rate ratio floor.
    """
    if scheduler_name == 'none':
        return None
    min_lr_ratio = max(0.0, min(float(min_lr_ratio), 1.0))
    if scheduler_name == 'plateau':
        min_lrs = [
            float(group['lr']) * min_lr_ratio for group in optimizer.param_groups
        ]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            min_lr=min_lrs,
        )
    if scheduler_name not in {'cosine', 'constant', 'exponential'}:
        raise ValueError(f'Unknown scheduler: {scheduler_name}')

    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int) -> float:
        if scheduler_name == 'constant':
            return 1.0
        if scheduler_name == 'exponential':
            return max(min_lr_ratio, float(exponential_gamma) ** max(step, 0))
        progress = min(max(step / total_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _scheduler_total_steps(
    *,
    epochs: int,
    steps_per_epoch: int,
    scheduler_interval: str,
) -> int:
    """
    Return scheduler step count implied by epochs and interval.
    """
    if scheduler_interval == 'epoch':
        return int(epochs)
    if scheduler_interval == 'step':
        return int(epochs) * int(steps_per_epoch)
    raise ValueError('scheduler_interval must be one of: epoch, step.')


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Return the largest current learning rate across optimizer groups.
    """
    return float(max(group['lr'] for group in optimizer.param_groups))
