from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from mirafrag.model import MiraFragModel


def _optimizer_param_groups(
    model: MiraFragModel,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
    head_lr: float | None = None,
    encoder_lr: float | None = None,
    head_weight_decay: float = 0.0,
    encoder_weight_decay: float | None = None,
    encoder_layer_lr_decay: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Build AdamW parameter groups for head and trainable encoder parameters.

    Learning rates and weight decays can be controlled separately for head and
    encoder parameters. Weight decay is applied only to decayable weight matrices;
    biases, normalization parameters, and embeddings stay in no-decay groups.
    """
    if lr is None and (head_lr is None or encoder_lr is None):
        raise ValueError('lr is required unless both head_lr and encoder_lr are set.')
    if head_lr is None:
        head_lr = float(lr)
    if encoder_lr is None:
        encoder_lr = float(lr)
    if encoder_weight_decay is None:
        encoder_weight_decay = 0.0 if weight_decay is None else float(weight_decay)
    if encoder_layer_lr_decay <= 0.0 or encoder_layer_lr_decay > 1.0:
        raise ValueError('encoder_layer_lr_decay must be in the interval (0, 1].')

    modules = dict(model.named_modules())
    encoder_named_params = [
        (f'encoder.{name}', param)
        for name, param in model.encoder.named_parameters()
        if param.requires_grad and not _is_uninitialized_parameter(param)
    ]
    encoder_param_ids = {id(param) for _, param in encoder_named_params}
    head_named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if (
            param.requires_grad
            and id(param) not in encoder_param_ids
            and not _is_uninitialized_parameter(param)
        )
    ]

    groups: list[dict[str, Any]] = []
    _append_decay_split_groups(
        groups,
        name='head',
        named_params=head_named_params,
        modules=modules,
        lr=float(head_lr),
        weight_decay=float(head_weight_decay),
    )
    _append_encoder_groups(
        groups,
        named_params=encoder_named_params,
        modules=modules,
        lr=float(encoder_lr),
        weight_decay=float(encoder_weight_decay),
        layer_lr_decay=float(encoder_layer_lr_decay),
    )
    if not groups:
        raise ValueError('No trainable parameters found.')
    return groups


def _append_encoder_groups(
    groups: list[dict[str, Any]],
    *,
    named_params: list[tuple[str, torch.nn.Parameter]],
    modules: dict[str, nn.Module],
    lr: float,
    weight_decay: float,
    layer_lr_decay: float,
) -> None:
    """
    Append encoder optimizer groups, optionally using AIMNet MLP layerwise LR decay.

    With layer_lr_decay < 1, the last AIMNet message block receives ``lr`` and
    earlier blocks receive progressively smaller rates. Non-MLP encoder
    parameters are treated as the earliest encoder group. Non-AIMNet encoders or
    the default decay of 1 preserve the historical single encoder group.
    """
    if not named_params:
        return
    if layer_lr_decay == 1.0:
        _append_decay_split_groups(
            groups,
            name='encoder',
            named_params=named_params,
            modules=modules,
            lr=lr,
            weight_decay=weight_decay,
        )
        return

    mlp_indices = sorted(
        {
            layer_index
            for name, _param in named_params
            if (layer_index := _aimnet_mlp_index(name)) is not None
        }
    )
    if not mlp_indices:
        _append_decay_split_groups(
            groups,
            name='encoder',
            named_params=named_params,
            modules=modules,
            lr=lr,
            weight_decay=weight_decay,
        )
        return

    max_mlp_index = max(mlp_indices)
    buckets: dict[tuple[str, int], list[tuple[str, torch.nn.Parameter]]] = defaultdict(
        list
    )
    for name, param in named_params:
        mlp_index = _aimnet_mlp_index(name)
        if mlp_index is None:
            bucket_name = 'encoder_base'
            exponent = max_mlp_index + 1
        else:
            bucket_name = f'encoder_mlp{mlp_index}'
            exponent = max_mlp_index - mlp_index
        buckets[(bucket_name, exponent)].append((name, param))

    for (bucket_name, exponent), bucket_params in sorted(
        buckets.items(), key=lambda item: (-item[0][1], item[0][0])
    ):
        _append_decay_split_groups(
            groups,
            name=bucket_name,
            named_params=bucket_params,
            modules=modules,
            lr=lr * (layer_lr_decay**exponent),
            weight_decay=weight_decay,
        )


def _aimnet_mlp_index(param_name: str) -> int | None:
    """
    Return the AIMNet ``model.mlps.<idx>`` block index for an encoder parameter.
    """
    match = re.search(r'(?:^|\.)model\.mlps\.(\d+)\.', param_name)
    if match is None:
        return None
    return int(match.group(1))


def _append_decay_split_groups(
    groups: list[dict[str, Any]],
    *,
    name: str,
    named_params: list[tuple[str, torch.nn.Parameter]],
    modules: dict[str, nn.Module],
    lr: float,
    weight_decay: float,
) -> None:
    """
    Append optimizer groups split into decay and no-decay parameters.
    """
    if not named_params:
        return
    if lr <= 0.0:
        raise ValueError(f'{name} learning rate must be positive.')
    if weight_decay < 0.0:
        raise ValueError(f'{name} weight_decay must be non-negative.')

    if weight_decay == 0.0:
        groups.append(
            {
                'params': [param for _, param in named_params],
                'lr': float(lr),
                'weight_decay': 0.0,
                'name': name,
            }
        )
        return

    decay_params = [
        param
        for param_name, param in named_params
        if _uses_weight_decay(param_name, param, modules)
    ]
    no_decay_params = [
        param
        for param_name, param in named_params
        if not _uses_weight_decay(param_name, param, modules)
    ]
    if decay_params:
        groups.append(
            {
                'params': decay_params,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': f'{name}_decay',
            }
        )
    if no_decay_params:
        groups.append(
            {
                'params': no_decay_params,
                'lr': float(lr),
                'weight_decay': 0.0,
                'name': f'{name}_no_decay',
            }
        )


def _uses_weight_decay(
    param_name: str,
    param: torch.nn.Parameter,
    modules: dict[str, nn.Module],
) -> bool:
    """
    Return whether AdamW weight decay should apply to a parameter.
    """
    if param.ndim < 2 or param_name.endswith('.bias'):
        return False
    module_name = param_name.rsplit('.', 1)[0] if '.' in param_name else ''
    module = modules.get(module_name)
    if isinstance(module, (nn.Embedding, nn.LayerNorm)):
        return False
    return True


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
