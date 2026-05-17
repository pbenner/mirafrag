from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mirafrag.data import move_batch_to_device
from mirafrag.model import MiraFragModel, save_checkpoint
from mirafrag.probability import fragment_oos_log_probs


def spectrum_loss(
    pred: dict[str, Any],
    batch: dict[str, Any],
    loss: str = 'kl',
    *,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    kl_weight: float = 0.7,
    coverage_weight: float = 0.1,
) -> torch.Tensor:
    return _sparse_spectrum_loss(
        pred,
        batch,
        loss=loss,
        mass_tolerance=mass_tolerance,
        relative_mass_tolerance=relative_mass_tolerance,
        mass_tolerance_min_mz=mass_tolerance_min_mz,
        kl_weight=kl_weight,
        coverage_weight=coverage_weight,
    )


def _sparse_spectrum_loss(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    loss: str,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    kl_weight: float,
    coverage_weight: float,
) -> torch.Tensor:
    if loss == 'kl':
        value = sparse_binned_kl_divergence(pred, batch).mean()
    elif loss == 'projected_kl':
        value = projected_sparse_binned_kl_divergence(pred, batch).mean()
    elif loss == 'soft_projected_kl':
        value = soft_projected_sparse_kl_divergence(
            pred,
            batch,
            tolerance=mass_tolerance,
            relative=relative_mass_tolerance,
            tolerance_min_mz=mass_tolerance_min_mz,
        ).mean()
    elif loss == 'soft_binned_kl':
        value = soft_binned_kl_divergence(
            pred,
            batch,
            tolerance=mass_tolerance,
            relative=relative_mass_tolerance,
            tolerance_min_mz=mass_tolerance_min_mz,
        ).mean()
    elif loss == 'soft_binned_coverage_kl':
        value = soft_binned_coverage_kl_divergence(
            pred,
            batch,
            tolerance=mass_tolerance,
            relative=relative_mass_tolerance,
            tolerance_min_mz=mass_tolerance_min_mz,
            coverage_weight=coverage_weight,
        ).mean()
    elif loss == 'fragnnet_ce':
        value = fragnnet_sparse_cross_entropy(
            pred,
            batch,
            tolerance=mass_tolerance,
            relative=relative_mass_tolerance,
            tolerance_min_mz=mass_tolerance_min_mz,
        ).mean()
    elif loss == 'cosine':
        value = 1.0 - sparse_binned_cosine_similarity(pred, batch).mean()
    elif loss == 'kl_cosine':
        kl_weight = max(0.0, min(float(kl_weight), 1.0))
        kl = sparse_binned_kl_divergence(pred, batch).mean()
        cosine_loss = 1.0 - sparse_binned_cosine_similarity(pred, batch).mean()
        value = kl_weight * kl + (1.0 - kl_weight) * cosine_loss
    elif loss == 'sqrt_cosine':
        value = (
            1.0
            - sparse_binned_cosine_similarity(
                pred,
                batch,
                sqrt=True,
            ).mean()
        )
    elif loss == 'tolerance_cosine':
        value = (
            1.0
            - sparse_tolerance_cosine_similarity(
                pred,
                batch,
                tolerance=mass_tolerance,
                relative=relative_mass_tolerance,
                tolerance_min_mz=mass_tolerance_min_mz,
            ).mean()
        )
    else:
        raise ValueError(
            f'Loss {loss!r} is not supported for sparse fragment predictions.'
        )
    return value


def sparse_binned_cosine_similarity(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    sqrt: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    pred_values = torch.exp(fragment_log_probs)
    oos_values = torch.exp(oos_log_probs)
    target_values = batch['target_intensity'].to(
        device=pred_values.device,
        dtype=pred_values.dtype,
    )
    if sqrt:
        pred_values = torch.sqrt(torch.clamp(pred_values, min=0.0))
        oos_values = torch.sqrt(torch.clamp(oos_values, min=0.0))
        target_values = torch.sqrt(torch.clamp(target_values, min=0.0))

    pred_bins = pred['bins'].long()
    pred_batch = pred['batch'].long()
    target_bins = torch.floor(
        batch['target_mz'].to(device=pred_values.device, dtype=pred_values.dtype)
        / _bin_width_from_batch(batch)
    ).long()
    target_batch = batch['target_batch'].to(device=pred_values.device).long()
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    out = pred_values.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(p_mask.any()) or not bool(t_mask.any()):
            continue
        p_keys, p_vals = _aggregate_values(
            pred_bins[p_mask],
            pred_values[p_mask],
        )
        t_keys, t_vals = _aggregate_values(
            target_bins[t_mask].clamp(0, num_bins - 1),
            target_values[t_mask],
        )
        out[batch_idx] = _keyed_cosine(
            p_keys,
            p_vals,
            t_keys,
            t_vals,
            eps=eps,
            pred_extra_values=oos_values[batch_idx].reshape(1),
        )
    return out


def sparse_tolerance_cosine_similarity(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 0.01,
    relative: bool = False,
    tolerance_min_mz: float = 200.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    pred_values = torch.exp(fragment_log_probs)
    oos_values = torch.exp(oos_log_probs)
    pred_mzs = pred['mzs'].to(device=pred_values.device, dtype=pred_values.dtype)
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(
        device=pred_values.device, dtype=pred_values.dtype
    )
    target_values = batch['target_intensity'].to(
        device=pred_values.device,
        dtype=pred_values.dtype,
    )
    target_batch = batch['target_batch'].to(device=pred_values.device).long()
    batch_size = int(pred['batch_size'])
    out = pred_values.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(p_mask.any()) or not bool(t_mask.any()):
            continue
        p_vals = pred_values[p_mask]
        t_vals = target_values[t_mask]
        p_mzs = pred_mzs[p_mask]
        t_mzs = target_mzs[t_mask]
        p_norm = torch.linalg.vector_norm(
            torch.cat([p_vals, oos_values[batch_idx].reshape(1)])
        )
        t_norm = torch.linalg.vector_norm(t_vals)
        diff = torch.abs(t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0))
        if relative:
            denom = torch.clamp(t_mzs, min=float(tolerance_min_mz)).unsqueeze(1)
            matched = diff / denom <= float(tolerance)
        else:
            matched = diff <= float(tolerance)
        weighted_pred = torch.where(
            matched,
            p_vals.unsqueeze(0).expand_as(diff),
            p_vals.new_zeros(diff.shape),
        ).amax(dim=1)
        dot = torch.sum(t_vals * weighted_pred)
        out[batch_idx] = dot / torch.clamp(p_norm * t_norm, min=eps)
    return out


def sparse_binned_kl_divergence(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_bins = pred['bins'].long()
    pred_batch = pred['batch'].long()
    target_bins = torch.floor(
        batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
        / _bin_width_from_batch(batch)
    ).long()
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue
        t_keys, t_vals = _aggregate_values(
            target_bins[t_mask].clamp(0, num_bins - 1),
            target_values[t_mask],
        )
        target_probs = t_vals / torch.clamp(t_vals.sum(), min=eps)
        if not bool(p_mask.any()):
            log_pred_at_target = oos_log_probs[batch_idx].expand_as(target_probs)
        else:
            p_keys, p_log_probs = _binned_log_probs_from_log_probs(
                pred_bins[p_mask].clamp(0, num_bins - 1),
                fragment_log_probs[p_mask],
                eps=eps,
            )
            log_pred_at_target = _lookup_log_probs(
                p_keys,
                p_log_probs,
                t_keys,
                eps=eps,
                default_log_prob=oos_log_probs[batch_idx],
            )
        out[batch_idx] = torch.sum(
            target_probs
            * (torch.log(torch.clamp(target_probs, min=eps)) - log_pred_at_target)
        )
    return out


def projected_sparse_binned_kl_divergence(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_bins = pred['bins'].long()
    pred_batch = pred['batch'].long()
    target_bins = torch.floor(
        batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
        / _bin_width_from_batch(batch)
    ).long()
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue
        if bool(p_mask.any()):
            p_keys, p_log_probs = _binned_log_probs_from_log_probs(
                pred_bins[p_mask].clamp(0, num_bins - 1),
                fragment_log_probs[p_mask],
                eps=eps,
            )
        else:
            p_keys = pred_bins.new_empty(0)
            p_log_probs = logits.new_empty(0)
        t_keys, t_vals = _aggregate_values(
            target_bins[t_mask].clamp(0, num_bins - 1),
            target_values[t_mask],
        )
        target_probs = t_vals / torch.clamp(t_vals.sum(), min=eps)
        log_pred = _lookup_log_probs(
            p_keys,
            p_log_probs,
            t_keys,
            eps=eps,
            default_log_prob=oos_log_probs[batch_idx],
        )
        out[batch_idx] = torch.sum(
            target_probs * (torch.log(torch.clamp(target_probs, min=eps)) - log_pred)
        )
    return out


def soft_projected_sparse_kl_divergence(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 0.01,
    relative: bool = False,
    tolerance_min_mz: float = 200.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_mzs = pred['mzs'].to(device=logits.device, dtype=logits.dtype)
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue

        t_values = target_values[t_mask]
        target_probs = t_values / torch.clamp(t_values.sum(), min=eps)
        if not bool(p_mask.any()):
            out[batch_idx] = -oos_log_probs[batch_idx]
            continue

        p_mzs = pred_mzs[p_mask]
        t_mzs = target_mzs[t_mask]
        tolerances = _target_tolerances(
            t_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        )
        sigma = 0.5 * tolerances
        sigma = torch.clamp(sigma, min=eps).unsqueeze(1)
        diff = t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0)
        matched = torch.abs(diff) <= tolerances.unsqueeze(1)
        kernel_logits = -0.5 * torch.square(diff / sigma)
        kernel_logits = torch.where(
            matched,
            kernel_logits,
            logits.new_full(kernel_logits.shape, math.log(eps)),
        )
        target_has_match = matched.any(dim=1)
        if bool(target_has_match.any()):
            target_to_candidate = torch.softmax(kernel_logits[target_has_match], dim=1)
            matched_target_probs = target_probs[target_has_match]
        else:
            target_to_candidate = logits.new_empty(0, int(p_mzs.numel()))
            matched_target_probs = target_probs.new_empty(0)
        projected_target = torch.sum(
            matched_target_probs.unsqueeze(1) * target_to_candidate,
            dim=0,
        )
        log_pred = fragment_log_probs[p_mask]
        ios_loss = torch.sum(
            projected_target
            * (torch.log(torch.clamp(projected_target, min=eps)) - log_pred)
        )
        oos_mass = torch.sum(target_probs[~target_has_match])
        out[batch_idx] = ios_loss + oos_mass * (
            torch.log(torch.clamp(oos_mass, min=eps)) - oos_log_probs[batch_idx]
        )
    return out


def soft_binned_coverage_kl_divergence(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 0.01,
    relative: bool = False,
    tolerance_min_mz: float = 200.0,
    coverage_weight: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_mzs = pred['mzs'].to(device=logits.device, dtype=logits.dtype)
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    bin_width = _bin_width_from_batch(batch)
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue

        t_mzs = target_mzs[t_mask]
        t_values = target_values[t_mask]
        t_probs = t_values / torch.clamp(t_values.sum(), min=eps)
        if not bool(p_mask.any()):
            out[batch_idx] = -oos_log_probs[batch_idx]
            continue

        p_mzs = pred_mzs[p_mask]
        p_probs = torch.exp(fragment_log_probs[p_mask])
        target_is_in_support = _target_support_mask(
            target_mzs=t_mzs,
            pred_mzs=p_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        )

        pred_keys, pred_probs = _soft_binned_distribution(
            p_mzs,
            p_probs,
            num_bins=num_bins,
            bin_width=bin_width,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            eps=eps,
            normalize=False,
        )
        target_keys, target_probs = _soft_binned_distribution(
            t_mzs[target_is_in_support],
            t_probs[target_is_in_support],
            num_bins=num_bins,
            bin_width=bin_width,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            eps=eps,
            normalize=False,
        )

        if target_keys.numel() > 0:
            pred_at_target = _lookup_values(
                pred_keys,
                pred_probs,
                target_keys,
                eps=eps,
            )
            kl = torch.sum(
                target_probs
                * (
                    torch.log(torch.clamp(target_probs, min=eps))
                    - torch.log(torch.clamp(pred_at_target, min=eps))
                )
            )
        else:
            kl = logits.sum() * 0.0

        target_tolerances = _target_tolerances(
            t_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        ).unsqueeze(1)
        within_tolerance = torch.abs(t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0))
        within_tolerance = within_tolerance <= target_tolerances
        coverage = torch.sum(p_probs.unsqueeze(0) * within_tolerance.float(), dim=1)
        coverage_loss = -torch.sum(
            t_probs[target_is_in_support]
            * torch.log(torch.clamp(coverage[target_is_in_support], min=eps))
        )
        oos_mass = torch.sum(t_probs[~target_is_in_support])
        oos_loss = oos_mass * (
            torch.log(torch.clamp(oos_mass, min=eps)) - oos_log_probs[batch_idx]
        )
        out[batch_idx] = kl + oos_loss + float(coverage_weight) * coverage_loss
    return out


def soft_binned_kl_divergence(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 0.01,
    relative: bool = False,
    tolerance_min_mz: float = 200.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_mzs = pred['mzs'].to(device=logits.device, dtype=logits.dtype)
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    fragment_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    bin_width = _bin_width_from_batch(batch)
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue

        t_mzs = target_mzs[t_mask]
        t_values = target_values[t_mask]
        t_probs = t_values / torch.clamp(t_values.sum(), min=eps)
        if not bool(p_mask.any()):
            out[batch_idx] = -oos_log_probs[batch_idx]
            continue

        p_mzs = pred_mzs[p_mask]
        p_probs = torch.exp(fragment_log_probs[p_mask])
        target_is_in_support = _target_support_mask(
            target_mzs=t_mzs,
            pred_mzs=p_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        )

        pred_keys, pred_probs = _soft_binned_distribution(
            p_mzs,
            p_probs,
            num_bins=num_bins,
            bin_width=bin_width,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            eps=eps,
            normalize=False,
        )
        target_keys, target_probs = _soft_binned_distribution(
            t_mzs[target_is_in_support],
            t_probs[target_is_in_support],
            num_bins=num_bins,
            bin_width=bin_width,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            eps=eps,
            normalize=False,
        )
        if target_keys.numel() > 0:
            pred_at_target = _lookup_values(
                pred_keys,
                pred_probs,
                target_keys,
                eps=eps,
            )
            kl = torch.sum(
                target_probs
                * (
                    torch.log(torch.clamp(target_probs, min=eps))
                    - torch.log(torch.clamp(pred_at_target, min=eps))
                )
            )
        else:
            kl = logits.sum() * 0.0
        oos_mass = torch.sum(t_probs[~target_is_in_support])
        out[batch_idx] = kl + oos_mass * (
            torch.log(torch.clamp(oos_mass, min=eps)) - oos_log_probs[batch_idx]
        )
    return out


def fragnnet_sparse_cross_entropy(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = 200.0,
    tolerance_multiple: float = 1.0,
    gaussian_renormalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    logits = pred['logits']
    pred_mzs = pred['mzs'].to(device=logits.device, dtype=logits.dtype)
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(device=logits.device, dtype=logits.dtype)
    target_values = batch['target_intensity'].to(
        device=logits.device,
        dtype=logits.dtype,
    )
    target_batch = batch['target_batch'].to(device=logits.device).long()
    frag_log_probs, oos_log_probs = fragment_oos_log_probs(pred)
    batch_size = int(pred['batch_size'])
    out = logits.sum() * 0.0 + logits.new_zeros(batch_size)
    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue

        t_mzs = target_mzs[t_mask]
        t_values = target_values[t_mask]
        target_probs = t_values / torch.clamp(t_values.sum(), min=eps)
        oos_log_prob = oos_log_probs[batch_idx]
        if not bool(p_mask.any()):
            out[batch_idx] = -oos_log_prob
            continue

        p_mzs = pred_mzs[p_mask]
        p_log_probs = frag_log_probs[p_mask]
        diff = torch.abs(t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0))
        match_tolerances = _target_tolerances(
            t_mzs,
            tolerance=float(tolerance) * float(tolerance_multiple),
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        ).unsqueeze(1)
        matched = diff <= match_tolerances

        if relative:
            stds = torch.clamp(p_mzs, min=float(tolerance_min_mz)) * float(tolerance)
        else:
            stds = p_mzs.new_full(p_mzs.shape, float(tolerance))
        variances = torch.square(torch.clamp(stds, min=eps))
        normal_log_probs = -0.5 * torch.square(
            t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0)
        ) / variances.unsqueeze(0) - 0.5 * torch.log(
            2.0 * math.pi * variances
        ).unsqueeze(0)
        if gaussian_renormalize:
            trunc_factor = torch.erf(
                logits.new_tensor(float(tolerance_multiple) / math.sqrt(2.0))
            )
            normal_log_probs = normal_log_probs - torch.log(
                torch.clamp(trunc_factor, min=eps)
            )
        masked_log_probs = torch.where(
            matched,
            normal_log_probs + p_log_probs.unsqueeze(0),
            logits.new_full(normal_log_probs.shape, math.log(eps)),
        )
        target_log_probs = torch.logsumexp(masked_log_probs, dim=1)
        target_is_in_support = matched.any(dim=1)
        ios_loss = -torch.sum(
            target_probs[target_is_in_support] * target_log_probs[target_is_in_support]
        )
        oos_mass = torch.sum(target_probs[~target_is_in_support])
        out[batch_idx] = ios_loss - oos_mass * oos_log_prob
    return out


def _binned_log_probs_from_log_probs(
    bins: torch.Tensor,
    log_probs: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_bins, inverse = torch.unique(bins.long(), sorted=True, return_inverse=True)
    if unique_bins.numel() == 0:
        return unique_bins, log_probs.new_empty(0)
    max_log_probs = log_probs.new_full(unique_bins.shape, -torch.inf)
    max_log_probs.scatter_reduce_(
        0, inverse, log_probs, reduce='amax', include_self=True
    )
    stable_values = torch.exp(log_probs - max_log_probs[inverse])
    stable_sums = log_probs.new_zeros(unique_bins.shape)
    stable_sums.index_add_(0, inverse, stable_values)
    return unique_bins, max_log_probs + torch.log(torch.clamp(stable_sums, min=eps))


def _lookup_log_probs(
    pred_keys: torch.Tensor,
    pred_log_probs: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    eps: float,
    default_log_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    if default_log_prob is None:
        out = pred_log_probs.new_full(target_keys.shape, math.log(eps))
    else:
        out = default_log_prob.expand(target_keys.shape).clone()
    positions = torch.searchsorted(pred_keys, target_keys)
    in_range = positions < pred_keys.numel()
    if bool(in_range.any()):
        in_range_positions = positions[in_range]
        valid = pred_keys[in_range_positions] == target_keys[in_range]
        if bool(valid.any()):
            valid_target_indices = in_range.nonzero(as_tuple=False).flatten()[valid]
            out[valid_target_indices] = pred_log_probs[in_range_positions[valid]]
    return out


def _lookup_values(
    pred_keys: torch.Tensor,
    pred_values: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    out = pred_values.new_full(target_keys.shape, float(eps))
    positions = torch.searchsorted(pred_keys, target_keys)
    in_range = positions < pred_keys.numel()
    if bool(in_range.any()):
        in_range_positions = positions[in_range]
        valid = pred_keys[in_range_positions] == target_keys[in_range]
        if bool(valid.any()):
            valid_target_indices = in_range.nonzero(as_tuple=False).flatten()[valid]
            out[valid_target_indices] = pred_values[in_range_positions[valid]]
    return out


def _bin_width_from_batch(batch: dict[str, Any]) -> float:
    bin_width = batch.get('bin_width', 0.01)
    if isinstance(bin_width, torch.Tensor):
        return float(bin_width.flatten()[0].detach().cpu())
    return float(bin_width)


def _target_tolerances(
    target_mzs: torch.Tensor,
    *,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
) -> torch.Tensor:
    if relative:
        denominator = torch.clamp(target_mzs, min=float(tolerance_min_mz))
        return denominator * float(tolerance)
    return target_mzs.new_full(target_mzs.shape, float(tolerance))


def _target_support_mask(
    *,
    target_mzs: torch.Tensor,
    pred_mzs: torch.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
) -> torch.Tensor:
    if pred_mzs.numel() == 0:
        return torch.zeros_like(target_mzs, dtype=torch.bool)
    tolerances = _target_tolerances(
        target_mzs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    diff = torch.abs(target_mzs.unsqueeze(1) - pred_mzs.unsqueeze(0))
    return (diff <= tolerances.unsqueeze(1)).any(dim=1)


def _soft_binned_distribution(
    mzs: torch.Tensor,
    values: torch.Tensor,
    *,
    num_bins: int,
    bin_width: float,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    eps: float,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mzs.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=mzs.device), values.new_empty(0)
    safe_bin_width = max(float(bin_width), eps)
    tolerances = torch.clamp(
        _target_tolerances(
            mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        ),
        min=safe_bin_width,
    )
    keys: list[torch.Tensor] = []
    out_values: list[torch.Tensor] = []
    for idx in range(int(mzs.numel())):
        mz = mzs[idx]
        tol = tolerances[idx]
        low_bin = max(0, int(torch.floor((mz - tol) / safe_bin_width).detach().cpu()))
        high_bin = min(
            int(num_bins) - 1,
            int(torch.floor((mz + tol) / safe_bin_width).detach().cpu()),
        )
        if high_bin < low_bin:
            center_bin = int(torch.floor(mz / safe_bin_width).detach().cpu())
            low_bin = high_bin = min(max(center_bin, 0), int(num_bins) - 1)
        bin_indices = torch.arange(
            low_bin,
            high_bin + 1,
            dtype=torch.long,
            device=mzs.device,
        )
        centers = (bin_indices.to(dtype=mzs.dtype) + 0.5) * safe_bin_width
        sigma = torch.clamp(0.5 * tol, min=eps)
        kernel = torch.exp(-0.5 * torch.square((centers - mz) / sigma))
        kernel = kernel / torch.clamp(kernel.sum(), min=eps)
        keys.append(bin_indices)
        out_values.append(values[idx] * kernel)
    all_keys = torch.cat(keys)
    all_values = torch.cat(out_values)
    dist_keys, dist_values = _aggregate_values(
        all_keys,
        all_values,
    )
    if normalize:
        dist_values = dist_values / torch.clamp(dist_values.sum(), min=eps)
    return dist_keys, dist_values


def _aggregate_values(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_keys, inverse = torch.unique(keys.long(), sorted=True, return_inverse=True)
    out = values.new_zeros(unique_keys.shape)
    out.index_add_(0, inverse, values)
    return unique_keys, out


def _keyed_cosine(
    pred_keys: torch.Tensor,
    pred_values: torch.Tensor,
    target_keys: torch.Tensor,
    target_values: torch.Tensor,
    *,
    eps: float,
    pred_extra_values: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred_extra_values is None or pred_extra_values.numel() == 0:
        pred_norm = torch.linalg.vector_norm(pred_values)
    else:
        pred_norm = torch.linalg.vector_norm(
            torch.cat([pred_values, pred_extra_values.to(pred_values)])
        )
    target_norm = torch.linalg.vector_norm(target_values)
    positions = torch.searchsorted(target_keys, pred_keys)
    in_range = positions < target_keys.numel()
    valid = torch.zeros_like(in_range)
    if bool(in_range.any()):
        in_range_positions = positions[in_range]
        valid[in_range] = target_keys[in_range_positions] == pred_keys[in_range]
    if bool(valid.any()):
        dot = torch.sum(pred_values[valid] * target_values[positions[valid]])
    else:
        dot = pred_values.sum() * 0.0
    return dot / torch.clamp(pred_norm * target_norm, min=eps)


def run_epoch(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
    loss_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = 'step',
    desc: str | None = None,
    show_progress: bool = True,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    kl_weight: float = 0.7,
    coverage_weight: float = 0.1,
) -> dict[str, float]:
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError('scheduler_interval must be one of: epoch, step.')
    training = optimizer is not None
    model.train(training)
    total_examples = 0
    total_batches = len(loader)
    processed_batches = 0
    loss_sum = 0.0
    cosine_sum = 0.0
    objective_name = _objective_display_name(loss_name)
    progress = tqdm(
        loader,
        desc=desc,
        total=total_batches,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for raw_batch in progress:
        processed_batches += 1
        batch = move_batch_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            pred = model(batch)
            batch_size = int(pred['batch_size'])
            loss = spectrum_loss(
                pred,
                batch,
                loss=loss_name,
                mass_tolerance=mass_tolerance,
                relative_mass_tolerance=relative_mass_tolerance,
                mass_tolerance_min_mz=mass_tolerance_min_mz,
                kl_weight=kl_weight,
                coverage_weight=coverage_weight,
            )
        if training:
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_interval == 'step':
                scheduler.step()
        with torch.no_grad():
            loss_value = float(loss.detach().cpu())
            cosine_value = float(
                sparse_binned_cosine_similarity(pred, batch).mean().cpu()
            )
            total_examples += batch_size
            loss_sum += loss_value * batch_size
            cosine_sum += cosine_value * batch_size
            if show_progress:
                progress.set_postfix(
                    {
                        f'{objective_name}_avg': (
                            f'{loss_sum / max(total_examples, 1):.4f}'
                        ),
                        'cosine_avg': (f'{cosine_sum / max(total_examples, 1):.4f}'),
                        'lr': f'{_current_lr(optimizer):.2e}'
                        if optimizer is not None
                        else 'n/a',
                    }
                )

    if processed_batches != total_batches:
        print(
            f'Warning: {desc or "epoch"} processed {processed_batches}/'
            f'{total_batches} batches.'
        )

    return {
        'loss': float(loss_sum / max(total_examples, 1)),
        'cosine': float(cosine_sum / max(total_examples, 1)),
    }


def train_model(
    model: MiraFragModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str | torch.device,
    output: str | Path,
    loss_name: str = 'cosine',
    train_config: dict[str, Any] | None = None,
    show_progress: bool = True,
    scheduler_name: str = 'exponential',
    scheduler_interval: str = 'epoch',
    min_lr_ratio: float = 0.1,
    exponential_gamma: float = 0.8,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
    evaluate_initial: bool = False,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    kl_weight: float = 0.7,
    coverage_weight: float = 0.1,
) -> dict[str, list[float]]:
    model.to(device)
    _materialize_lazy_modules(model, train_loader, device=device)
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(
            model,
            lr=lr,
            weight_decay=weight_decay,
        ),
        lr=lr,
        weight_decay=weight_decay,
    )
    _print_optimizer_groups(optimizer)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        total_steps=_scheduler_total_steps(
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            scheduler_interval=scheduler_interval,
        ),
        min_lr_ratio=min_lr_ratio,
        exponential_gamma=exponential_gamma,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
    )
    history: dict[str, list[float]] = {
        'epoch': [],
        'train_loss': [],
        'train_cosine': [],
        'val_loss': [],
        'val_cosine': [],
    }
    best_val = float('inf')
    objective_name = _objective_display_name(loss_name)

    if evaluate_initial and val_loader is not None:
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            loss_name=loss_name,
            desc=f'val epoch 0/{epochs}',
            show_progress=show_progress,
            mass_tolerance=mass_tolerance,
            relative_mass_tolerance=relative_mass_tolerance,
            mass_tolerance_min_mz=mass_tolerance_min_mz,
            kl_weight=kl_weight,
            coverage_weight=coverage_weight,
        )
        _append_history(
            history,
            epoch=0,
            train_stats=None,
            val_stats=val_stats,
        )
        best_val = val_stats['loss']
        print(
            'epoch=0 '
            f'val_{objective_name}={val_stats["loss"]:.5f} '
            f'val_cosine={val_stats["cosine"]:.5f} '
            f'lr={_current_lr(optimizer):.2e}'
        )
        save_checkpoint(output, model, train_config=train_config)
        print(f'saved checkpoint to {output} val_{objective_name}={best_val:.5f}')

    for epoch in range(1, epochs + 1):
        epoch_lr = _current_lr(optimizer)
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_name != 'plateau' else None,
            scheduler_interval=scheduler_interval,
            device=device,
            loss_name=loss_name,
            desc=f'train epoch {epoch}/{epochs}',
            show_progress=show_progress,
            mass_tolerance=mass_tolerance,
            relative_mass_tolerance=relative_mass_tolerance,
            mass_tolerance_min_mz=mass_tolerance_min_mz,
            kl_weight=kl_weight,
            coverage_weight=coverage_weight,
        )
        val_stats = (
            run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_name=loss_name,
                desc=f'val epoch {epoch}/{epochs}',
                show_progress=show_progress,
                mass_tolerance=mass_tolerance,
                relative_mass_tolerance=relative_mass_tolerance,
                mass_tolerance_min_mz=mass_tolerance_min_mz,
                kl_weight=kl_weight,
                coverage_weight=coverage_weight,
            )
            if val_loader is not None
            else train_stats
        )

        _append_history(
            history,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
        )

        print(
            f'epoch={epoch} '
            f'train_{objective_name}={train_stats["loss"]:.5f} '
            f'train_cosine={train_stats["cosine"]:.5f} '
            f'val_{objective_name}={val_stats["loss"]:.5f} '
            f'val_cosine={val_stats["cosine"]:.5f} '
            f'lr={epoch_lr:.2e}'
        )

        if val_stats['loss'] <= best_val:
            best_val = val_stats['loss']
            save_checkpoint(output, model, train_config=train_config)
            print(f'saved checkpoint to {output} val_{objective_name}={best_val:.5f}')

        if scheduler_name == 'plateau' and scheduler is not None:
            scheduler.step(val_stats['loss'])
        elif scheduler is not None and scheduler_interval == 'epoch':
            scheduler.step()

    history_path = Path(str(output).replace('.pt', '.history.json'))
    with open(history_path, 'w') as fp:
        json.dump(history, fp, indent=2)
    return history


def _append_history(
    history: dict[str, list[float]],
    *,
    epoch: int,
    train_stats: dict[str, float] | None,
    val_stats: dict[str, float],
) -> None:
    history['epoch'].append(epoch)
    if train_stats is None:
        history['train_loss'].append(float('nan'))
        history['train_cosine'].append(float('nan'))
    else:
        history['train_loss'].append(float(train_stats['loss']))
        history['train_cosine'].append(float(train_stats['cosine']))
    history['val_loss'].append(float(val_stats['loss']))
    history['val_cosine'].append(float(val_stats['cosine']))


def _objective_display_name(loss_name: str) -> str:
    names = {
        'kl': 'kl',
        'projected_kl': 'projected_kl',
        'soft_projected_kl': 'soft_projected_kl',
        'soft_binned_kl': 'soft_binned_kl',
        'soft_binned_coverage_kl': 'soft_binned_coverage_kl',
        'fragnnet_ce': 'fragnnet_ce',
        'kl_cosine': 'kl_cosine_loss',
        'cosine': 'cosine_loss',
        'sqrt_cosine': 'sqrt_cosine_loss',
        'tolerance_cosine': 'tolerance_cosine_loss',
    }
    return names.get(loss_name, 'loss')


def _optimizer_param_groups(
    model: MiraFragModel,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
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
    if scheduler_interval == 'epoch':
        return int(epochs)
    if scheduler_interval == 'step':
        return int(epochs) * int(steps_per_epoch)
    raise ValueError('scheduler_interval must be one of: epoch, step.')


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(max(group['lr'] for group in optimizer.param_groups))


def _materialize_lazy_modules(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> None:
    try:
        first_item = loader.dataset[0]
    except IndexError as exc:
        raise ValueError('Cannot train MiraFrag with an empty DataLoader.') from exc
    collate_fn = loader.collate_fn or default_collate
    raw_batch = collate_fn([first_item])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(move_batch_to_device(raw_batch, device))
    model.train(was_training)


@torch.no_grad()
def evaluate_model(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    device: str | torch.device,
    min_intensity: float = 0.001,
    top_k: int = 100,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.to(device)
    model.eval()
    rows = []
    all_cos = []
    all_sqrt = []

    progress = tqdm(
        loader,
        desc='evaluate',
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for raw_batch in progress:
        batch = move_batch_to_device(raw_batch, device)
        probs = model.predict_proba(batch)
        cos = sqrt_cos = None
        if 'target_mz' in batch:
            cos = sparse_binned_cosine_similarity(probs, batch)
            sqrt_cos = sparse_binned_cosine_similarity(probs, batch, sqrt=True)
            all_cos.extend(float(x) for x in cos.detach().cpu())
            all_sqrt.extend(float(x) for x in sqrt_cos.detach().cpu())
        sparse_rows = _sparse_prediction_rows(
            probs,
            bin_width=_bin_width_from_batch(batch),
            min_intensity=min_intensity,
            top_k=top_k,
        )

        for i, smiles in enumerate(batch['smiles']):
            row = {
                'identifier': batch['identifier'][i],
                'smiles': smiles,
                'pred_peaks': json.dumps(sparse_rows[i]),
            }
            if cos is not None and sqrt_cos is not None:
                row['cosine'] = float(cos[i].detach().cpu())
                row['sqrt_cosine'] = float(sqrt_cos[i].detach().cpu())
            rows.append(row)

    summary = {
        'cosine_mean': float(sum(all_cos) / len(all_cos)) if all_cos else float('nan'),
        'sqrt_cosine_mean': float(sum(all_sqrt) / len(all_sqrt))
        if all_sqrt
        else float('nan'),
    }
    return pd.DataFrame(rows), summary


def _sparse_prediction_rows(
    pred: dict[str, Any],
    *,
    bin_width: float,
    min_intensity: float,
    top_k: int,
) -> list[dict[str, list[float]]]:
    log_probs, _oos_log_probs = fragment_oos_log_probs(pred)
    values = torch.exp(log_probs).detach()
    bins = pred['bins'].detach().long()
    batch = pred['batch'].long()
    rows = []
    for batch_idx in range(int(pred['batch_size'])):
        mask = batch == batch_idx
        if not bool(mask.any()):
            rows.append({'mz': [], 'intensity': []})
            continue
        batch_bins, batch_values = _aggregate_values(
            bins[mask],
            values[mask],
        )
        if top_k is not None and top_k > 0 and top_k < batch_values.numel():
            batch_values, top_idx = torch.topk(batch_values, k=top_k)
            batch_bins = batch_bins[top_idx]
        keep = batch_values >= float(min_intensity)
        batch_values = batch_values[keep]
        batch_bins = batch_bins[keep]
        order = torch.argsort(batch_bins)
        batch_values = batch_values[order]
        batch_bins = batch_bins[order]
        if batch_values.numel() > 0:
            batch_values = (
                batch_values / torch.clamp(batch_values.max(), min=1e-12) * 100.0
            )
        batch_mzs = (batch_bins.to(dtype=batch_values.dtype) + 0.5) * float(bin_width)
        rows.append(
            {
                'mz': [round(float(x), 6) for x in batch_mzs.cpu().tolist()],
                'intensity': [round(float(x), 6) for x in batch_values.cpu().tolist()],
            }
        )
    return rows
