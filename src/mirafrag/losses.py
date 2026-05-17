from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from mirafrag.probability import fragment_oos_log_probs


@dataclass(frozen=True)
class LossSpec:
    """
    Registry entry for a spectrum loss.

    Each entry pairs the callable implementation with the display name printed in progress bars and epoch summaries.
    """

    compute: Callable[..., torch.Tensor]
    display_name: str


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
    """
    Compute the configured sparse spectrum training loss.

    The prediction dictionary must come from :class:`MiraFragModel` and the batch must contain flattened target m/z, target intensity, and target batch tensors. The returned scalar is suitable for backpropagation.
    """
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
    """
    Dispatch a sparse prediction to the selected registered loss implementation.
    """
    try:
        loss_spec = LOSS_REGISTRY[loss]
    except KeyError as exc:
        raise ValueError(
            f'Loss {loss!r} is not supported for sparse fragment predictions. '
            f'Supported losses: {", ".join(LOSS_NAMES)}.'
        ) from exc
    return loss_spec.compute(
        pred,
        batch,
        mass_tolerance=mass_tolerance,
        relative_mass_tolerance=relative_mass_tolerance,
        mass_tolerance_min_mz=mass_tolerance_min_mz,
        kl_weight=kl_weight,
        coverage_weight=coverage_weight,
    )


def sparse_binned_cosine_similarity(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    sqrt: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute MassSpecGym-style binned cosine for sparse predictions.

    Fragment probabilities are aggregated into bins, OOS probability contributes to the prediction norm but not the target dot product, and one cosine value is returned per spectrum.
    """
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
    """
    Compute cosine similarity using m/z tolerance matching instead of exact bins.

    Each target peak is matched to the largest predicted probability within absolute or relative tolerance. OOS probability still contributes to the prediction norm.
    """
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
    """
    Compute binned KL divergence from target bins to predicted sparse bins.

    Target intensities are normalized per spectrum. Missing target bins are assigned to the learned OOS probability, which lets the model account for peaks outside generated support.
    """
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
    """
    Compute KL on target bins projected onto generated support plus OOS.

    This has the same reachable-support optimum as binned cosine while retaining KL-style gradients and explicit supervision for unreachable target mass.
    """
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
    """
    Compute projected KL with Gaussian target-to-candidate assignment.

    Measured peaks distribute their probability mass over nearby candidate m/z values within tolerance, reducing hard bin-boundary effects while keeping an OOS term for unmatched peaks.
    """
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
    """
    Compute soft-binned KL plus an explicit coverage penalty.

    Predicted and target peaks are smoothed into neighboring bins with Gaussian kernels. A separate coverage term rewards placing probability within tolerance of measured peaks.
    """
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
    """
    Compute KL between softly binned target and predicted distributions.

    Both measured peaks and candidate probabilities are spread into neighboring bins with Gaussian kernels, preserving probability mass while reducing hard bin artifacts.
    """
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
    """
    Compute FraGNNet-style sparse Gaussian cross entropy.

    Predicted fragment probabilities define a Gaussian mixture over m/z. Target peaks within tolerance are trained against that mixture, while unmatched target mass is trained against the OOS probability.
    """
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


def _loss_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for sparse binned KL.
    """
    return sparse_binned_kl_divergence(pred, batch).mean()


def _loss_projected_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for projected sparse binned KL.
    """
    return projected_sparse_binned_kl_divergence(pred, batch).mean()


def _loss_soft_projected_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for soft projected KL.
    """
    return soft_projected_sparse_kl_divergence(
        pred,
        batch,
        tolerance=mass_tolerance,
        relative=relative_mass_tolerance,
        tolerance_min_mz=mass_tolerance_min_mz,
    ).mean()


def _loss_soft_binned_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for soft binned KL.
    """
    return soft_binned_kl_divergence(
        pred,
        batch,
        tolerance=mass_tolerance,
        relative=relative_mass_tolerance,
        tolerance_min_mz=mass_tolerance_min_mz,
    ).mean()


def _loss_soft_binned_coverage_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    coverage_weight: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for soft binned KL with coverage penalty.
    """
    return soft_binned_coverage_kl_divergence(
        pred,
        batch,
        tolerance=mass_tolerance,
        relative=relative_mass_tolerance,
        tolerance_min_mz=mass_tolerance_min_mz,
        coverage_weight=coverage_weight,
    ).mean()


def _loss_fragnnet_ce(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for FraGNNet-style sparse cross entropy.
    """
    return fragnnet_sparse_cross_entropy(
        pred,
        batch,
        tolerance=mass_tolerance,
        relative=relative_mass_tolerance,
        tolerance_min_mz=mass_tolerance_min_mz,
    ).mean()


def _loss_cosine(
    pred: dict[str, Any],
    batch: dict[str, Any],
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for direct binned cosine loss.
    """
    return 1.0 - sparse_binned_cosine_similarity(pred, batch).mean()


def _loss_kl_cosine(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    kl_weight: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for weighted KL and cosine losses.
    """
    kl_weight = max(0.0, min(float(kl_weight), 1.0))
    kl = sparse_binned_kl_divergence(pred, batch).mean()
    cosine_loss = 1.0 - sparse_binned_cosine_similarity(pred, batch).mean()
    return kl_weight * kl + (1.0 - kl_weight) * cosine_loss


def _loss_sqrt_cosine(
    pred: dict[str, Any],
    batch: dict[str, Any],
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for square-root transformed cosine loss.
    """
    return (
        1.0
        - sparse_binned_cosine_similarity(
            pred,
            batch,
            sqrt=True,
        ).mean()
    )


def _loss_tolerance_cosine(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    mass_tolerance: float,
    relative_mass_tolerance: bool,
    mass_tolerance_min_mz: float,
    **_: Any,
) -> torch.Tensor:
    """
    Loss-registry wrapper for tolerance-based cosine loss.
    """
    return (
        1.0
        - sparse_tolerance_cosine_similarity(
            pred,
            batch,
            tolerance=mass_tolerance,
            relative=relative_mass_tolerance,
            tolerance_min_mz=mass_tolerance_min_mz,
        ).mean()
    )


LOSS_REGISTRY: dict[str, LossSpec] = {
    'kl': LossSpec(_loss_kl, 'kl'),
    'projected_kl': LossSpec(_loss_projected_kl, 'projected_kl'),
    'soft_projected_kl': LossSpec(_loss_soft_projected_kl, 'soft_projected_kl'),
    'soft_binned_kl': LossSpec(_loss_soft_binned_kl, 'soft_binned_kl'),
    'soft_binned_coverage_kl': LossSpec(
        _loss_soft_binned_coverage_kl,
        'soft_binned_coverage_kl',
    ),
    'fragnnet_ce': LossSpec(_loss_fragnnet_ce, 'fragnnet_ce'),
    'kl_cosine': LossSpec(_loss_kl_cosine, 'kl_cosine_loss'),
    'cosine': LossSpec(_loss_cosine, 'cosine_loss'),
    'sqrt_cosine': LossSpec(_loss_sqrt_cosine, 'sqrt_cosine_loss'),
    'tolerance_cosine': LossSpec(_loss_tolerance_cosine, 'tolerance_cosine_loss'),
}
LOSS_NAMES = tuple(LOSS_REGISTRY.keys())


def _binned_log_probs_from_log_probs(
    bins: torch.Tensor,
    log_probs: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate candidate log probabilities by m/z bin using stable log-sum-exp.
    """
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
    """
    Look up sorted predicted log probabilities at target keys with a default fallback.
    """
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
    """
    Look up sorted predicted values at target keys with an epsilon fallback.
    """
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
    """
    Read the scalar bin width stored in a collated batch.
    """
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
    """
    Return absolute m/z tolerances for target peaks.
    """
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
    """
    Return which target peaks have at least one candidate within tolerance.
    """
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
    """
    Spread sparse peak values into neighboring bins with normalized Gaussian kernels.
    """
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
    """
    Sum values that share the same integer key.
    """
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
    """
    Compute cosine similarity between two sparse keyed vectors.
    """
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
