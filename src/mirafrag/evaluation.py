from __future__ import annotations

import json
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mirafrag.data import move_batch_to_device
from mirafrag.losses import (
    _aggregate_values,
    _bin_width_from_batch,
    _fragment_only_log_probs,
    _target_tolerances,
    sparse_binned_cosine_similarity,
    sparse_decoupled_oos_probability,
    sparse_fragment_only_binned_cosine_similarity,
    sparse_oos_probability,
)
from mirafrag.model import MiraFragModel
from mirafrag.probability import fragment_oos_log_probs


@torch.no_grad()
def evaluate_model(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    device: str | torch.device,
    min_intensity: float = 0.001,
    top_k: int = 100,
    show_progress: bool = True,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    probability_mode: str = 'joint',
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Evaluate a MiraFrag model and return predictions plus summary metrics.

    The function runs the model in evaluation mode, computes sparse binned cosine metrics when targets are present, converts probabilities to sparse peak rows, and returns a dataframe-ready result. ``probability_mode='decoupled'`` uses fragment-only probabilities and sigmoid OOS semantics for checkpoints trained with ``LOSS=decoupled_kl``.
    """
    if probability_mode not in {'joint', 'decoupled'}:
        raise ValueError("probability_mode must be one of: 'joint', 'decoupled'.")
    model.to(device)
    model.eval()
    rows = []
    all_cos = []
    all_sqrt = []
    all_cos_no_oos = []
    all_predicted_oos = []
    all_candidate_coverage = []
    all_oos_target_mass = []
    all_oracle_binned = []
    all_oracle_tolerance = []

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
        cos = sqrt_cos = diagnostics = None
        if 'target_mz' in batch:
            if probability_mode == 'decoupled':
                cos = sparse_fragment_only_binned_cosine_similarity(probs, batch)
                sqrt_cos = sparse_fragment_only_binned_cosine_similarity(
                    probs,
                    batch,
                    sqrt=True,
                )
                cos_no_oos = cos
                predicted_oos = sparse_decoupled_oos_probability(probs)
            else:
                cos = sparse_binned_cosine_similarity(probs, batch)
                sqrt_cos = sparse_binned_cosine_similarity(probs, batch, sqrt=True)
                cos_no_oos = sparse_binned_cosine_similarity(
                    probs,
                    batch,
                    include_oos=False,
                )
                predicted_oos = sparse_oos_probability(probs)
            diagnostics = support_diagnostics(
                probs,
                batch,
                tolerance=mass_tolerance,
                relative=relative_mass_tolerance,
                tolerance_min_mz=mass_tolerance_min_mz,
            )
            all_cos.extend(float(x) for x in cos.detach().cpu())
            all_sqrt.extend(float(x) for x in sqrt_cos.detach().cpu())
            all_cos_no_oos.extend(float(x) for x in cos_no_oos.detach().cpu())
            all_predicted_oos.extend(float(x) for x in predicted_oos.detach().cpu())
            all_candidate_coverage.extend(
                float(x) for x in diagnostics['candidate_coverage'].detach().cpu()
            )
            all_oos_target_mass.extend(
                float(x) for x in diagnostics['oos_target_mass'].detach().cpu()
            )
            all_oracle_binned.extend(
                float(x) for x in diagnostics['oracle_binned_cosine'].detach().cpu()
            )
            all_oracle_tolerance.extend(
                float(x) for x in diagnostics['oracle_tolerance_cosine'].detach().cpu()
            )
        sparse_rows = _sparse_prediction_rows(
            probs,
            bin_width=_bin_width_from_batch(batch),
            min_intensity=min_intensity,
            top_k=top_k,
            probability_mode=probability_mode,
        )

        for i, smiles in enumerate(batch['smiles']):
            row = {
                'identifier': batch['identifier'][i],
                'smiles': smiles,
                'pred_peaks': json.dumps(sparse_rows[i]),
            }
            if cos is not None and sqrt_cos is not None and diagnostics is not None:
                row['cosine'] = float(cos[i].detach().cpu())
                row['sqrt_cosine'] = float(sqrt_cos[i].detach().cpu())
                row['cosine_no_oos'] = float(cos_no_oos[i].detach().cpu())
                row['predicted_oos_probability'] = float(
                    predicted_oos[i].detach().cpu()
                )
                row['candidate_coverage'] = float(
                    diagnostics['candidate_coverage'][i].detach().cpu()
                )
                row['oos_target_mass'] = float(
                    diagnostics['oos_target_mass'][i].detach().cpu()
                )
                row['oracle_binned_cosine'] = float(
                    diagnostics['oracle_binned_cosine'][i].detach().cpu()
                )
                row['oracle_tolerance_cosine'] = float(
                    diagnostics['oracle_tolerance_cosine'][i].detach().cpu()
                )
            rows.append(row)

    summary = {
        'cosine_mean': _mean_or_nan(all_cos),
        'sqrt_cosine_mean': _mean_or_nan(all_sqrt),
        'cosine_no_oos_mean': _mean_or_nan(all_cos_no_oos),
        'predicted_oos_probability_mean': _mean_or_nan(all_predicted_oos),
        'candidate_coverage_mean': _mean_or_nan(all_candidate_coverage),
        'oos_target_mass_mean': _mean_or_nan(all_oos_target_mass),
        'oracle_binned_cosine_mean': _mean_or_nan(all_oracle_binned),
        'oracle_tolerance_cosine_mean': _mean_or_nan(all_oracle_tolerance),
    }
    return pd.DataFrame(rows), summary


def probability_mode_from_checkpoint_payload(payload: dict[str, Any]) -> str:
    """
    Select prediction probability semantics from checkpoint training metadata.

    Checkpoints trained with ``LOSS=decoupled_kl`` use a fragment-only softmax
    plus a sigmoid OOS head. Older and standard-loss checkpoints retain the
    joint fragment-plus-OOS softmax semantics.
    """
    train_config = payload.get('train_config')
    if not isinstance(train_config, dict):
        return 'joint'
    mode = train_config.get('prediction_probability_mode')
    if mode in {'joint', 'decoupled'}:
        return str(mode)
    if train_config.get('loss') == 'decoupled_kl':
        return 'decoupled'
    return 'joint'


def support_diagnostics(
    pred: dict[str, Any],
    batch: dict[str, Any],
    *,
    tolerance: float = 0.01,
    relative: bool = False,
    tolerance_min_mz: float = 200.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """
    Compute candidate-support upper bounds for sparse fragment predictions.

    ``candidate_coverage`` is the target intensity fraction whose binned m/z is
    present in the generated candidate support. ``oos_target_mass`` is the
    complementary target mass. The oracle cosine values ignore learned logits and
    ask how well any scorer could do if it placed probability only on reachable
    target peaks, under exact MassSpecGym bins or m/z tolerance matching.
    """
    pred_bins = pred['bins'].long()
    pred_mzs = pred['mzs'].to(device=pred_bins.device, dtype=torch.get_default_dtype())
    pred_batch = pred['batch'].long()
    target_mzs = batch['target_mz'].to(device=pred_bins.device, dtype=pred_mzs.dtype)
    target_values = batch['target_intensity'].to(
        device=pred_bins.device,
        dtype=pred_mzs.dtype,
    )
    target_batch = batch['target_batch'].to(device=pred_bins.device).long()
    target_bins = torch.floor(target_mzs / _bin_width_from_batch(batch)).long()
    batch_size = int(pred['batch_size'])
    num_bins = int(pred['num_bins'])
    candidate_coverage = target_values.new_zeros(batch_size)
    oos_target_mass = target_values.new_zeros(batch_size)
    oracle_binned_cosine = target_values.new_zeros(batch_size)
    oracle_tolerance_cosine = target_values.new_zeros(batch_size)

    for batch_idx in range(batch_size):
        p_mask = pred_batch == batch_idx
        t_mask = target_batch == batch_idx
        if not bool(t_mask.any()):
            continue

        t_bins, t_bin_values = _aggregate_values(
            target_bins[t_mask].clamp(0, num_bins - 1),
            target_values[t_mask],
        )
        total_mass = torch.clamp(t_bin_values.sum(), min=eps)
        target_norm = torch.clamp(torch.linalg.vector_norm(t_bin_values), min=eps)
        if bool(p_mask.any()):
            p_bins = torch.unique(pred_bins[p_mask].clamp(0, num_bins - 1), sorted=True)
            reachable_bins = _membership_mask(t_bins, p_bins)
        else:
            reachable_bins = torch.zeros_like(t_bins, dtype=torch.bool)
        reachable_mass = torch.sum(t_bin_values[reachable_bins])
        reachable_norm = torch.linalg.vector_norm(t_bin_values[reachable_bins])
        candidate_coverage[batch_idx] = reachable_mass / total_mass
        oos_target_mass[batch_idx] = 1.0 - candidate_coverage[batch_idx]
        oracle_binned_cosine[batch_idx] = reachable_norm / target_norm

        t_mzs = target_mzs[t_mask]
        t_values = target_values[t_mask]
        tolerance_target_norm = torch.clamp(torch.linalg.vector_norm(t_values), min=eps)
        if bool(p_mask.any()):
            p_mzs = pred_mzs[p_mask]
            tolerances = _target_tolerances(
                t_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
            reachable_peaks = (
                torch.abs(t_mzs.unsqueeze(1) - p_mzs.unsqueeze(0))
                <= tolerances.unsqueeze(1)
            ).any(dim=1)
        else:
            reachable_peaks = torch.zeros_like(t_mzs, dtype=torch.bool)
        tolerance_reachable_norm = torch.linalg.vector_norm(t_values[reachable_peaks])
        oracle_tolerance_cosine[batch_idx] = (
            tolerance_reachable_norm / tolerance_target_norm
        )

    return {
        'candidate_coverage': candidate_coverage,
        'oos_target_mass': oos_target_mass,
        'oracle_binned_cosine': oracle_binned_cosine,
        'oracle_tolerance_cosine': oracle_tolerance_cosine,
    }


def _membership_mask(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Return whether each sorted value is present in a sorted support tensor.
    """
    if support.numel() == 0 or values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    positions = torch.searchsorted(support, values)
    in_range = positions < support.numel()
    out = torch.zeros_like(values, dtype=torch.bool)
    if bool(in_range.any()):
        in_range_positions = positions[in_range]
        out[in_range] = support[in_range_positions] == values[in_range]
    return out


def _mean_or_nan(values: list[float]) -> float:
    """
    Return the arithmetic mean or NaN for an empty sequence.
    """
    return float(sum(values) / len(values)) if values else float('nan')


def _sparse_prediction_rows(
    pred: dict[str, Any],
    *,
    bin_width: float,
    min_intensity: float,
    top_k: int,
    probability_mode: str = 'joint',
) -> list[dict[str, list[float]]]:
    """
    Convert sparse fragment probabilities into exported peak lists.

    Candidate probabilities are aggregated by bin, optionally truncated to top-k bins, filtered by minimum intensity, normalized to base peak 100, and converted back to bin-center m/z values. ``probability_mode='decoupled'`` applies a fragment-only softmax before filtering.
    """
    if probability_mode == 'decoupled':
        log_probs = _fragment_only_log_probs(pred)
    elif probability_mode == 'joint':
        log_probs, _oos_log_probs = fragment_oos_log_probs(pred)
    else:
        raise ValueError("probability_mode must be one of: 'joint', 'decoupled'.")
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
