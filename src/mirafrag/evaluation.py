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
    sparse_binned_cosine_similarity,
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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Evaluate a MiraFrag model and return predictions plus summary metrics.

    The function runs the model in evaluation mode, computes sparse binned cosine metrics when targets are present, converts probabilities to sparse peak rows, and returns a dataframe-ready result.
    """
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
    """
    Convert sparse fragment probabilities into exported peak lists.

    Candidate probabilities are aggregated by bin, optionally truncated to top-k bins, filtered by minimum intensity, normalized to base peak 100, and converted back to bin-center m/z values.
    """
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
