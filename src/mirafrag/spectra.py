from __future__ import annotations

import ast
import json
import math
from collections.abc import Iterable

import numpy as np
import torch

MASS_SPEC_GYM_MZ_MAX = 1005.0
MASS_SPEC_GYM_BIN_WIDTH = 0.01
MASS_SPEC_GYM_NUM_BINS = 100500


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def num_spectrum_bins(mz_max: float, bin_width: float) -> int:
    return int(math.ceil(float(mz_max) / float(bin_width)))


def parse_number_list(value) -> list[float]:
    if _is_missing(value):
        values = []
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            values = []
        elif text.startswith('['):
            try:
                values = json.loads(text)
            except Exception:
                values = ast.literal_eval(text)
        else:
            values = [x for x in text.split(',') if x.strip()]
    return [float(x) for x in values]


def parse_peaks(
    row,
    *,
    mzs_col: str = 'mzs',
    intensities_col: str = 'intensities',
    peaks_col: str = 'peaks',
) -> tuple[np.ndarray, np.ndarray]:
    if peaks_col in row and not _is_missing(row[peaks_col]):
        peaks = row[peaks_col]
        if isinstance(peaks, str):
            try:
                peaks = json.loads(peaks)
            except Exception:
                peaks = ast.literal_eval(peaks)
        mzs = peaks.get('mz', peaks.get('mzs'))
        intensities = peaks.get('intensity', peaks.get('intensities'))
    else:
        mzs = row[mzs_col]
        intensities = row[intensities_col]

    mz_arr = np.asarray(parse_number_list(mzs), dtype=np.float32)
    int_arr = np.asarray(parse_number_list(intensities), dtype=np.float32)
    if mz_arr.shape != int_arr.shape:
        raise ValueError(
            f'm/z and intensity lengths differ: {len(mz_arr)} != {len(int_arr)}'
        )
    mask = np.isfinite(mz_arr) & np.isfinite(int_arr) & (int_arr > 0)
    return mz_arr[mask], int_arr[mask]


def bin_spectrum(
    mzs: np.ndarray | Iterable[float],
    intensities: np.ndarray | Iterable[float],
    *,
    mz_max: float = MASS_SPEC_GYM_MZ_MAX,
    bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
    normalize: str = 'l1',
) -> torch.Tensor:
    mz_arr = np.asarray(list(mzs), dtype=np.float32)
    int_arr = np.asarray(list(intensities), dtype=np.float32)
    num_bins = num_spectrum_bins(mz_max, bin_width)
    out = np.zeros(num_bins, dtype=np.float32)
    if mz_arr.size == 0:
        return torch.from_numpy(out)

    bins = np.floor(mz_arr / float(bin_width)).astype(np.int64)
    mask = (bins >= 0) & (bins < num_bins) & np.isfinite(int_arr) & (int_arr > 0)
    np.add.at(out, bins[mask], int_arr[mask])

    if normalize == 'l1':
        denom = float(out.sum())
    elif normalize == 'max':
        denom = float(out.max())
    elif normalize == 'none':
        denom = 1.0
    else:
        raise ValueError(f'Unknown normalization: {normalize}')
    if denom > 0:
        out = out / denom
    return torch.from_numpy(out)


def cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sqrt: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    if sqrt:
        pred = torch.sqrt(torch.clamp(pred, min=0.0))
        target = torch.sqrt(torch.clamp(target, min=0.0))
    dot = (pred * target).sum(dim=-1)
    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    return dot / torch.clamp(pred_norm * target_norm, min=eps)


def peaks_from_bins(
    probs: torch.Tensor,
    *,
    bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
    min_intensity: float = 0.001,
    top_k: int | None = 100,
) -> dict[str, list[float]]:
    values = probs.detach().float().cpu()
    if values.ndim != 1:
        raise ValueError('peaks_from_bins expects a single spectrum vector.')
    if top_k is not None and top_k > 0 and top_k < values.numel():
        top_values, top_idx = torch.topk(values, k=top_k)
        mask = top_values >= float(min_intensity)
        idx = top_idx[mask]
        intensities = top_values[mask]
    else:
        idx = torch.nonzero(values >= float(min_intensity), as_tuple=False).flatten()
        intensities = values[idx]
    order = torch.argsort(idx)
    idx = idx[order]
    intensities = intensities[order]
    mzs = (idx.float() + 0.5) * float(bin_width)
    if intensities.numel() > 0:
        intensities = intensities / torch.clamp(intensities.max(), min=1e-12) * 100.0
    return {
        'mz': [round(float(x), 6) for x in mzs.tolist()],
        'intensity': [round(float(x), 6) for x in intensities.tolist()],
    }
