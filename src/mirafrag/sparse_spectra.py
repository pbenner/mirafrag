from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mirafrag.spectra import MASS_SPEC_GYM_BIN_WIDTH, MASS_SPEC_GYM_MZ_MAX


@dataclass(frozen=True)
class SparseSpectrum:
    bins: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'bins', np.asarray(self.bins, dtype=np.int64))
        object.__setattr__(self, 'values', np.asarray(self.values, dtype=np.float32))


def sparse_from_peaks(
    mzs,
    intensities,
    *,
    mz_max: float = MASS_SPEC_GYM_MZ_MAX,
    bin_width: float = MASS_SPEC_GYM_BIN_WIDTH,
) -> SparseSpectrum:
    mz_arr = np.asarray(mzs, dtype=np.float32)
    value_arr = np.asarray(intensities, dtype=np.float32)
    if mz_arr.shape != value_arr.shape:
        raise ValueError('m/z and intensity arrays must have the same shape.')
    num_bins = int(math.ceil(float(mz_max) / float(bin_width)))
    bins = np.floor(mz_arr / float(bin_width)).astype(np.int64)
    mask = (
        np.isfinite(mz_arr)
        & np.isfinite(value_arr)
        & (value_arr > 0.0)
        & (bins >= 0)
        & (bins < num_bins)
    )
    return normalize_sparse_spectrum(
        SparseSpectrum(bins=bins[mask], values=value_arr[mask])
    )


def normalize_sparse_spectrum(spectrum: SparseSpectrum) -> SparseSpectrum:
    bins = np.asarray(spectrum.bins, dtype=np.int64)
    values = np.asarray(spectrum.values, dtype=np.float32)
    if bins.size == 0:
        return SparseSpectrum(bins=bins, values=values)
    order = np.argsort(bins)
    bins = bins[order]
    values = values[order]
    unique, inverse = np.unique(bins, return_inverse=True)
    out = np.zeros(unique.shape[0], dtype=np.float32)
    np.add.at(out, inverse, values)
    denom = float(out.sum())
    if denom > 0.0:
        out = out / denom
    return SparseSpectrum(bins=unique.astype(np.int64), values=out)


def scale_sparse_spectrum(spectrum: SparseSpectrum, weight: float) -> SparseSpectrum:
    return SparseSpectrum(
        bins=np.asarray(spectrum.bins, dtype=np.int64),
        values=np.asarray(spectrum.values, dtype=np.float32) * float(weight),
    )


def combine_sparse_spectra(spectra: Sequence[SparseSpectrum]) -> SparseSpectrum:
    non_empty = [spectrum for spectrum in spectra if spectrum.values.size > 0]
    if not non_empty:
        return SparseSpectrum(
            np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
        )
    return normalize_sparse_spectrum(
        SparseSpectrum(
            bins=np.concatenate([spectrum.bins for spectrum in non_empty]),
            values=np.concatenate([spectrum.values for spectrum in non_empty]),
        )
    )


def sparse_cosine(
    left: SparseSpectrum,
    right: SparseSpectrum,
    *,
    sqrt: bool = False,
    eps: float = 1e-12,
) -> float:
    if left.values.size == 0 or right.values.size == 0:
        return 0.0
    left_values = np.sqrt(left.values) if sqrt else left.values
    right_values = np.sqrt(right.values) if sqrt else right.values
    dot = _sparse_dot(left.bins, left_values, right.bins, right_values)
    denom = max(float(np.linalg.norm(left_values) * np.linalg.norm(right_values)), eps)
    return float(dot / denom)


def sparse_jensen_shannon_similarity(
    left: SparseSpectrum,
    right: SparseSpectrum,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute Jensen-Shannon similarity for two sparse binned spectra.

    The score follows the MassSpecGym/FraGNNet convention: both spectra are
    treated as L1-normalized distributions, JSD is computed with natural logs,
    and similarity is ``1 - JSD / log(2)``. Non-overlapping non-empty spectra
    therefore score 0, identical spectra score 1.
    """
    left_norm = normalize_sparse_spectrum(left)
    right_norm = normalize_sparse_spectrum(right)
    if left_norm.values.size == 0 and right_norm.values.size == 0:
        return 1.0
    if left_norm.values.size == 0 or right_norm.values.size == 0:
        return 0.0

    left_bins = np.asarray(left_norm.bins, dtype=np.int64)
    right_bins = np.asarray(right_norm.bins, dtype=np.int64)
    left_values = np.asarray(left_norm.values, dtype=np.float64)
    right_values = np.asarray(right_norm.values, dtype=np.float64)
    union = np.union1d(left_bins, right_bins)
    left_dense = np.zeros(union.shape[0], dtype=np.float64)
    right_dense = np.zeros(union.shape[0], dtype=np.float64)
    left_pos = np.searchsorted(union, left_bins)
    right_pos = np.searchsorted(union, right_bins)
    left_dense[left_pos] = left_values
    right_dense[right_pos] = right_values
    mixture = 0.5 * (left_dense + right_dense)

    left_mask = left_dense > 0.0
    right_mask = right_dense > 0.0
    kl_left = float(
        np.sum(
            left_dense[left_mask]
            * (
                np.log(np.maximum(left_dense[left_mask], eps))
                - np.log(np.maximum(mixture[left_mask], eps))
            )
        )
    )
    kl_right = float(
        np.sum(
            right_dense[right_mask]
            * (
                np.log(np.maximum(right_dense[right_mask], eps))
                - np.log(np.maximum(mixture[right_mask], eps))
            )
        )
    )
    jsd = 0.5 * (kl_left + kl_right)
    score = 1.0 - jsd / math.log(2.0)
    if abs(score) < 1e-12:
        score = 0.0
    if abs(score - 1.0) < 1e-12:
        score = 1.0
    return float(max(0.0, min(1.0, score)))


def _sparse_dot(
    left_bins: np.ndarray,
    left_values: np.ndarray,
    right_bins: np.ndarray,
    right_values: np.ndarray,
) -> float:
    i = j = 0
    total = 0.0
    while i < left_bins.size and j < right_bins.size:
        if left_bins[i] == right_bins[j]:
            total += float(left_values[i] * right_values[j])
            i += 1
            j += 1
        elif left_bins[i] < right_bins[j]:
            i += 1
        else:
            j += 1
    return total
