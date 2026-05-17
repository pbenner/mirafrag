import numpy as np
import torch

from mirafrag.spectra import (
    MASS_SPEC_GYM_BIN_WIDTH,
    MASS_SPEC_GYM_MZ_MAX,
    MASS_SPEC_GYM_NUM_BINS,
    bin_spectrum,
    cosine_similarity,
    num_spectrum_bins,
    parse_peaks,
    peaks_from_bins,
)


def test_parse_massspecgym_peaks():
    row = {'mzs': '10,20,20', 'intensities': '1,2,3'}
    mzs, ints = parse_peaks(row)
    assert mzs.tolist() == [10.0, 20.0, 20.0]
    assert ints.tolist() == [1.0, 2.0, 3.0]


def test_bin_spectrum_l1():
    binned = bin_spectrum(
        np.array([10.2, 10.8, 12.1]), np.array([1, 3, 2]), mz_max=20, bin_width=1
    )
    assert torch.isclose(binned.sum(), torch.tensor(1.0))
    assert torch.isclose(binned[10], torch.tensor(4 / 6))
    assert torch.isclose(binned[12], torch.tensor(2 / 6))


def test_bin_spectrum_massspecgym_defaults():
    binned = bin_spectrum(
        np.array([0.0, 1004.999, 1005.0]),
        np.array([1.0, 2.0, 4.0]),
        normalize='none',
    )
    assert binned.shape == (MASS_SPEC_GYM_NUM_BINS,)
    assert num_spectrum_bins(MASS_SPEC_GYM_MZ_MAX, MASS_SPEC_GYM_BIN_WIDTH) == 100500
    assert torch.isclose(binned[0], torch.tensor(1.0))
    assert torch.isclose(binned[-1], torch.tensor(2.0))


def test_cosine_and_peaks_from_bins():
    pred = torch.tensor([[0.0, 0.25, 0.75]])
    target = torch.tensor([[0.0, 0.25, 0.75]])
    assert torch.isclose(cosine_similarity(pred, target), torch.tensor([1.0])).all()
    peaks = peaks_from_bins(pred[0], bin_width=1.0, min_intensity=0.1, top_k=2)
    assert peaks['mz'] == [1.5, 2.5]
    assert peaks['intensity'][-1] == 100.0
