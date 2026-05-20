# ruff: noqa: F401
import math
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.nn import LazyLinear
from torch.utils.data import DataLoader

from mirafrag.checkpoint import load_checkpoint
from mirafrag.chem import GraphConfig
from mirafrag.cli.cache import (
    _apply_fragment_args_to_model_config as _apply_cache_fragment_args_to_model_config,
)
from mirafrag.cli.train import (
    _apply_fragment_args_to_model_config as _apply_train_fragment_args_to_model_config,
)
from mirafrag.config import MiraFragConfig
from mirafrag.data import (
    BinnedSpectrumDataset,
    MetadataConfig,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    select_split,
)
from mirafrag.encoders.mace import repair_mace_cuequivariance_config
from mirafrag.evaluation import _sparse_prediction_rows, support_diagnostics
from mirafrag.fragments import (
    FRAGMENT_EDGE_FEATURE_DIM,
    PROTON_MASS,
    SODIUM_ADDUCT_MASS,
    FragmentConfig,
    collate_fragment_candidates,
    fragment_config_from_model_config,
    parse_fragment_adduct,
    smiles_to_fragment_candidates,
)
from mirafrag.losses import (
    LOSS_NAMES,
    fragnnet_sparse_cross_entropy,
    projected_sparse_binned_kl_divergence,
    soft_binned_coverage_kl_divergence,
    soft_binned_kl_divergence,
    soft_projected_sparse_kl_divergence,
    sparse_binned_cosine_similarity,
    sparse_binned_kl_divergence,
    sparse_oos_probability,
    sparse_prediction_entropy,
    spectrum_loss,
)
from mirafrag.model import MiraFragModel, set_encoder_finetune_strategy
from mirafrag.optim import (
    _build_scheduler,
    _optimizer_param_groups,
    _scheduler_total_steps,
)
from mirafrag.training import train_model
from tests.helpers import (
    FakeChargeEncoder,
    FakeCueProduct,
    FakeMace,
    FakeTrainAwareEncoder,
    _tiny_loader,
    _tiny_training_df,
)


def test_loss_registry_exposes_cli_choices():
    assert 'kl' in LOSS_NAMES
    assert 'fragnnet_ce' in LOSS_NAMES
    assert 'cosine' in LOSS_NAMES


def test_fragment_sparse_loss_with_fake_mace():
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    dataset = BinnedSpectrumDataset(
        _tiny_training_df(),
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_config=FragmentConfig(max_tree_depth=2, max_fragments=16),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_spectrum_batch,
    )
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=64,
            hidden_dim=8,
            metadata_dim=4,
            max_fragment_tree_depth=2,
            max_fragments=16,
        ),
    )
    batch = next(iter(loader))
    pred = model(batch)
    loss = spectrum_loss(pred, batch, loss='cosine')
    kl_loss = spectrum_loss(pred, batch, loss='kl')
    hybrid_loss = spectrum_loss(pred, batch, loss='kl_cosine', kl_weight=0.7)
    cosine = sparse_binned_cosine_similarity(pred, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(kl_loss)
    assert torch.isfinite(hybrid_loss)
    assert cosine.shape == (2,)


def test_binned_cosine_penalizes_oos_probability():
    batch = {
        'target_mz': torch.tensor([0.5]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([1.0]),
    }
    base_pred = {
        'logits': torch.tensor([0.0]),
        'bins': torch.tensor([0]),
        'batch': torch.tensor([0]),
        'batch_size': 1,
        'num_bins': 8,
    }
    low_oos = dict(base_pred, oos_logits=torch.tensor([-20.0]))
    high_oos = dict(base_pred, oos_logits=torch.tensor([0.0]))

    assert sparse_binned_cosine_similarity(low_oos, batch).item() > 0.99
    assert sparse_binned_cosine_similarity(high_oos, batch).item() < 0.72


def test_sparse_binned_kl_aggregates_duplicate_prediction_bins():
    pred = {
        'logits': torch.log(torch.tensor([0.125, 0.125, 0.75])),
        'oos_logits': torch.tensor([-100.0]),
        'bins': torch.tensor([1, 1, 2]),
        'batch': torch.tensor([0, 0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    batch = {
        'target_mz': torch.tensor([1.2, 2.2]),
        'target_intensity': torch.tensor([0.25, 0.75]),
        'target_batch': torch.tensor([0, 0]),
        'bin_width': torch.tensor([1.0]),
    }
    loss = sparse_binned_kl_divergence(pred, batch)
    assert loss.shape == (1,)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_reported_cosine_penalizes_oos_probability():
    batch = {
        'target_mz': torch.tensor([1.2, 2.2]),
        'target_intensity': torch.tensor([0.25, 0.75]),
        'target_batch': torch.tensor([0, 0]),
        'bin_width': torch.tensor([1.0]),
    }
    pred_low_oos = {
        'logits': torch.log(torch.tensor([0.25, 0.75])),
        'oos_logits': torch.tensor([-10.0]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    pred_high_oos = {
        **pred_low_oos,
        'oos_logits': torch.tensor([10.0]),
    }

    low_oos_cosine = sparse_binned_cosine_similarity(pred_low_oos, batch)
    high_oos_cosine = sparse_binned_cosine_similarity(pred_high_oos, batch)

    assert torch.allclose(low_oos_cosine, torch.ones_like(low_oos_cosine))
    assert high_oos_cosine.item() < 1e-4


def test_prediction_rows_aggregate_duplicate_bins_before_top_k():
    pred = {
        'logits': torch.log(torch.tensor([0.25, 0.35, 0.40])),
        'bins': torch.tensor([1, 1, 2]),
        'batch': torch.tensor([0, 0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }

    rows = _sparse_prediction_rows(
        pred,
        bin_width=1.0,
        min_intensity=0.0,
        top_k=1,
    )

    assert rows == [{'mz': [1.5], 'intensity': [100.0]}]


def test_support_diagnostics_report_candidate_oracle_bounds():
    pred = {
        'mzs': torch.tensor([1.5, 3.5]),
        'bins': torch.tensor([1, 3]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 5,
    }
    batch = {
        'target_mz': torch.tensor([1.5, 2.5, 3.5]),
        'target_intensity': torch.tensor([1.0, 2.0, 2.0]),
        'target_batch': torch.tensor([0, 0, 0]),
        'bin_width': torch.tensor([1.0]),
    }

    diagnostics = support_diagnostics(pred, batch, tolerance=0.01)

    assert torch.allclose(diagnostics['candidate_coverage'], torch.tensor([0.6]))
    assert torch.allclose(diagnostics['oos_target_mass'], torch.tensor([0.4]))
    expected_oracle = torch.sqrt(torch.tensor([5.0])) / 3.0
    assert torch.allclose(diagnostics['oracle_binned_cosine'], expected_oracle)
    assert torch.allclose(diagnostics['oracle_tolerance_cosine'], expected_oracle)


def test_projected_kl_assigns_unreachable_target_bins_to_oos():
    target_probs = torch.tensor([0.25, 0.75, 10.0]) / 11.0
    pred = {
        'logits': torch.log(target_probs[:2]),
        'oos_logits': torch.log(target_probs[2:]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 5,
    }
    batch = {
        'target_mz': torch.tensor([1.2, 2.2, 4.2]),
        'target_intensity': torch.tensor([0.25, 0.75, 10.0]),
        'target_batch': torch.tensor([0, 0, 0]),
        'bin_width': torch.tensor([1.0]),
    }
    loss = projected_sparse_binned_kl_divergence(pred, batch)
    assert loss.shape == (1,)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_projected_kl_trains_oos_when_no_target_bin_is_reachable():
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    oos_logits = torch.tensor([0.0], requires_grad=True)
    pred = {
        'logits': logits,
        'oos_logits': oos_logits,
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 5,
    }
    batch = {
        'target_mz': torch.tensor([4.2]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([1.0]),
    }
    loss = projected_sparse_binned_kl_divergence(pred, batch)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert logits.grad is not None
    assert oos_logits.grad is not None
    assert float(oos_logits.grad[0]) < 0.0


def test_soft_projected_kl_prefers_nearby_candidate():
    batch = {
        'target_mz': torch.tensor([1.0]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
    }
    pred_good = {
        'logits': torch.tensor([3.0, 0.0]),
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.0, 2.0]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    pred_bad = {
        **pred_good,
        'logits': torch.tensor([0.0, 3.0]),
    }

    good_loss = soft_projected_sparse_kl_divergence(
        pred_good,
        batch,
        tolerance=0.1,
    )
    bad_loss = soft_projected_sparse_kl_divergence(
        pred_bad,
        batch,
        tolerance=0.1,
    )

    assert good_loss.shape == (1,)
    assert float(good_loss) < float(bad_loss)


def test_soft_projected_kl_has_gradient_without_exact_bin_match():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    pred = {
        'logits': logits,
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.0, 2.0]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    batch = {
        'target_mz': torch.tensor([1.4]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
    }
    loss = spectrum_loss(
        pred,
        batch,
        loss='soft_projected_kl',
        mass_tolerance=0.2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_binned_coverage_kl_prefers_nearby_probability():
    batch = {
        'target_mz': torch.tensor([1.04]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([0.01]),
    }
    pred_good = {
        'logits': torch.tensor([3.0, 0.0]),
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.04, 1.20]),
        'bins': torch.tensor([104, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    pred_bad = {
        **pred_good,
        'logits': torch.tensor([0.0, 3.0]),
    }

    good_loss = soft_binned_coverage_kl_divergence(
        pred_good,
        batch,
        tolerance=0.03,
        coverage_weight=0.5,
    )
    bad_loss = soft_binned_coverage_kl_divergence(
        pred_bad,
        batch,
        tolerance=0.03,
        coverage_weight=0.5,
    )

    assert good_loss.shape == (1,)
    assert float(good_loss) < float(bad_loss)


def test_soft_binned_kl_prefers_nearby_probability():
    batch = {
        'target_mz': torch.tensor([1.04]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([0.01]),
    }
    pred_good = {
        'logits': torch.tensor([3.0, 0.0]),
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.04, 1.20]),
        'bins': torch.tensor([104, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    pred_bad = {
        **pred_good,
        'logits': torch.tensor([0.0, 3.0]),
    }

    good_loss = soft_binned_kl_divergence(
        pred_good,
        batch,
        tolerance=0.03,
    )
    bad_loss = soft_binned_kl_divergence(
        pred_bad,
        batch,
        tolerance=0.03,
    )

    assert good_loss.shape == (1,)
    assert float(good_loss) < float(bad_loss)


def test_soft_binned_kl_has_gradient_across_neighboring_bins():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    pred = {
        'logits': logits,
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.045, 1.20]),
        'bins': torch.tensor([104, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    batch = {
        'target_mz': torch.tensor([1.055]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([0.01]),
    }
    loss = spectrum_loss(
        pred,
        batch,
        loss='soft_binned_kl',
        mass_tolerance=0.03,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0]) < 0.0


def test_fragnnet_ce_prefers_nearby_probability():
    batch = {
        'target_mz': torch.tensor([100.0]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
    }
    pred_good = {
        'logits': torch.tensor([3.0, 0.0]),
        'oos_logits': torch.tensor([-3.0]),
        'mzs': torch.tensor([100.0, 120.0]),
        'bins': torch.tensor([100, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    pred_bad = {
        **pred_good,
        'logits': torch.tensor([0.0, 3.0]),
    }

    good_loss = fragnnet_sparse_cross_entropy(
        pred_good,
        batch,
        tolerance=0.01,
        relative=False,
    )
    bad_loss = fragnnet_sparse_cross_entropy(
        pred_bad,
        batch,
        tolerance=0.01,
        relative=False,
    )

    assert good_loss.shape == (1,)
    assert float(good_loss) < float(bad_loss)


def test_fragnnet_ce_uses_oos_probability_for_unmatched_peaks():
    batch = {
        'target_mz': torch.tensor([150.0]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
    }
    pred_low_oos = {
        'logits': torch.tensor([3.0]),
        'oos_logits': torch.tensor([-3.0]),
        'mzs': torch.tensor([100.0]),
        'bins': torch.tensor([100]),
        'batch': torch.tensor([0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    pred_high_oos = {
        **pred_low_oos,
        'oos_logits': torch.tensor([3.0]),
    }

    low_oos_loss = fragnnet_sparse_cross_entropy(
        pred_low_oos,
        batch,
        tolerance=0.01,
        relative=False,
    )
    high_oos_loss = fragnnet_sparse_cross_entropy(
        pred_high_oos,
        batch,
        tolerance=0.01,
        relative=False,
    )

    assert float(high_oos_loss) < float(low_oos_loss)


def test_fragnnet_ce_backpropagates_to_fragment_and_oos_logits():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    oos_logits = torch.tensor([0.0], requires_grad=True)
    pred = {
        'logits': logits,
        'oos_logits': oos_logits,
        'mzs': torch.tensor([100.0, 120.0]),
        'bins': torch.tensor([100, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    batch = {
        'target_mz': torch.tensor([100.0, 180.0]),
        'target_intensity': torch.tensor([1.0, 1.0]),
        'target_batch': torch.tensor([0, 0]),
    }
    loss = spectrum_loss(
        pred,
        batch,
        loss='fragnnet_ce',
        mass_tolerance=0.01,
        relative_mass_tolerance=False,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert oos_logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(oos_logits.grad).all()


def test_soft_binned_coverage_kl_has_gradient_across_neighboring_bins():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    pred = {
        'logits': logits,
        'oos_logits': torch.tensor([-10.0]),
        'mzs': torch.tensor([1.045, 1.20]),
        'bins': torch.tensor([104, 120]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 200,
    }
    batch = {
        'target_mz': torch.tensor([1.055]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([0.01]),
    }
    loss = spectrum_loss(
        pred,
        batch,
        loss='soft_binned_coverage_kl',
        mass_tolerance=0.03,
        coverage_weight=0.5,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0]) < 0.0


def test_kl_cosine_loss_matches_weighted_components():
    pred = {
        'logits': torch.tensor([0.0, 1.0, -1.0]),
        'oos_logits': torch.tensor([-10.0]),
        'bins': torch.tensor([1, 2, 3]),
        'batch': torch.tensor([0, 0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    batch = {
        'target_mz': torch.tensor([1.2, 2.2]),
        'target_intensity': torch.tensor([0.25, 0.75]),
        'target_batch': torch.tensor([0, 0]),
        'bin_width': torch.tensor([1.0]),
    }
    kl_weight = 0.7
    kl_loss = spectrum_loss(pred, batch, loss='kl')
    cosine_loss = spectrum_loss(pred, batch, loss='cosine')
    hybrid_loss = spectrum_loss(
        pred,
        batch,
        loss='kl_cosine',
        kl_weight=kl_weight,
    )
    expected = kl_weight * kl_loss + (1.0 - kl_weight) * cosine_loss
    assert torch.allclose(hybrid_loss, expected)


def test_sparse_oos_probability_and_cosine_without_oos():
    batch = {
        'target_mz': torch.tensor([1.2]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([1.0]),
    }
    pred = {
        'logits': torch.tensor([0.0]),
        'oos_logits': torch.tensor([0.0]),
        'bins': torch.tensor([1]),
        'batch': torch.tensor([0]),
        'batch_size': 1,
        'num_bins': 4,
    }

    assert torch.allclose(sparse_oos_probability(pred), torch.tensor([0.5]))
    assert sparse_binned_cosine_similarity(pred, batch).item() < 1.0
    assert torch.allclose(
        sparse_binned_cosine_similarity(pred, batch, include_oos=False),
        torch.ones(1),
    )


def test_sparse_prediction_entropy_regularizer_adds_to_loss():
    pred = {
        'logits': torch.tensor([0.0, 0.0]),
        'oos_logits': torch.tensor([0.0]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }
    batch = {
        'target_mz': torch.tensor([1.2]),
        'target_intensity': torch.tensor([1.0]),
        'target_batch': torch.tensor([0]),
        'bin_width': torch.tensor([1.0]),
    }

    base = spectrum_loss(pred, batch, loss='kl')
    regularized = spectrum_loss(pred, batch, loss='kl', entropy_weight=0.1)

    assert sparse_prediction_entropy(pred).item() > 0.0
    assert regularized > base


def test_kl_target_power_emphasizes_intense_peaks():
    batch = {
        'target_mz': torch.tensor([1.2, 2.2]),
        'target_intensity': torch.tensor([0.1, 0.9]),
        'target_batch': torch.tensor([0, 0]),
        'bin_width': torch.tensor([1.0]),
    }
    pred = {
        'logits': torch.log(torch.tensor([0.5, 0.5])),
        'oos_logits': torch.tensor([-100.0]),
        'bins': torch.tensor([1, 2]),
        'batch': torch.tensor([0, 0]),
        'batch_size': 1,
        'num_bins': 4,
    }

    standard = sparse_binned_kl_divergence(pred, batch, target_power=1.0)
    sharpened = sparse_binned_kl_divergence(pred, batch, target_power=2.0)

    assert sharpened > standard


def test_sparse_binned_kl_aggregates_unreachable_bins_into_single_oos_event():
    pred = {
        'logits': torch.log(torch.tensor([0.2])),
        'oos_logits': torch.log(torch.tensor([0.8])),
        'bins': torch.tensor([1]),
        'batch': torch.tensor([0]),
        'batch_size': 1,
        'num_bins': 8,
    }
    batch = {
        'target_mz': torch.tensor([4.2, 5.2]),
        'target_intensity': torch.tensor([0.5, 0.5]),
        'target_batch': torch.tensor([0, 0]),
        'bin_width': torch.tensor([1.0]),
    }

    loss = sparse_binned_kl_divergence(pred, batch)

    assert loss.item() >= 0.0
    assert torch.allclose(loss, -torch.log(torch.tensor([0.8])), atol=1e-6)
