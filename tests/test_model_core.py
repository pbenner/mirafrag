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
from mirafrag.evaluation import _sparse_prediction_rows
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
from mirafrag.heads.fragment import FragmentSpectrumHead
from mirafrag.losses import (
    LOSS_NAMES,
    fragnnet_sparse_cross_entropy,
    projected_sparse_binned_kl_divergence,
    soft_binned_coverage_kl_divergence,
    soft_binned_kl_divergence,
    soft_projected_sparse_kl_divergence,
    sparse_binned_cosine_similarity,
    sparse_binned_kl_divergence,
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


def test_mirafrag_forward_with_fake_mace():
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
    config = MiraFragConfig(num_bins=64, hidden_dim=8, metadata_dim=4)
    model = MiraFragModel(FakeMace(), metadata_config=metadata, config=config)
    pred = model(next(iter(loader)))
    assert pred['kind'] == 'sparse'
    assert pred['batch_size'] == 2
    assert pred['logits'].ndim == 1
    assert pred['oos_logits'].shape == (2,)
    assert pred['edge_index'].shape[0] == 2
    assert pred['edge_attr'].shape[1] == FRAGMENT_EDGE_FEATURE_DIM


def test_mirafrag_derives_encoder_charge_from_adduct():
    metadata = MetadataConfig(
        adduct_to_idx={'[M+H]+': 0, '[M-H]-': 1, '[M+2H]2+': 2},
        instrument_to_idx={'HCD': 0},
    )
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=16, hidden_dim=8, metadata_dim=4),
    )
    charge = model._molecular_charge(
        {'adduct': torch.tensor([0, 1, 2, 3], dtype=torch.long)}
    )
    assert torch.equal(charge.cpu(), torch.tensor([1.0, -1.0, 2.0, 0.0]))

    raw_charge = model._molecular_charge(
        {
            'adduct': torch.tensor([3], dtype=torch.long),
            'adduct_charge': torch.tensor([1.0], dtype=torch.float32),
        }
    )
    assert torch.equal(raw_charge.cpu(), torch.tensor([1.0]))


def test_mirafrag_passes_charge_only_to_charge_aware_encoder():
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M-H]-': 0}, instrument_to_idx={'HCD': 0})
    dataset = BinnedSpectrumDataset(
        pd.DataFrame(
            {
                'smiles': ['CCO'],
                'adduct': ['[M-H]-'],
                'instrument_type': ['HCD'],
                'collision_energy': [20.0],
                'precursor_mz': [47.0],
                'peaks': ['[(31.0, 1.0)]'],
            }
        ),
        graph_config=graph_config,
        metadata_config=metadata,
        require_spectrum=False,
        include_fragments=False,
    )
    batch = collate_spectrum_batch([dataset[0]])
    encoder = FakeChargeEncoder()
    model = MiraFragModel(
        encoder,
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=16, hidden_dim=8, metadata_dim=4),
    )
    model._encode_node_features(
        batch['graph'],
        molecular_charge=model._molecular_charge(batch),
    )
    assert torch.equal(encoder.last_molecular_charge.cpu(), torch.tensor([-1.0]))


def test_charge_aware_encoder_requires_charge_for_direct_encoding():
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    dataset = BinnedSpectrumDataset(
        pd.DataFrame(
            {
                'smiles': ['CCO'],
                'adduct': ['[M+H]+'],
                'instrument_type': ['HCD'],
                'collision_energy': [20.0],
                'precursor_mz': [47.0],
            }
        ),
        graph_config=graph_config,
        metadata_config=metadata,
        require_spectrum=False,
        include_fragments=False,
    )
    batch = collate_spectrum_batch([dataset[0]])
    model = MiraFragModel(
        FakeChargeEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=16, hidden_dim=8, metadata_dim=4),
    )

    try:
        model.encode_node_features(batch['graph'])
    except ValueError as exc:
        assert 'molecular_charge is required' in str(exc)
    else:
        raise AssertionError('Expected charge-aware encoder to require charge.')


def test_head_strategy_keeps_encoder_in_eval_mode():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeTrainAwareEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='head',
        ),
    )

    model.train()
    assert model.training
    assert not model.encoder.training

    full_model = MiraFragModel(
        FakeTrainAwareEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='full',
        ),
    )
    full_model.train()
    assert full_model.encoder.training


def test_fragment_head_with_fake_mace():
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
    config = MiraFragConfig(
        num_bins=64,
        hidden_dim=8,
        metadata_dim=4,
        max_fragment_tree_depth=2,
        max_fragments=16,
    )
    model = MiraFragModel(FakeMace(), metadata_config=metadata, config=config)
    pred = model(next(iter(loader)))
    assert pred['kind'] == 'sparse'
    assert pred['batch_size'] == 2
    assert pred['logits'].ndim == 1
    assert pred['mzs'].shape == pred['logits'].shape
    assert pred['oos_logits'].shape == (2,)
    assert pred['edge_index'].shape[0] == 2
    assert pred['edge_attr'].shape[1] == FRAGMENT_EDGE_FEATURE_DIM
    assert torch.isfinite(pred['logits']).all()
    assert torch.isfinite(pred['oos_logits']).all()


def test_fragment_path_scorer_starts_as_noop_and_can_train_output_layer():
    torch.manual_seed(11)
    config = MiraFragConfig(
        num_bins=16,
        hidden_dim=8,
        metadata_dim=4,
        dropout=0.0,
        fragment_path_layers=3,
    )
    head = FragmentSpectrumHead(config)
    formula_features = torch.randn(4, 8)
    context_features = torch.randn(4, 8)
    collision_features = torch.randn(4, 8)
    formula_batch = torch.zeros(4, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.zeros(3, FRAGMENT_EDGE_FEATURE_DIM)
    edge_attr[:, 0] = 1.0

    delta = head._fragment_path_delta(
        formula_features,
        context_features,
        collision_features,
        edge_index,
        edge_attr,
        formula_batch,
        batch_size=1,
    )

    assert torch.allclose(delta, torch.zeros_like(delta))
    base_logits = torch.randn_like(delta)
    loss = (base_logits + delta).square().sum()
    loss.backward()

    final_layer = head.fragment_path_residual[-1]
    assert isinstance(final_layer, nn.Linear)
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad).item() > 0


def test_fragment_path_propagation_starts_only_from_retained_roots():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            dropout=0.0,
            fragment_path_layers=2,
        )
    )
    root_scores = torch.tensor([0.0, 1.0, 2.0])
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 2], dtype=torch.long)
    edge_scores = torch.tensor([0.5, 0.7])

    root_frontier = head._fragment_path_root_frontier(root_scores, dst)
    path_scores = head._propagate_fragment_path_scores(
        root_frontier,
        edge_scores,
        src,
        dst,
        num_nodes=3,
    )

    assert torch.isclose(root_frontier[0], torch.tensor(0.0))
    assert (root_frontier[1:] < -1.0e8).all()
    assert torch.allclose(path_scores, torch.tensor([0.0, 0.5, 1.2]))


def test_fragment_path_gradients_stay_finite_with_unreachable_frontier_nodes():
    torch.manual_seed(31)
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            dropout=0.0,
            fragment_path_layers=3,
        )
    )
    final_layer = head.fragment_path_residual[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.fill_(0.01)
        final_layer.bias.zero_()

    formula_features = torch.randn(4, 8)
    context_features = torch.randn(4, 8)
    collision_features = torch.randn(4, 8)
    formula_batch = torch.zeros(4, dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.zeros(3, FRAGMENT_EDGE_FEATURE_DIM)
    edge_attr[:, 0] = 1.0

    delta = head._fragment_path_delta(
        formula_features,
        context_features,
        collision_features,
        edge_index,
        edge_attr,
        formula_batch,
        batch_size=1,
    )
    delta.square().sum().backward()

    for module in (
        head.fragment_path_edge_scorer,
        head.fragment_path_root_scorer,
        head.fragment_path_score_encoder,
        head.fragment_path_residual,
    ):
        for param in module.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_fragment_path_branch_preserves_initial_head_predictions():
    config = MiraFragConfig(
        num_bins=16,
        hidden_dim=8,
        metadata_dim=4,
        dropout=0.0,
        fragment_gnn_layers=1,
    )
    path_config = MiraFragConfig(
        num_bins=16,
        hidden_dim=8,
        metadata_dim=4,
        dropout=0.0,
        fragment_gnn_layers=1,
        fragment_path_layers=2,
    )
    node_feats = torch.randn(3, 5)
    metadata_features = torch.randn(1, 10)
    graph_batch = torch.zeros(3, dtype=torch.long)
    fragments = {
        'batch': torch.zeros(3, dtype=torch.long),
        'formula_batch': torch.zeros(3, dtype=torch.long),
        'atom_index': torch.tensor([0, 1, 2], dtype=torch.long),
        'atom_ptr': torch.tensor([0, 1, 2, 3], dtype=torch.long),
        'features': torch.randn(3, 6),
        'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        'edge_attr': torch.zeros(2, FRAGMENT_EDGE_FEATURE_DIM),
        'formula_index': torch.tensor([0, 1, 2], dtype=torch.long),
        'mz': torch.tensor([10.0, 20.0, 30.0]),
        'bin': torch.tensor([10, 20, 30], dtype=torch.long),
        'log_prior': torch.zeros(3),
    }
    fragments['edge_attr'][:, 0] = 1.0

    torch.manual_seed(23)
    base_head = FragmentSpectrumHead(config)
    torch.manual_seed(23)
    path_head = FragmentSpectrumHead(path_config)
    torch.manual_seed(29)
    base_pred = base_head(node_feats, fragments, metadata_features, graph_batch)
    torch.manual_seed(29)
    path_pred = path_head(node_feats, fragments, metadata_features, graph_batch)

    assert torch.allclose(path_pred['logits'], base_pred['logits'])
    assert torch.allclose(path_pred['oos_logits'], base_pred['oos_logits'])


def test_predict_proba_returns_oos_aware_probabilities():
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
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_spectrum_batch,
            )
        )
    )
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=64, hidden_dim=8, metadata_dim=4),
    )

    pred = model.predict_proba(batch)
    probs = torch.exp(pred['log_probs'])
    oos_probs = torch.exp(pred['oos_log_probs'])
    for batch_idx in range(int(pred['batch_size'])):
        mask = pred['batch'] == batch_idx
        total = probs[mask].sum() + oos_probs[batch_idx]
        assert torch.allclose(total, torch.tensor(1.0), atol=1e-6)


def test_eval_forward_does_not_reenable_mace_grad_for_full_finetune():
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    dataset = BinnedSpectrumDataset(
        _tiny_training_df().iloc[:1],
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_config=FragmentConfig(max_tree_depth=2, max_fragments=16),
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_spectrum_batch,
            )
        )
    )
    config = MiraFragConfig(
        num_bins=16,
        hidden_dim=8,
        metadata_dim=4,
        encoder_finetune_strategy='full',
    )
    model = MiraFragModel(FakeMace(), metadata_config=metadata, config=config)

    model.eval()
    with torch.no_grad():
        pred = model(batch)

    assert not pred['logits'].requires_grad


def test_metadata_config_learns_collision_energy_scaling():
    df = pd.DataFrame(
        {
            'adduct': ['[M+H]+'] * 10,
            'instrument_type': ['HCD'] * 5 + ['CID'] * 5,
            'collision_energy': [10, 20, 30, 40, 50, 100, 110, 120, 130, 140],
        }
    )
    metadata = MetadataConfig.from_dataframe(df)
    assert metadata.collision_energy_center == 75.0
    assert metadata.collision_energy_scale > 0
    assert metadata.collision_energy_by_instrument['HCD']['center'] == 30.0
    assert metadata.collision_energy_by_instrument['CID']['center'] == 120.0


def test_model_uses_instrument_specific_collision_energy_scaling():
    metadata = MetadataConfig(
        adduct_to_idx={'[M+H]+': 0},
        instrument_to_idx={'HCD': 0, 'CID': 1},
        precursor_mz_max=100.0,
        collision_energy_center=50.0,
        collision_energy_scale=10.0,
        collision_energy_by_instrument={
            'HCD': {'center': 20.0, 'scale': 5.0},
        },
    )
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=16, hidden_dim=8, metadata_dim=4),
    )
    features = model.metadata_features(
        {
            'precursor_mz': torch.tensor([50.0, 50.0]),
            'collision_energy': torch.tensor([25.0, 70.0]),
            'adduct': torch.tensor([0, 0]),
            'instrument_type': torch.tensor([0, 1]),
        }
    )
    assert torch.allclose(features[:, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(features[:, 1], torch.tensor([1.0, 2.0]))
