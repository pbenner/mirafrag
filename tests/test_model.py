import math
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.nn import LazyLinear
from torch.utils.data import DataLoader

from mirafrag.chem import GraphConfig
from mirafrag.cli.cache import (
    _apply_fragment_args_to_model_config as _apply_cache_fragment_args_to_model_config,
)
from mirafrag.cli.train import (
    _apply_fragment_args_to_model_config as _apply_train_fragment_args_to_model_config,
)
from mirafrag.data import (
    BinnedSpectrumDataset,
    MetadataConfig,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    select_split,
)
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
from mirafrag.model import (
    MiraFragConfig,
    MiraFragModel,
    _repair_mace_cuequivariance_config,
    load_checkpoint,
    set_encoder_finetune_strategy,
)
from mirafrag.training import (
    _build_scheduler,
    _optimizer_param_groups,
    _scheduler_total_steps,
    _sparse_prediction_rows,
    fragnnet_sparse_cross_entropy,
    projected_sparse_binned_kl_divergence,
    soft_binned_coverage_kl_divergence,
    soft_binned_kl_divergence,
    soft_projected_sparse_kl_divergence,
    sparse_binned_cosine_similarity,
    sparse_binned_kl_divergence,
    spectrum_loss,
    train_model,
)


class FakeMace(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('atomic_numbers', torch.tensor([1, 6, 8]))
        self.register_buffer('r_max', torch.tensor(5.0))
        self.proj = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(3))

    def forward(self, graph, **kwargs):
        return {'node_feats': self.proj(graph['node_attrs'].float())}


class FakeChargeEncoder(FakeMace):
    uses_molecular_charge = True

    def __init__(self):
        super().__init__()
        self.last_molecular_charge = None

    def forward(self, graph, **kwargs):
        self.last_molecular_charge = kwargs.get('molecular_charge')
        return super().forward(graph, **kwargs)


class FakeTrainAwareEncoder(FakeMace):
    pass


class FakeCueSymmetricContraction(nn.Module):
    def forward(self, x, indices):
        return x


FakeCueSymmetricContraction.__module__ = (
    'cuequivariance_torch.operations.symmetric_contraction'
)


class FakeCueProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.symmetric_contractions = FakeCueSymmetricContraction()
        self.cueq_config = None


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


def test_fragment_candidates_include_recursive_fragments():
    fragments = smiles_to_fragment_candidates(
        'CCO',
        mz_max=64.0,
        bin_width=1.0,
        config=FragmentConfig(max_tree_depth=2, max_fragments=16),
    )
    assert fragments['bins']
    assert any(len(atom_indices) < 3 for atom_indices in fragments['atom_indices'])
    assert any(feature[3] != 0.0 for feature in fragments['features'])


def test_fragment_candidates_include_fragment_graph_edges():
    fragments = smiles_to_fragment_candidates(
        'CC(C)O',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=3,
            max_broken_bonds=6,
            max_fragments=128,
            max_edges=256,
        ),
    )

    assert fragments['edge_index']
    assert len(fragments['edge_index']) == len(fragments['edge_features'])
    assert len(fragments['edge_features'][0]) == FRAGMENT_EDGE_FEATURE_DIM
    assert any(edge_feature[0] > 0.0 for edge_feature in fragments['edge_features'])


def test_fragment_candidates_include_isotope_peak_support():
    fragments = smiles_to_fragment_candidates(
        'CCCCCCCCCC',
        mz_max=256.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=0,
            max_fragments=8,
            include_isotopes=True,
            isotope_threshold=0.001,
            max_isotope_peaks=4,
        ),
    )
    assert len(fragments['mzs']) >= 2
    assert len(fragments['log_priors']) == len(fragments['mzs'])
    assert any(log_prior < 0.0 for log_prior in fragments['log_priors'])
    assert any(feature[6] > 0.0 for feature in fragments['features'])
    assert len(fragments['formula_index']) == len(fragments['mzs'])
    assert len(set(fragments['formula_index'])) < len(fragments['formula_index'])


def test_collate_offsets_formula_indices_between_molecules():
    config = FragmentConfig(
        max_tree_depth=0,
        max_fragments=8,
        include_isotopes=True,
        isotope_threshold=0.001,
        max_isotope_peaks=4,
    )
    first = smiles_to_fragment_candidates(
        'CCCCCCCCCC',
        mz_max=256.0,
        bin_width=0.01,
        config=config,
    )
    second = smiles_to_fragment_candidates(
        'CCCCCCCCCC',
        mz_max=256.0,
        bin_width=0.01,
        config=config,
    )
    batch = collate_fragment_candidates([first, second], node_offsets=[0, 10])
    first_count = len(first['mzs'])
    assert (
        batch['formula_index'][:first_count].max()
        < batch['formula_index'][first_count:].min()
    )


def test_recursive_fragment_candidates_reach_atom_pull_depth():
    fragments = smiles_to_fragment_candidates(
        'CC(C)O',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=3,
            max_broken_bonds=6,
            max_fragments=128,
        ),
    )
    assert fragments['bins']
    assert any(feature[1] > 0.5 for feature in fragments['features'])


def test_branching_fragment_tree_keeps_single_atom_components():
    fragments = smiles_to_fragment_candidates(
        'CC(C)(C)C',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=1,
            max_broken_bonds=1,
            max_fragments=32,
            include_isotopes=False,
        ),
    )

    assert any(len(atom_indices) == 1 for atom_indices in fragments['atom_indices'])


def test_fragment_tree_respects_broken_bond_budget():
    fragments = smiles_to_fragment_candidates(
        'CCO',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=2,
            max_broken_bonds=0,
            max_fragments=32,
            include_isotopes=False,
        ),
    )

    assert {tuple(atom_indices) for atom_indices in fragments['atom_indices']} == {
        (0, 1, 2)
    }


def test_fragment_tree_collapses_isomorphic_fragments_like_fragnnet():
    fragments = smiles_to_fragment_candidates(
        'CC',
        mz_max=64.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=1,
            max_broken_bonds=1,
            max_fragments=16,
            include_isotopes=False,
        ),
    )
    unique_atom_sets = {
        tuple(atom_indices) for atom_indices in fragments['atom_indices']
    }
    assert len(unique_atom_sets) == 2


def test_fragment_tree_edges_preserve_hydrogen_shift():
    fragments = smiles_to_fragment_candidates(
        'CCO',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=2,
            max_broken_bonds=2,
            max_fragments=64,
            max_edges=512,
            include_isotopes=False,
        ),
    )
    for (src, dst), edge_feature in zip(
        fragments['edge_index'], fragments['edge_features']
    ):
        if edge_feature[0] > 0.0 or edge_feature[1] > 0.0:
            assert math.isclose(
                fragments['features'][src][3],
                fragments['features'][dst][3],
                rel_tol=0.0,
                abs_tol=1e-12,
            )


def test_fragment_candidate_pruning_limits_formula_count():
    fragments = smiles_to_fragment_candidates(
        'CC(C)O',
        mz_max=128.0,
        bin_width=0.01,
        config=FragmentConfig(
            max_tree_depth=3,
            max_broken_bonds=6,
            max_fragments=4,
        ),
    )
    assert len(set(fragments['formula_index'])) <= 4


def test_fragment_candidates_use_adduct_mass():
    config = FragmentConfig(max_tree_depth=0, max_fragments=1)
    protonated = smiles_to_fragment_candidates(
        'CCO',
        mz_max=128.0,
        bin_width=0.01,
        adduct='[M+H]+',
        config=config,
    )
    sodiated = smiles_to_fragment_candidates(
        'CCO',
        mz_max=128.0,
        bin_width=0.01,
        adduct='[M+Na]+',
        config=config,
    )
    assert math.isclose(
        sodiated['mzs'][0] - protonated['mzs'][0],
        SODIUM_ADDUCT_MASS - PROTON_MASS,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_massspecgym_simulation_filter_keeps_all_adducts_with_collision_energy():
    df = pd.DataFrame(
        {
            'simulation_challenge': [True, True, True, False],
            'adduct': ['[M+H]+', '[M+Na]+', '[M+K]+', '[M+Na]+'],
            'collision_energy': [20.0, 30.0, None, 40.0],
        }
    )
    out = filter_massspecgym_simulation(df)
    assert out['adduct'].tolist() == ['[M+H]+', '[M+Na]+']


def test_massspecgym_simulation_filter_parses_string_booleans():
    df = pd.DataFrame(
        {
            'simulation_challenge': ['True', 'False', '1', '0', 'yes', 'no'],
            'adduct': ['a', 'b', 'c', 'd', 'e', 'f'],
            'collision_energy': [10.0] * 6,
        }
    )
    out = filter_massspecgym_simulation(df)
    assert out['adduct'].tolist() == ['a', 'c', 'e']


def test_select_split_strips_split_labels_and_explicit_values():
    df = pd.DataFrame({'split': [' train ', 'VAL', ' test '], 'value': [1, 2, 3]})
    assert select_split(df, split='train')['value'].tolist() == [1]
    assert select_split(df, split='unused', split_col='split', split_value=' test ')[
        'value'
    ].tolist() == [3]


def test_binned_dataset_disk_cache_reuses_graphs_and_fragments(tmp_path):
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    cache_dir = tmp_path / 'features'
    kwargs = {
        'graph_config': graph_config,
        'metadata_config': metadata,
        'mz_max': 64.0,
        'bin_width': 1.0,
        'include_fragments': True,
        'fragment_config': FragmentConfig(max_tree_depth=2, max_fragments=16),
        'cache_dir': cache_dir,
    }
    dataset = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)
    first = dataset[0]

    assert list((cache_dir / 'graphs').glob('*.pt'))
    assert list((cache_dir / 'fragments').glob('*.pt'))

    reloaded = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)[0]
    assert torch.equal(first['graph']['node_attrs'], reloaded['graph']['node_attrs'])
    assert first['fragments']['bins'] == reloaded['fragments']['bins']


def test_fragment_disk_cache_is_adduct_specific(tmp_path):
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(
        adduct_to_idx={'[M+H]+': 0, '[M+Na]+': 1},
        instrument_to_idx={'HCD': 0},
    )
    df = _tiny_training_df()
    df.loc[1, 'adduct'] = '[M+Na]+'
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=128.0,
        bin_width=0.01,
        include_fragments=True,
        fragment_config=FragmentConfig(max_tree_depth=0, max_fragments=1),
        cache_dir=tmp_path / 'features',
    )

    protonated = dataset[0]['fragments']
    sodiated = dataset[1]['fragments']

    assert len(list((tmp_path / 'features' / 'fragments').glob('*.pt'))) == 2
    assert math.isclose(
        sodiated['mzs'][0] - protonated['mzs'][0],
        SODIUM_ADDUCT_MASS - PROTON_MASS,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_fragment_adduct_parser_handles_signed_and_multicharged_adducts():
    deprotonated = parse_fragment_adduct('[M-H]-')
    double_protonated = parse_fragment_adduct('[M+2H]2+')
    formate = parse_fragment_adduct('[M+FA-H]-')

    assert deprotonated.charge == -1
    assert math.isclose(deprotonated.mass_delta, -PROTON_MASS, abs_tol=1e-6)
    assert double_protonated.charge == 2
    assert math.isclose(double_protonated.mass_delta, 2.0 * PROTON_MASS, abs_tol=1e-6)
    assert formate.charge == -1
    assert formate.mass_delta > 40.0


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


def test_repair_mace_cuequivariance_config_restores_product_flags():
    mace = nn.Module()
    mace.product = FakeCueProduct()
    _repair_mace_cuequivariance_config(mace)
    assert mace.product.cueq_config is not None
    assert mace.product.cueq_config.enabled
    assert mace.product.cueq_config.optimize_symmetric


def test_set_encoder_finetune_strategy_wraps_head_checkpoint_for_delta():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='head',
        ),
    )
    assert not any(param.requires_grad for param in model.encoder.parameters())

    set_encoder_finetune_strategy(model, 'delta')

    assert model.config.encoder_finetune_strategy == 'delta'
    assert hasattr(model.encoder, 'delta_parameters')
    assert not any(
        param.requires_grad for param in model.encoder.base_module.parameters()
    )
    assert all(param.requires_grad for param in model.encoder.delta_parameters())


def test_loaded_checkpoint_fragment_args_do_not_rewrite_head_architecture():
    args = SimpleNamespace(
        max_fragment_tree_depth=4,
        max_fragment_broken_bonds=7,
        max_fragments=1024,
        max_fragment_edges=8192,
        fragment_gnn_layers=9,
        include_fragment_isotopes=False,
        fragment_isotope_threshold=0.01,
        max_fragment_isotope_peaks=3,
    )

    for apply_args in (
        _apply_train_fragment_args_to_model_config,
        _apply_cache_fragment_args_to_model_config,
    ):
        config = MiraFragConfig(num_bins=16, fragment_gnn_layers=2)
        apply_args(config, args)

        assert config.max_fragment_tree_depth == 4
        assert config.max_fragment_broken_bonds == 7
        assert config.max_fragments == 1024
        assert config.max_fragment_edges == 8192
        assert config.include_fragment_isotopes is False
        assert config.fragment_isotope_threshold == 0.01
        assert config.max_fragment_isotope_peaks == 3
        assert config.fragment_gnn_layers == 2


def test_loaded_checkpoint_omitted_fragment_args_preserve_candidate_support():
    args = SimpleNamespace(
        max_fragment_tree_depth=None,
        max_fragment_broken_bonds=None,
        max_fragments=None,
        max_fragment_edges=None,
        fragment_gnn_layers=9,
        include_fragment_isotopes=None,
        fragment_isotope_threshold=None,
        max_fragment_isotope_peaks=None,
    )

    for apply_args in (
        _apply_train_fragment_args_to_model_config,
        _apply_cache_fragment_args_to_model_config,
    ):
        config = MiraFragConfig(
            num_bins=16,
            max_fragment_tree_depth=5,
            max_fragment_broken_bonds=8,
            max_fragments=2048,
            max_fragment_edges=9000,
            include_fragment_isotopes=False,
            fragment_isotope_threshold=0.02,
            max_fragment_isotope_peaks=4,
            fragment_gnn_layers=2,
        )
        apply_args(config, args)

        assert config.max_fragment_tree_depth == 5
        assert config.max_fragment_broken_bonds == 8
        assert config.max_fragments == 2048
        assert config.max_fragment_edges == 9000
        assert config.include_fragment_isotopes is False
        assert config.fragment_isotope_threshold == 0.02
        assert config.max_fragment_isotope_peaks == 4
        assert config.fragment_gnn_layers == 2


def test_fragment_config_from_model_config_preserves_candidate_support_settings():
    config = MiraFragConfig(
        num_bins=16,
        max_fragment_tree_depth=4,
        max_fragment_broken_bonds=7,
        max_fragments=1024,
        max_fragment_edges=8192,
        include_fragment_isotopes=False,
        fragment_isotope_threshold=0.01,
        max_fragment_isotope_peaks=3,
    )

    fragment_config = fragment_config_from_model_config(config)

    assert fragment_config.max_tree_depth == 4
    assert fragment_config.max_broken_bonds == 7
    assert fragment_config.max_fragments == 1024
    assert fragment_config.max_edges == 8192
    assert fragment_config.include_isotopes is False
    assert fragment_config.isotope_threshold == 0.01
    assert fragment_config.max_isotope_peaks == 3


def test_default_fragment_config_matches_model_config():
    model_config = MiraFragConfig(num_bins=16)
    fragment_config = FragmentConfig()

    assert model_config.max_fragment_tree_depth == fragment_config.max_tree_depth == 3
    assert (
        model_config.max_fragment_broken_bonds == fragment_config.max_broken_bonds == 6
    )
    assert model_config.max_fragments == fragment_config.max_fragments == 2048
    assert model_config.max_fragment_edges == fragment_config.max_edges == 8192
    assert model_config.include_fragment_isotopes is fragment_config.include_isotopes
    assert model_config.fragment_isotope_threshold == fragment_config.isotope_threshold
    assert (
        model_config.max_fragment_isotope_peaks
        == fragment_config.max_isotope_peaks
        == 1
    )


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


def test_train_model_materializes_lazy_head(tmp_path):
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    loader = _tiny_loader(df, graph_config, metadata)
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
        ),
    )
    output = tmp_path / 'mirafrag.pt'
    history = train_model(
        model,
        loader,
        None,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        device='cpu',
        output=output,
        show_progress=False,
    )
    assert output.exists()
    assert history['epoch'] == [1]
    loaded, payload = load_checkpoint(output, device='cpu')
    assert payload['mirafrag_config']['encoder_finetune_strategy'] == 'head'
    batch = next(iter(loader))
    assert loaded(batch)['logits'].ndim == 1


def test_train_model_records_initial_validation_for_checkpoint_resume(tmp_path):
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    loader = _tiny_loader(df, graph_config, metadata)
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
        ),
    )

    history = train_model(
        model,
        loader,
        loader,
        epochs=1,
        lr=0.0,
        weight_decay=0.0,
        device='cpu',
        output=tmp_path / 'mirafrag_resume.pt',
        show_progress=False,
        scheduler_name='none',
        evaluate_initial=True,
    )

    assert history['epoch'] == [0, 1]
    assert math.isnan(history['train_loss'][0])
    assert not math.isnan(history['val_loss'][0])


def test_delta_finetune_keeps_base_mace_parameters_frozen(tmp_path):
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    loader = _tiny_loader(df, graph_config, metadata)
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='delta',
        ),
    )
    base_before = [
        param.detach().clone() for param in model.encoder.base_module.parameters()
    ]
    delta_before = [
        param.detach().clone() for param in model.encoder.delta_parameters()
    ]
    assert delta_before

    train_model(
        model,
        loader,
        None,
        epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        device='cpu',
        output=tmp_path / 'mirafrag_delta.pt',
        show_progress=False,
    )

    base_after = [
        param.detach().clone() for param in model.encoder.base_module.parameters()
    ]
    delta_after = [param.detach().clone() for param in model.encoder.delta_parameters()]
    assert all(
        torch.equal(before, after) for before, after in zip(base_before, base_after)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(delta_before, delta_after)
    )


def test_optimizer_param_groups_use_single_lr():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='full',
        ),
    )
    groups = _optimizer_param_groups(
        model,
        lr=1e-4,
        weight_decay=1e-8,
    )
    assert {group['name'] for group in groups} == {'head', 'encoder'}
    lr_by_name = {group['name']: group['lr'] for group in groups}
    wd_by_name = {group['name']: group['weight_decay'] for group in groups}
    assert lr_by_name['head'] == 1e-4
    assert lr_by_name['encoder'] == 1e-4
    assert wd_by_name['head'] == 0.0
    assert wd_by_name['encoder'] == 1e-8


def test_optimizer_param_groups_skip_unused_lazy_parameters():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
        ),
    )
    model.unused_lazy = LazyLinear(8)

    groups = _optimizer_param_groups(
        model,
        lr=1e-4,
        weight_decay=1e-8,
    )

    grouped_params = [param for group in groups for param in group['params']]
    grouped_param_ids = {id(param) for param in grouped_params}
    assert id(model.unused_lazy.weight) not in grouped_param_ids
    assert id(model.unused_lazy.bias) not in grouped_param_ids


def test_cosine_scheduler_decays():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([{'params': [param], 'lr': 1e-4}])
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name='cosine',
        total_steps=10,
        min_lr_ratio=0.1,
    )
    assert scheduler is not None
    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    assert max(lrs) <= 1e-4
    assert lrs[-1] < lrs[1]
    assert lrs[-1] >= 1e-5


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


def test_exponential_scheduler_decays_by_gamma():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([{'params': [param], 'lr': 1e-4}])
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name='exponential',
        total_steps=20,
        min_lr_ratio=0.1,
        exponential_gamma=0.8,
    )
    assert scheduler is not None
    assert optimizer.param_groups[0]['lr'] == 1e-4
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 8e-5

    for _ in range(20):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] >= 1e-5


def test_plateau_scheduler_decays_after_stalled_validation():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([{'params': [param], 'lr': 1e-4}])
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name='plateau',
        total_steps=10,
        min_lr_ratio=0.1,
        plateau_factor=0.5,
        plateau_patience=0,
    )
    assert scheduler is not None
    scheduler.step(1.0)
    assert optimizer.param_groups[0]['lr'] == 1e-4
    scheduler.step(1.0)
    assert optimizer.param_groups[0]['lr'] == 5e-5


def test_scheduler_total_steps_respects_interval():
    assert (
        _scheduler_total_steps(
            epochs=20,
            steps_per_epoch=12_404,
            scheduler_interval='epoch',
        )
        == 20
    )
    assert (
        _scheduler_total_steps(
            epochs=20,
            steps_per_epoch=12_404,
            scheduler_interval='step',
        )
        == 248_080
    )


def test_filter_supported_elements_drops_unsupported_boron():
    df = pd.DataFrame({'smiles': ['CCO', 'B(O)O', 'not-a-smiles']})
    filtered, stats = filter_supported_elements(
        df,
        supported_atomic_numbers=(1, 6, 7, 8),
    )
    assert filtered['smiles'].tolist() == ['CCO']
    assert stats['input'] == 3
    assert stats['kept'] == 1
    assert stats['dropped_invalid_smiles'] == 1
    assert stats['dropped_unsupported_elements'] == 1
    assert stats['unsupported_Z_5'] == 1


def test_dataloader_workers_silence_rdkit_logs():
    kwargs = dataloader_performance_kwargs(num_workers=2, device='cpu')
    assert callable(kwargs['worker_init_fn'])


def _tiny_training_df():
    df = pd.DataFrame(
        {
            'smiles': ['CCO', 'CCO'],
            'mzs': ['18,30', '18,30'],
            'intensities': ['1,2', '1,3'],
            'precursor_mz': [47.0, 47.0],
            'adduct': ['[M+H]+', '[M+H]+'],
            'instrument_type': ['HCD', 'HCD'],
            'collision_energy': [20.0, 30.0],
        }
    )
    return df


def _tiny_loader(df, graph_config, metadata):
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=32.0,
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
    return loader
