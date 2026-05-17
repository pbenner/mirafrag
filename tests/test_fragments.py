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
        disk_cache_dir=tmp_path / 'features',
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
