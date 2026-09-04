# ruff: noqa: F401
import json
import math
from dataclasses import asdict
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
    RAW_COLLISION_ENERGY_COLUMN,
    BinnedSpectrumDataset,
    MetadataConfig,
    _graph_config_cache_settings,
    collate_spectrum_batch,
    dataloader_performance_kwargs,
    filter_massspecgym_simulation,
    filter_supported_elements,
    merge_group_spectra,
    normalize_collision_energy_dataframe,
    select_split,
)
from mirafrag.encoders.mace import repair_mace_cuequivariance_config
from mirafrag.evaluation import _sparse_prediction_rows
from mirafrag.fragments import (
    FRAGMENT_EDGE_FEATURE_DIM,
    PROTON_MASS,
    SODIUM_ADDUCT_MASS,
    FragmentConfig,
    FragmentSupportProfile,
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
from mirafrag.spectra import parse_number_list
from mirafrag.training import train_model
from tests.helpers import (
    FakeChargeEncoder,
    FakeCueProduct,
    FakeMace,
    FakeTrainAwareEncoder,
    _tiny_loader,
    _tiny_training_df,
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


def test_normalize_collision_energy_dataframe_uses_instrument_stats():
    train = pd.DataFrame(
        {
            'smiles': ['C'] * 10,
            'adduct': ['[M+H]+'] * 10,
            'instrument_type': ['Orbitrap'] * 5 + ['QTOF'] * 5,
            'collision_energy': [
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
            ],
        }
    )
    val = pd.DataFrame(
        {
            'smiles': ['C', 'C', 'C'],
            'adduct': ['[M+H]+'] * 3,
            'instrument_type': ['Orbitrap', 'QTOF', 'unknown'],
            'collision_energy': [20.0, 140.0, 75.0],
        }
    )
    metadata = MetadataConfig.from_dataframe(
        train,
        collision_energy_mode='normalized',
    )

    out = normalize_collision_energy_dataframe(val, metadata_config=metadata)

    assert out['collision_energy'].iloc[0] == 0.0
    assert out['collision_energy'].iloc[1] > 1.0
    assert abs(out['collision_energy'].iloc[2]) < 0.2


def test_normalized_collision_energy_preserves_raw_ce_for_fragment_support():
    df = pd.DataFrame(
        {
            'smiles': ['CC', 'CC'],
            'adduct': ['[M+H]+', '[M+H]+'],
            'instrument_type': ['Orbitrap', 'Orbitrap'],
            'collision_energy': [20.0, 80.0],
        }
    )
    metadata = MetadataConfig.from_dataframe(
        df,
        collision_energy_mode='normalized',
    )

    out = normalize_collision_energy_dataframe(df, metadata_config=metadata)

    assert RAW_COLLISION_ENERGY_COLUMN in out.columns
    assert out[RAW_COLLISION_ENERGY_COLUMN].tolist() == [20.0, 80.0]
    assert out['collision_energy'].tolist() != [20.0, 80.0]
    normalized_again = normalize_collision_energy_dataframe(
        out, metadata_config=metadata
    )
    assert (
        normalized_again['collision_energy'].tolist()
        == out['collision_energy'].tolist()
    )

    with_peaks = out.assign(
        precursor_mz=[100.0, 100.0],
        mzs=['10', '20'],
        intensities=['1', '1'],
    )
    merged = merge_group_spectra(with_peaks, mz_max=64.0, bin_width=1.0)
    assert json.loads(merged[RAW_COLLISION_ENERGY_COLUMN].iloc[0]) == [20.0, 80.0]


def test_merge_group_spectra_merges_replicates_and_drops_precursor():
    df = pd.DataFrame(
        {
            'identifier': ['a', 'b', 'c'],
            'smiles': ['CCO', 'CCO', 'CCN'],
            'adduct': ['[M+H]+', '[M+H]+', '[M+H]+'],
            'instrument_type': ['Orbitrap', 'Orbitrap', 'Orbitrap'],
            'precursor_mz': [47.0, 47.0, 46.0],
            'collision_energy': [10.0, 30.0, 20.0],
            'mzs': ['10,47', '10,20,47', '10'],
            'intensities': ['1,100', '2,3,100', '4'],
        }
    )

    out = merge_group_spectra(df, mz_max=64.0, bin_width=1.0)

    assert len(out) == 2
    merged = out[out['smiles'] == 'CCO'].iloc[0]
    assert merged['identifier'] == 'a|b'
    assert json.loads(merged['collision_energy']) == [10.0, 30.0]
    mzs = parse_number_list(merged['mzs'])
    intensities = parse_number_list(merged['intensities'])
    assert mzs == [10.5, 20.5]
    assert intensities == [3.0, 3.0]


def test_merged_collision_energy_lists_collate_with_scalar_fallback():
    df = pd.DataFrame(
        {
            'identifier': ['a', 'b'],
            'smiles': ['CCO', 'CCO'],
            'adduct': ['[M+H]+', '[M+H]+'],
            'instrument_type': ['Orbitrap', 'Orbitrap'],
            'precursor_mz': [47.0, 47.0],
            'collision_energy': [10.0, 30.0],
            'mzs': ['10', '20'],
            'intensities': ['1', '1'],
        }
    )
    merged = merge_group_spectra(df, mz_max=64.0, bin_width=1.0)
    metadata = MetadataConfig.from_dataframe(merged)
    dataset = BinnedSpectrumDataset(
        merged,
        graph_config=GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7),
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=False,
    )

    batch = collate_spectrum_batch([dataset[0]])

    assert torch.allclose(batch['collision_energy'], torch.tensor([20.0]))
    assert torch.allclose(
        batch['collision_energy_values'],
        torch.tensor([10.0, 30.0]),
    )
    assert torch.equal(batch['collision_energy_batch'], torch.tensor([0, 0]))


def test_binned_dataset_excludes_precursor_peaks_from_targets():
    df = _tiny_training_df().copy()
    df.loc[0, 'mzs'] = '18,47,60'
    df.loc[0, 'intensities'] = '1,10,2'
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    dataset = BinnedSpectrumDataset(
        df.iloc[:1],
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=False,
    )

    item = dataset[0]

    assert item['true_mzs'].tolist() == [18.0, 60.0]
    assert item['true_intensities'].tolist() == [1.0, 2.0]


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
        'disk_cache_dir': cache_dir,
    }
    dataset = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)
    first = dataset[0]

    assert list((cache_dir / 'graphs').glob('graph-*/*.pt'))
    assert list((cache_dir / 'fragments').glob('*.pt'))

    reloaded = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)[0]
    assert torch.equal(first['graph']['node_attrs'], reloaded['graph']['node_attrs'])
    assert first['fragments']['bins'] == reloaded['fragments']['bins']


def test_fragment_disk_cache_augments_bond_break_fields_from_shared_cache(tmp_path):
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    cache_dir = tmp_path / 'features'
    base_kwargs = {
        'graph_config': graph_config,
        'metadata_config': metadata,
        'mz_max': 64.0,
        'bin_width': 1.0,
        'include_fragments': True,
        'disk_cache_dir': cache_dir,
    }
    base_dataset = BinnedSpectrumDataset(
        _tiny_training_df(),
        fragment_config=FragmentConfig(max_tree_depth=2, max_fragments=16),
        **base_kwargs,
    )
    base_fragments = base_dataset[0]['fragments']
    fragment_files_before = set((cache_dir / 'fragments').glob('*.pt'))

    bond_dataset = BinnedSpectrumDataset(
        _tiny_training_df(),
        fragment_config=FragmentConfig(
            max_tree_depth=2,
            max_fragments=16,
            include_bond_breaks=True,
        ),
        **base_kwargs,
    )
    bond_fragments = bond_dataset[0]['fragments']
    fragment_files_after = set((cache_dir / 'fragments').glob('*.pt'))

    assert 'bond_atom_indices' not in base_fragments
    assert 'bond_features' not in base_fragments
    assert fragment_files_before == fragment_files_after
    assert len(bond_fragments['bond_atom_indices']) == len(
        bond_fragments['atom_indices']
    )
    assert len(bond_fragments['bond_features']) == len(bond_fragments['atom_indices'])
    assert any(bond_fragments['bond_atom_indices'])


def test_graph_cache_is_namespaced_but_fragment_cache_is_shared(tmp_path):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    fragment_config = FragmentConfig(max_tree_depth=2, max_fragments=16)
    cache_dir = tmp_path / 'features'
    base_kwargs = {
        'metadata_config': metadata,
        'mz_max': 64.0,
        'bin_width': 1.0,
        'include_fragments': True,
        'fragment_config': fragment_config,
        'disk_cache_dir': cache_dir,
    }
    tiny_df = _tiny_training_df()
    dataset_a = BinnedSpectrumDataset(
        tiny_df,
        graph_config=GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7),
        **base_kwargs,
    )
    dataset_b = BinnedSpectrumDataset(
        tiny_df,
        graph_config=GraphConfig(atomic_numbers=(1, 6, 8), cutoff=3.0, seed=7),
        **base_kwargs,
    )
    smiles = tiny_df.at[0, 'smiles']
    graph_path_a = dataset_a._feature_cache_path(
        'graphs',
        smiles,
        {'graph_config': _graph_config_cache_settings(dataset_a.graph_config)},
    )
    graph_path_b = dataset_b._feature_cache_path(
        'graphs',
        smiles,
        {'graph_config': _graph_config_cache_settings(dataset_b.graph_config)},
    )
    fragment_settings = {
        'fragment_config': asdict(fragment_config),
        'adduct': '[M+H]+',
        'mz_max': 64.0,
        'bin_width': 1.0,
    }
    fragment_path_a = dataset_a._feature_cache_path(
        'fragments',
        smiles,
        fragment_settings,
    )
    fragment_path_b = dataset_b._feature_cache_path(
        'fragments',
        smiles,
        fragment_settings,
    )

    assert graph_path_a.parent != graph_path_b.parent
    assert graph_path_a.parent.parent == cache_dir / 'graphs'
    assert graph_path_b.parent.parent == cache_dir / 'graphs'
    assert fragment_path_a == fragment_path_b
    assert fragment_path_a.parent == cache_dir / 'fragments'


def test_default_graph_cache_settings_preserve_legacy_key():
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    settings = _graph_config_cache_settings(graph_config)

    assert settings == {
        'atomic_numbers': (1, 6, 8),
        'cutoff': 5.0,
        'seed': 7,
        'add_hydrogens': True,
        'optimize': True,
        'max_embed_attempts': 50,
        'embed_timeout_seconds': 10,
        'fallback_to_2d': True,
        'warn_2d_fallback': False,
        'validate_bond_geometry': True,
    }


def test_aimnet_relaxation_graph_cache_settings_fork_key():
    graph_config = GraphConfig(
        atomic_numbers=(1, 6, 8),
        cutoff=5.0,
        seed=7,
        relaxation='aimnet',
        aimnet_relax_model='aimnet2',
        aimnet_relax_steps=20,
        aimnet_relax_fmax=0.1,
        aimnet_relax_device='cpu',
    )
    settings = _graph_config_cache_settings(graph_config)

    assert settings['relaxation'] == 'aimnet'
    assert settings['aimnet_relax_model'] == 'aimnet2'
    assert settings['aimnet_relax_steps'] == 20
    assert settings['aimnet_relax_fmax'] == 0.1
    assert settings['aimnet_relax_device'] == 'cpu'


def test_high_ce_fragment_support_uses_raw_ce_after_normalization():
    df = _tiny_training_df().copy()
    df.loc[0, 'collision_energy'] = 20.0
    df.loc[1, 'collision_energy'] = 80.0
    metadata = MetadataConfig.from_dataframe(df, collision_energy_mode='normalized')
    normalized = normalize_collision_energy_dataframe(df, metadata_config=metadata)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    dataset = BinnedSpectrumDataset(
        normalized,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_support_profile=FragmentSupportProfile(
            base=FragmentConfig(max_tree_depth=1, max_fragments=8),
            high_ce_threshold=60.0,
            high_ce=FragmentConfig(max_tree_depth=3, max_fragments=16),
        ),
    )

    assert dataset._fragment_config_for_row(0).max_tree_depth == 1
    assert dataset._fragment_config_for_row(1).max_tree_depth == 3


def test_high_ce_fragment_support_selects_row_specific_cache_config(tmp_path):
    df = _tiny_training_df()
    df.loc[1, 'collision_energy'] = 70.0
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    base = FragmentConfig(
        max_tree_depth=1,
        max_broken_bonds=2,
        max_fragments=16,
        max_edges=32,
    )
    high_ce = FragmentConfig(
        max_tree_depth=4,
        max_broken_bonds=8,
        max_fragments=64,
        max_edges=128,
    )
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        disk_cache_dir=tmp_path / 'features',
        fragment_support_profile=FragmentSupportProfile(
            base=base,
            high_ce_threshold=60.0,
            high_ce=high_ce,
        ),
    )

    assert dataset._fragment_config_for_row(0) == base
    assert dataset._fragment_config_for_row(1) == high_ce
    low_path = dataset._feature_cache_path(
        'fragments',
        df.at[0, 'smiles'],
        {
            'fragment_config': asdict(base),
            'adduct': '[M+H]+',
            'mz_max': 64.0,
            'bin_width': 1.0,
        },
    )
    high_path = dataset._feature_cache_path(
        'fragments',
        df.at[1, 'smiles'],
        {
            'fragment_config': asdict(high_ce),
            'adduct': '[M+H]+',
            'mz_max': 64.0,
            'bin_width': 1.0,
        },
    )
    assert low_path != high_path


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
    assert kwargs['persistent_workers'] is True
    assert 'pin_memory' not in kwargs
    assert 'multiprocessing_context' not in kwargs


def test_cuda_dataloader_workers_use_spawn_and_pinned_memory():
    kwargs = dataloader_performance_kwargs(num_workers=2, device='cuda')
    assert kwargs['pin_memory'] is True
    assert kwargs['persistent_workers'] is True
    assert kwargs['multiprocessing_context'] == 'spawn'


def test_cuda_single_process_loader_uses_pinned_memory_without_spawn():
    kwargs = dataloader_performance_kwargs(num_workers=0, device='cuda')
    assert kwargs == {'pin_memory': True}
