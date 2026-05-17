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
        'disk_cache_dir': cache_dir,
    }
    dataset = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)
    first = dataset[0]

    assert list((cache_dir / 'graphs').glob('*.pt'))
    assert list((cache_dir / 'fragments').glob('*.pt'))

    reloaded = BinnedSpectrumDataset(_tiny_training_df(), **kwargs)[0]
    assert torch.equal(first['graph']['node_attrs'], reloaded['graph']['node_attrs'])
    assert first['fragments']['bins'] == reloaded['fragments']['bins']


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
