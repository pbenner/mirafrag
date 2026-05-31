# ruff: noqa: F401
import math
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn
from torch.nn import LazyLinear
from torch.utils.data import DataLoader

from mirafrag.checkpoint import load_checkpoint, save_checkpoint
from mirafrag.chem import GraphConfig
from mirafrag.cli.cache import (
    _apply_fragment_args_to_model_config as _apply_cache_fragment_args_to_model_config,
)
from mirafrag.cli.oracle import (
    _load_oracle_checkpoint_config,
    compute_oracle_diagnostics,
)
from mirafrag.cli.train import (
    _apply_fragment_args_to_model_config as _apply_train_fragment_args_to_model_config,
)
from mirafrag.cli.train import _set_head_dropout, _validation_tune_candidates
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


def test_oracle_checkpoint_config_reads_encoder_metadata(tmp_path):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    output = tmp_path / 'mirafrag.pt'
    save_checkpoint(output, model)

    config, loaded_metadata, graph_config = _load_oracle_checkpoint_config(
        str(output),
        mz_max=32.0,
        bin_width=1.0,
    )

    assert config.num_bins == 32
    assert loaded_metadata.adduct_to_idx == {'[M+H]+': 0}
    assert graph_config.atomic_numbers == (1, 6, 8)
    assert graph_config.cutoff == 5.0


def test_compute_oracle_diagnostics_uses_fragment_support_without_graph_forward():
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_config=FragmentConfig(max_tree_depth=2, max_fragments=16),
    )

    rows, summary = compute_oracle_diagnostics(
        dataset,
        batch_size=2,
        mz_max=64.0,
        bin_width=1.0,
        mass_tolerance=0.01,
        relative_mass_tolerance=False,
        mass_tolerance_min_mz=200.0,
        show_progress=False,
    )

    assert len(rows) == 2
    assert summary['n'] == 2
    assert 0.0 <= summary['oracle_binned_cosine_mean'] <= 1.0
    assert 0.0 <= summary['oracle_tolerance_cosine_mean'] <= 1.0


def test_parallel_oracle_diagnostics_match_serial():
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    dataset = BinnedSpectrumDataset(
        df,
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=64.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_config=FragmentConfig(max_tree_depth=2, max_fragments=16),
    )

    serial_rows, serial_summary = compute_oracle_diagnostics(
        dataset,
        batch_size=2,
        mz_max=64.0,
        bin_width=1.0,
        mass_tolerance=0.01,
        relative_mass_tolerance=False,
        mass_tolerance_min_mz=200.0,
        show_progress=False,
        num_workers=0,
    )
    parallel_rows, parallel_summary = compute_oracle_diagnostics(
        dataset,
        batch_size=2,
        mz_max=64.0,
        bin_width=1.0,
        mass_tolerance=0.01,
        relative_mass_tolerance=False,
        mass_tolerance_min_mz=200.0,
        show_progress=False,
        num_workers=2,
        chunk_size=1,
    )

    assert parallel_summary == serial_summary
    pd.testing.assert_frame_equal(parallel_rows, serial_rows)


def test_validation_tune_candidates_are_limited_and_include_baseline():
    args = SimpleNamespace(
        learning_rate=3e-5,
        dropout=0.01,
        weight_decay=1e-6,
        swa=False,
        swa_start_epoch=None,
        swa_lr=None,
        swa_anneal_epochs=1,
        tune_trials=4,
        tune_epochs=6,
        tune_lrs='1e-5,3e-5',
        tune_dropouts='0,0.02',
        tune_weight_decays='0,1e-6',
        tune_swa_start_epochs='3,8',
        tune_swa_lrs='1e-5',
        tune_seed=11,
        seed=17,
    )

    candidates = _validation_tune_candidates(args)

    assert len(candidates) == 4
    assert candidates[0].lr == 3e-5
    assert candidates[0].dropout == 0.01
    assert candidates[0].weight_decay == 1e-6
    assert all(candidate.swa_start_epoch != 8 for candidate in candidates)


def test_validation_tune_candidates_do_not_increase_lr_for_swa():
    args = SimpleNamespace(
        learning_rate=1e-5,
        dropout=0.0,
        weight_decay=0.0,
        swa=False,
        swa_start_epoch=None,
        swa_lr=None,
        swa_anneal_epochs=1,
        tune_trials=20,
        tune_epochs=4,
        tune_lrs='1e-5,3e-5',
        tune_dropouts='0',
        tune_weight_decays='0',
        tune_swa_start_epochs='2',
        tune_swa_lrs='1e-5,1e-4',
        tune_seed=3,
        seed=17,
    )

    candidates = _validation_tune_candidates(args)

    assert all(
        (not candidate.swa) or candidate.swa_lr <= candidate.lr
        for candidate in candidates
    )


def test_set_head_dropout_only_updates_head_dropout_modules():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            dropout=0.1,
        ),
    )

    _set_head_dropout(model, 0.25)

    assert model.config.dropout == 0.25
    assert {
        module.p for module in model.head.modules() if isinstance(module, nn.Dropout)
    } == {0.25}


def test_repair_mace_cuequivariance_config_restores_product_flags():
    mace = nn.Module()
    mace.product = FakeCueProduct()
    repair_mace_cuequivariance_config(mace)
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


def test_train_model_materializes_lazy_head(tmp_path, monkeypatch):
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
    checkpoint_payload = torch.load(output, map_location='cpu', weights_only=True)
    assert checkpoint_payload['checkpoint_format'] == 'mirafrag-state-v1'
    assert 'model_state_dict' in checkpoint_payload
    assert 'model' not in checkpoint_payload
    monkeypatch.setattr(
        'mirafrag.checkpoint.load_foundation_encoder', lambda **_: FakeMace()
    )
    loaded, payload = load_checkpoint(output, device='cpu')
    assert payload['mirafrag_config']['encoder_finetune_strategy'] == 'head'
    batch = next(iter(loader))
    assert loaded(batch)['logits'].ndim == 1


def test_train_model_can_print_verbose_epoch_config(tmp_path, capsys):
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
            dropout=0.02,
            encoder_finetune_strategy='full',
        ),
    )

    train_model(
        model,
        loader,
        None,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-6,
        device='cpu',
        output=tmp_path / 'mirafrag_verbose.pt',
        show_progress=False,
        verbose_epoch_config=True,
        swa=True,
        swa_start_epoch=1,
        swa_lr=3e-5,
    )

    output = capsys.readouterr().out
    assert 'epoch_config epoch=1/1' in output
    assert 'dropout=0.02' in output
    assert 'weight_decay=head=0.00e+00,encoder=1.00e-06' in output
    assert 'swa=True' in output
    assert 'swa_active=True' in output
    assert 'swa_lr=3e-05' in output


def test_train_model_swa_handles_integer_encoder_buffers(tmp_path):
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
            dropout=0.0,
        ),
    )

    history = train_model(
        model,
        loader,
        loader,
        epochs=2,
        lr=0.0,
        weight_decay=0.0,
        device='cpu',
        output=tmp_path / 'mirafrag_swa_integer_buffers.pt',
        show_progress=False,
        scheduler_name='none',
        checkpoint_metric='val_cosine',
        swa=True,
        swa_start_epoch=1,
    )

    assert history['swa_n_averaged'] == [1.0, 2.0]


def test_train_model_can_save_swa_checkpoint_by_val_cosine(tmp_path):
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
            dropout=0.0,
        ),
    )
    output = tmp_path / 'mirafrag_swa.pt'

    history = train_model(
        model,
        loader,
        loader,
        epochs=1,
        lr=0.0,
        weight_decay=0.0,
        device='cpu',
        output=output,
        show_progress=False,
        scheduler_name='none',
        checkpoint_metric='val_cosine',
        swa=True,
        swa_start_epoch=1,
    )

    payload = torch.load(output, map_location='cpu', weights_only=True)
    assert payload['train_config']['checkpoint_metric'] == 'val_cosine'
    assert payload['train_config']['swa'] is True
    assert payload['train_config']['swa_checkpoint'] is True
    assert payload['train_config']['swa_n_averaged'] == 1
    assert history['swa_n_averaged'] == [1.0]
    assert not math.isnan(history['swa_val_cosine'][0])


def test_train_model_can_checkpoint_by_train_loss(tmp_path):
    df = _tiny_training_df()
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=100.0)
    train_loader = _tiny_loader(df, graph_config, metadata)
    val_df = df.copy()
    val_df['intensities'] = ['10,1', '10,1']
    val_loader = _tiny_loader(val_df, graph_config, metadata)
    model = MiraFragModel(
        FakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            dropout=0.0,
        ),
    )
    output = tmp_path / 'mirafrag_train_best.pt'

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=2,
        lr=0.0,
        weight_decay=0.0,
        device='cpu',
        output=output,
        show_progress=False,
        scheduler_name='none',
        checkpoint_metric='train_loss',
    )

    payload = torch.load(output, map_location='cpu', weights_only=True)
    assert payload['train_config']['loss'] == 'cosine'
    assert payload['train_config']['checkpoint_metric'] == 'train_loss'
    assert payload['train_config']['prediction_probability_mode'] == 'joint'
    assert history['train_loss'][1] <= history['train_loss'][0]


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
