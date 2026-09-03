# ruff: noqa: F401
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import LazyLinear
from torch.utils.data import DataLoader

import mirafrag.cli.oracle as oracle_cli
from mirafrag.checkpoint import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint
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
from mirafrag.cli.train import (
    _collate_aux_retrieval_batch,
    _load_distillation_spectra,
    _maybe_rebuild_encoder_bond_adapter_model,
    _maybe_rebuild_encoder_metadata_adapter_model,
    _maybe_rebuild_fragment_bond_break_model,
    _retrieval_group_batches,
    _SafeAuxRetrievalDataset,
    _set_head_dropout,
    _validation_tune_candidates,
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
from mirafrag.encoders import load_foundation_encoder
from mirafrag.encoders.aimnet import AIMNET2_ATOMIC_NUMBERS, AIMNET2_R_MAX
from mirafrag.encoders.mace import repair_mace_cuequivariance_config
from mirafrag.encoders.small3d import SMALL3D_R_MAX, Small3DNodeEncoder
from mirafrag.evaluation import _sparse_prediction_rows
from mirafrag.fragments import (
    BOND_BREAK_FEATURE_DIM,
    FRAGMENT_EDGE_FEATURE_DIM,
    FRAGMENT_FEATURE_DIM,
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
from mirafrag.model import (
    AimnetMultipassAdapter,
    EncoderBondAdapter,
    MetadataConditionedEncoderAdapter,
    MiraFragModel,
    set_encoder_finetune_strategy,
)
from mirafrag.optim import (
    _build_scheduler,
    _optimizer_param_groups,
    _scheduler_total_steps,
)
from mirafrag.training import (
    _delta_penalty,
    _encoder_delta_reference,
    _head_delta_reference,
    train_model,
)
from tests.helpers import (
    FakeChargeEncoder,
    FakeCueProduct,
    FakeMace,
    FakeTrainAwareEncoder,
    _tiny_loader,
    _tiny_training_df,
)


class FakeAimnetLayoutEncoder(nn.Module):
    """Tiny encoder with AIMNet-style model.mlps parameter names."""

    node_feature_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.pre = nn.Linear(4, 4)
        self.model.mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4)) for _ in range(3)]
        )

    def forward(self, graph, **_kwargs):
        n = int(graph['x'].shape[0])
        return {'node_feats': torch.zeros(n, 4, device=graph['x'].device)}


class NonPersistentFakeMace(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'atomic_numbers', torch.tensor([1, 6, 8]), persistent=False
        )
        self.register_buffer('r_max', torch.tensor(5.0), persistent=False)
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, graph, **kwargs):
        return {'node_feats': self.proj(graph['node_attrs'].float())}


def test_rowwise_oracle_distillation_uses_per_row_teacher_weights(tmp_path):
    train_df = pd.DataFrame(
        {
            'identifier': ['row1', 'row2'],
            'mzs': ['[10.5]', '[20.5]'],
            'intensities': ['[1.0]', '[1.0]'],
            'PrecursorMZ': [100.0, 100.0],
        }
    )
    first = pd.DataFrame(
        {
            'identifier': ['row1', 'row2'],
            'pred_peaks': [
                json.dumps({'mz': [10.5], 'intensity': [1.0]}),
                json.dumps({'mz': [30.5], 'intensity': [1.0]}),
            ],
        }
    )
    second = pd.DataFrame(
        {
            'identifier': ['row1', 'row2'],
            'pred_peaks': [
                json.dumps({'mz': [30.5], 'intensity': [1.0]}),
                json.dumps({'mz': [20.5], 'intensity': [1.0]}),
            ],
        }
    )
    first_path = tmp_path / 'first.csv'
    second_path = tmp_path / 'second.csv'
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    args = SimpleNamespace(
        distill_loss_weight=1.0,
        distill_predictions=f'{first_path},{second_path}',
        distill_weights=None,
        distill_mode='rowwise-oracle',
        distill_oracle_temperature=0.0,
        distill_oracle_min_teacher_weight=0.0,
        distill_ce_min=None,
        distill_ce_max=None,
    )

    spectra = _load_distillation_spectra(
        args, train_df=train_df, mz_max=50.0, bin_width=1.0
    )

    assert spectra is not None
    assert spectra['row1'][0].tolist() == [10]
    assert spectra['row2'][0].tolist() == [20]
    assert torch.allclose(spectra['row1'][1], torch.tensor([1.0]))
    assert torch.allclose(spectra['row2'][1], torch.tensor([1.0]))


def test_retrieval_group_batches_do_not_split_query_groups():
    retrieval_df = pd.DataFrame(
        {
            '_retrieval_query_identifier': [
                'q1',
                'q1',
                'q1',
                'q2',
                'q2',
                'q3',
                'q3',
                'q3',
                'q3',
            ],
        }
    )

    batches = _retrieval_group_batches(retrieval_df, max_batch_size=4, seed=13)

    assert sorted(idx for batch in batches for idx in batch) == list(
        range(len(retrieval_df))
    )
    for query_id, positions in retrieval_df.groupby(
        '_retrieval_query_identifier',
        sort=False,
    ).indices.items():
        containing = [
            batch for batch in batches if any(idx in set(positions) for idx in batch)
        ]
        assert len(containing) == 1, query_id
        assert set(positions).issubset(set(containing[0]))


def test_retrieval_calibration_head_starts_as_noop():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=16,
            metadata_dim=8,
            encoder_finetune_strategy='full',
            retrieval_calibration_head=True,
        ),
    )

    pred = model(batch)

    assert 'retrieval_logit' in pred
    assert pred['retrieval_logit'].shape == (2,)
    assert torch.allclose(
        pred['retrieval_logit'], torch.zeros_like(pred['retrieval_logit'])
    )


class BrokenAuxRetrievalDataset:
    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        raise ValueError('bad candidate')


def test_safe_aux_retrieval_dataset_returns_empty_failed_batch():
    item = _SafeAuxRetrievalDataset(BrokenAuxRetrievalDataset())[0]

    assert item['_retrieval_failed'] is True
    assert 'bad candidate' in item['_retrieval_error']

    batch = _collate_aux_retrieval_batch([item])

    assert batch['_retrieval_empty'] is True
    assert batch['_retrieval_errors'] == ['bad candidate']


def test_head_delta_regularization_tracks_only_head_weight_matrices():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=16,
            metadata_dim=8,
            encoder_finetune_strategy='full',
        ),
    )
    with torch.no_grad():
        model(batch)
    head_reference = _head_delta_reference(model)
    encoder_reference = _encoder_delta_reference(model)

    assert head_reference
    assert encoder_reference
    assert _delta_penalty(model, head_reference).item() == pytest.approx(0.0)
    assert _delta_penalty(model, encoder_reference).item() == pytest.approx(0.0)

    with torch.no_grad():
        model.encoder.proj.weight.add_(1.0)
    assert _delta_penalty(model, head_reference).item() == pytest.approx(0.0)
    assert _delta_penalty(model, encoder_reference).item() > 0.0

    with torch.no_grad():
        model.head.scorer[0].weight.add_(1.0)
    assert _delta_penalty(model, head_reference).item() > 0.0


class FakeAimnetChargeEncoder(NonPersistentFakeMace):
    uses_molecular_charge = True

    def forward(self, graph, **kwargs):
        out = super().forward(graph, **kwargs)
        num_atoms = out['node_feats'].shape[0]
        charge = torch.linspace(-0.2, 0.3, num_atoms).unsqueeze(-1)
        out['aimnet_charge_features'] = torch.cat(
            [
                charge,
                charge.abs(),
                charge.clamp_min(0.0),
                (-charge).clamp_min(0.0),
                charge * 0.5,
                charge * 0.5,
            ],
            dim=-1,
        )
        return out


class FakeAimnetMultipassEncoder(FakeAimnetChargeEncoder):
    def __init__(self):
        super().__init__()
        self.export_multipass_features = False
        self.node_feature_dim = 3

    def forward(self, graph, **kwargs):
        out = super().forward(graph, **kwargs)
        if self.export_multipass_features:
            node_feats = out['node_feats']
            out['aimnet_multipass_features'] = torch.cat(
                [node_feats, node_feats.square(), node_feats.sin()], dim=-1
            )
        return out


class FakeAimnetFinalEncoder(NonPersistentFakeMace):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.mlps = nn.ModuleList(
            [
                nn.Linear(3, 3),
                nn.Linear(3, 3),
                nn.Sequential(
                    nn.Linear(3, 3),
                    nn.SiLU(),
                    nn.Linear(3, 3),
                    nn.SiLU(),
                    nn.Linear(3, 3),
                    nn.SiLU(),
                    nn.Linear(3, 3),
                ),
            ]
        )

    def forward(self, graph, **kwargs):
        node_feats = super().forward(graph, **kwargs)['node_feats']
        return {'node_feats': self.model.mlps[2][6](node_feats)}


def test_pool_fragment_atoms_allows_empty_pointer_ranges():
    node_feats = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    atom_index = torch.tensor([0, 1, 3], dtype=torch.long)
    atom_ptr = torch.tensor([0, 2, 2, 3], dtype=torch.long)

    pooled = FragmentSpectrumHead._pool_fragment_atoms(
        node_feats,
        atom_index,
        atom_ptr,
    )

    assert torch.allclose(pooled[0], node_feats[:2].mean(dim=0))
    assert torch.allclose(pooled[1], torch.zeros(3))
    assert torch.allclose(pooled[2], node_feats[3])


def test_oracle_checkpoint_config_reads_saved_graph_config_for_nonpersistent_encoder(
    tmp_path,
):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    output = tmp_path / 'mirafrag.pt'
    save_checkpoint(output, model)
    payload = torch.load(output, map_location='cpu', weights_only=True)

    assert payload['graph_config']['atomic_numbers'] == (1, 6, 8)
    assert not any(
        key.endswith('atomic_numbers') for key in payload['model_state_dict']
    )

    _config, _loaded_metadata, graph_config = _load_oracle_checkpoint_config(
        str(output),
        mz_max=32.0,
        bin_width=1.0,
    )

    assert graph_config.atomic_numbers == (1, 6, 8)
    assert graph_config.cutoff == 5.0


def test_oracle_checkpoint_config_uses_static_aimnet2_graph_config(
    tmp_path, monkeypatch
):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    config = MiraFragConfig(
        num_bins=32,
        hidden_dim=8,
        metadata_dim=4,
        encoder_type='aimnet',
        aimnet_model='aimnet2',
        aimnet_path=None,
    )
    output = tmp_path / 'old-aimnet-mirafrag.pt'
    torch.save(
        {
            'checkpoint_format': CHECKPOINT_FORMAT,
            'model_state_dict': {},
            'mirafrag_config': config.__dict__,
            'metadata_config': metadata.to_dict(),
            'train_config': {},
        },
        output,
    )
    monkeypatch.setattr(
        oracle_cli,
        'load_foundation_encoder',
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError('should not load encoder')
        ),
    )

    _config, _loaded_metadata, graph_config = _load_oracle_checkpoint_config(
        str(output),
        mz_max=32.0,
        bin_width=1.0,
    )

    assert graph_config.atomic_numbers == tuple(AIMNET2_ATOMIC_NUMBERS)
    assert graph_config.cutoff == AIMNET2_R_MAX


def test_oracle_checkpoint_config_uses_static_small3d_graph_config(
    tmp_path, monkeypatch
):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    config = MiraFragConfig(
        num_bins=32,
        hidden_dim=8,
        metadata_dim=4,
        encoder_type='small3d',
    )
    output = tmp_path / 'old-small3d-mirafrag.pt'
    torch.save(
        {
            'checkpoint_format': CHECKPOINT_FORMAT,
            'model_state_dict': {},
            'mirafrag_config': config.__dict__,
            'metadata_config': metadata.to_dict(),
            'train_config': {},
        },
        output,
    )
    monkeypatch.setattr(
        oracle_cli,
        'load_foundation_encoder',
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError('should not load encoder')
        ),
    )

    _config, _loaded_metadata, graph_config = _load_oracle_checkpoint_config(
        str(output),
        mz_max=32.0,
        bin_width=1.0,
    )

    assert graph_config.atomic_numbers == tuple(AIMNET2_ATOMIC_NUMBERS)
    assert graph_config.cutoff == SMALL3D_R_MAX


def test_small3d_encoder_runs_through_mirafrag_head():
    df = _tiny_training_df()
    encoder = Small3DNodeEncoder(hidden_dim=32, num_layers=1, num_radial=4)
    graph_config = GraphConfig(
        atomic_numbers=tuple(encoder.atomic_numbers.tolist()),
        cutoff=5.0,
    )
    metadata_config = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    loader = _tiny_loader(df, graph_config, metadata_config)
    batch = next(iter(loader))
    model = MiraFragModel(
        encoder,
        metadata_config=metadata_config,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=32,
            num_layers=1,
            metadata_dim=8,
            max_fragments=16,
            max_fragment_edges=64,
            encoder_type='small3d',
            encoder_finetune_strategy='full',
        ),
    )

    output = model(batch)

    assert output['logits'].shape[0] == batch['fragments']['features'].shape[0]
    assert output['oos_logits'].shape[0] == len(df)


def test_load_foundation_encoder_supports_small3d():
    encoder = load_foundation_encoder(encoder_type='small3d', device='cpu')

    assert isinstance(encoder, Small3DNodeEncoder)
    assert tuple(encoder.atomic_numbers.tolist()) == tuple(AIMNET2_ATOMIC_NUMBERS)


def test_oracle_checkpoint_config_falls_back_to_encoder_for_old_payload(
    tmp_path, monkeypatch
):
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    config = MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4)
    output = tmp_path / 'old-mirafrag.pt'
    torch.save(
        {
            'checkpoint_format': CHECKPOINT_FORMAT,
            'model_state_dict': {},
            'mirafrag_config': config.__dict__,
            'metadata_config': metadata.to_dict(),
            'train_config': {},
        },
        output,
    )
    monkeypatch.setattr(
        oracle_cli, 'load_foundation_encoder', lambda **_kwargs: FakeMace()
    )

    _config, _loaded_metadata, graph_config = _load_oracle_checkpoint_config(
        str(output),
        mz_max=32.0,
        bin_width=1.0,
    )

    assert graph_config.atomic_numbers == (1, 6, 8)
    assert graph_config.cutoff == 5.0


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


def test_validation_tune_candidates_are_limited_without_baseline():
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
    assert all(candidate.dropout in {0.0, 0.02} for candidate in candidates)
    assert all(
        candidate.encoder_weight_decay in {0.0, 1e-6} for candidate in candidates
    )
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
        (not candidate.swa)
        or candidate.swa_lr <= min(candidate.head_lr, candidate.encoder_lr)
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
    assert model.head.fragment_input_dropout.p == 0.25
    assert model.head.context_input_dropout.p == 0.25
    assert model.head.collision_input_dropout.p == 0.25
    assert model.head.oos_input_dropout.p == 0.25
    assert model.head.oos_hidden_dropout.p == 0.25
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


def test_aimnet_final_strategy_trains_only_final_projection():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeAimnetFinalEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='aimnet-final',
        ),
    )

    trainable = [
        name for name, param in model.encoder.named_parameters() if param.requires_grad
    ]

    assert trainable == [
        'model.mlps.2.6.weight',
        'model.mlps.2.6.bias',
    ]


def test_set_encoder_finetune_strategy_switches_to_aimnet_final():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeAimnetFinalEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='full',
        ),
    )
    assert any(
        name == 'proj.weight' and param.requires_grad
        for name, param in model.encoder.named_parameters()
    )

    set_encoder_finetune_strategy(model, 'aimnet-final')
    trainable = [
        name for name, param in model.encoder.named_parameters() if param.requires_grad
    ]

    assert model.config.encoder_finetune_strategy == 'aimnet-final'
    assert trainable == [
        'model.mlps.2.6.weight',
        'model.mlps.2.6.bias',
    ]


def test_aimnet_final_strategy_rejects_non_aimnet_encoder():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})

    with pytest.raises(ValueError, match='aimnet-final'):
        MiraFragModel(
            NonPersistentFakeMace(),
            metadata_config=metadata,
            config=MiraFragConfig(
                num_bins=16,
                hidden_dim=8,
                metadata_dim=4,
                encoder_finetune_strategy='aimnet-final',
            ),
        )


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
        head_delta_regularization=0.01,
        show_progress=False,
    )
    assert output.exists()
    assert history['epoch'] == [1]
    checkpoint_payload = torch.load(output, map_location='cpu', weights_only=True)
    assert checkpoint_payload['checkpoint_format'] == 'mirafrag-state-v1'
    assert checkpoint_payload['train_config']['head_delta_regularization'] == 0.01
    assert checkpoint_payload['train_config']['encoder_delta_regularization'] == 0.0
    assert 'model_state_dict' in checkpoint_payload
    assert 'model' not in checkpoint_payload
    monkeypatch.setattr(
        'mirafrag.checkpoint.load_foundation_encoder', lambda **_: FakeMace()
    )
    loaded, payload = load_checkpoint(output, device='cpu')
    assert payload['mirafrag_config']['encoder_finetune_strategy'] == 'head'
    batch = next(iter(loader))
    assert loaded(batch)['logits'].ndim == 1


def test_train_model_accepts_encoder_delta_regularization(tmp_path):
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
            encoder_finetune_strategy='full',
        ),
    )

    history = train_model(
        model,
        loader,
        None,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        device='cpu',
        output=tmp_path / 'mirafrag_encoder_delta.pt',
        encoder_delta_regularization=0.01,
        show_progress=False,
    )

    assert history['epoch'] == [1]


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
    assert 'weight_decay=head=0.00e+00,encoder_decay=1.00e-06' in output
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
        lr=1e-3,
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
        lr=1e-3,
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
        lr=1e-3,
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
        lr=1e-3,
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


def test_optimizer_param_groups_use_aimnet_layerwise_lr_decay():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        FakeAimnetLayoutEncoder(),
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
        head_lr=1e-3,
        encoder_lr=3e-4,
        head_weight_decay=0.0,
        encoder_weight_decay=1e-2,
        encoder_layer_lr_decay=0.1,
    )
    lr_by_name = {group['name']: group['lr'] for group in groups}
    wd_by_name = {group['name']: group['weight_decay'] for group in groups}

    assert lr_by_name['head'] == 1e-3
    assert lr_by_name['encoder_mlp2_decay'] == pytest.approx(3e-4)
    assert lr_by_name['encoder_mlp1_decay'] == pytest.approx(3e-5)
    assert lr_by_name['encoder_mlp0_decay'] == pytest.approx(3e-6)
    assert lr_by_name['encoder_base_decay'] == pytest.approx(3e-7)
    assert wd_by_name['encoder_mlp2_decay'] == 1e-2
    assert wd_by_name['encoder_base_decay'] == 1e-2


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
    assert {group['name'] for group in groups} == {'head', 'encoder_decay'}
    lr_by_name = {group['name']: group['lr'] for group in groups}
    wd_by_name = {group['name']: group['weight_decay'] for group in groups}
    assert lr_by_name['head'] == 1e-4
    assert lr_by_name['encoder_decay'] == 1e-4
    assert wd_by_name['head'] == 0.0
    assert wd_by_name['encoder_decay'] == 1e-8


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


def test_encoder_bond_adapter_starts_as_noop_on_node_features():
    adapter = EncoderBondAdapter(feature_dim=4, num_layers=1, dropout=0.0)
    node_feats = torch.randn(4, 3)
    graph = {
        'positions': torch.randn(4, 3),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    }

    adapted = adapter(node_feats, graph)

    assert adapted.shape == node_feats.shape
    assert torch.allclose(adapted, node_feats)


def test_model_encoder_bond_adapter_preserves_initial_encoder_features():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=16,
            metadata_dim=8,
            encoder_bond_adapter_layers=1,
            encoder_bond_adapter_feature_dim=4,
        ),
    )

    direct = model.encoder(batch['graph'])['node_feats'].float()
    adapted = model.encode_node_features(batch['graph'])

    assert torch.allclose(adapted, direct)


def test_encoder_metadata_adapter_starts_as_noop_on_node_features():
    adapter = MetadataConditionedEncoderAdapter(
        feature_dim=4, num_layers=1, dropout=0.0
    )
    node_feats = torch.randn(4, 3)
    graph = {
        'positions': torch.randn(4, 3),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        'batch': torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }
    metadata_features = torch.randn(2, 5)

    adapted = adapter(node_feats, graph, metadata_features)

    assert adapted.shape == node_feats.shape
    assert torch.allclose(adapted, node_feats)


def test_model_encoder_metadata_adapter_preserves_initial_encoder_features():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=16,
            metadata_dim=8,
            encoder_metadata_adapter_layers=1,
            encoder_metadata_adapter_feature_dim=4,
        ),
    )

    direct = model.encoder(batch['graph'])['node_feats'].float()
    metadata_features = model.metadata_features(batch)
    adapted = model.encode_node_features(
        batch['graph'], metadata_features=metadata_features
    )

    assert torch.allclose(adapted, direct)


def test_encoder_metadata_adapter_materialized_state_loads_into_fresh_model():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    config = MiraFragConfig(
        num_bins=100,
        hidden_dim=16,
        metadata_dim=8,
        encoder_metadata_adapter_layers=1,
        encoder_metadata_adapter_feature_dim=4,
    )
    trained = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=config,
    )
    trained.eval()
    with torch.no_grad():
        trained(batch)
    state_dict = trained.state_dict()

    fresh = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=config,
    )
    incompatible = fresh.load_state_dict(state_dict)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_aimnet_multipass_adapter_starts_as_noop_on_node_features():
    adapter = AimnetMultipassAdapter(feature_dim=4, dropout=0.0)
    node_feats = torch.randn(5, 3)
    aux = {'aimnet_multipass_features': torch.randn(5, 9)}

    adapted = adapter(node_feats, aux)

    assert adapted.shape == node_feats.shape
    assert torch.allclose(adapted, node_feats)


def test_aimnet_charge_branch_starts_as_noop_on_rebuilt_checkpoint():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    base = MiraFragModel(
        FakeAimnetChargeEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=100, hidden_dim=16, metadata_dim=8),
    )
    base.eval()
    with torch.no_grad():
        base_pred = base(batch)
    args = SimpleNamespace(
        fragment_bond_break_layers=None,
        fragment_path_layers=None,
        fragment_path_primary=None,
        fragment_action_path_layers=None,
        ce_basis_features=None,
        aimnet_charge_features=True,
    )

    charge = _maybe_rebuild_fragment_bond_break_model(base, args, device='cpu')
    charge.eval()
    with torch.no_grad():
        charge_pred = charge(batch)

    assert torch.allclose(charge_pred['logits'], base_pred['logits'])


def test_aimnet_multipass_branch_starts_as_noop_on_rebuilt_checkpoint():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    base = MiraFragModel(
        FakeAimnetMultipassEncoder(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=100, hidden_dim=16, metadata_dim=8),
    )
    base.eval()
    with torch.no_grad():
        base_pred = base(batch)
    args = SimpleNamespace(
        fragment_bond_break_layers=None,
        fragment_path_layers=None,
        fragment_path_primary=None,
        fragment_action_path_layers=None,
        ce_basis_features=None,
        aimnet_charge_features=None,
        aimnet_multipass_features=True,
    )

    rebuilt = _maybe_rebuild_fragment_bond_break_model(base, args, device='cpu')
    rebuilt.eval()
    assert rebuilt.config.aimnet_multipass_features is True
    assert rebuilt.aimnet_multipass_adapter is not None
    assert rebuilt.encoder.export_multipass_features is True
    with torch.no_grad():
        rebuilt_pred = rebuilt(batch)

    assert torch.allclose(rebuilt_pred['logits'], base_pred['logits'])


def test_aimnet_charge_features_require_encoder_auxiliary_output():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=16,
            metadata_dim=8,
            aimnet_charge_features=True,
        ),
    )

    with pytest.raises(ValueError, match='aimnet_charge_features'):
        model(batch)


def test_aimnet_multipass_materialized_state_loads_into_fresh_model():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    config = MiraFragConfig(
        num_bins=100,
        hidden_dim=16,
        metadata_dim=8,
        aimnet_multipass_features=True,
    )
    trained = MiraFragModel(
        FakeAimnetMultipassEncoder(),
        metadata_config=metadata,
        config=config,
    )
    trained.eval()
    with torch.no_grad():
        trained(batch)
    state_dict = trained.state_dict()

    fresh = MiraFragModel(
        FakeAimnetMultipassEncoder(),
        metadata_config=metadata,
        config=config,
    )
    incompatible = fresh.load_state_dict(state_dict)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_aimnet_multipass_features_require_encoder_node_feature_dim():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})

    with pytest.raises(ValueError, match='node_feature_dim'):
        MiraFragModel(
            NonPersistentFakeMace(),
            metadata_config=metadata,
            config=MiraFragConfig(
                num_bins=100,
                hidden_dim=16,
                metadata_dim=8,
                aimnet_multipass_features=True,
            ),
        )


def test_rebuild_aimnet_charge_features_model_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(
        fragment_bond_break_layers=None,
        fragment_path_layers=None,
        fragment_path_primary=None,
        fragment_action_path_layers=None,
        ce_basis_features=None,
        aimnet_charge_features=True,
    )

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.aimnet_charge_features is True
    assert rebuilt.head.aimnet_charge_residual is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_rebuild_encoder_bond_adapter_model_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(
        encoder_bond_adapter_layers=1,
        encoder_bond_adapter_feature_dim=16,
        encoder_bond_adapter_dropout=0.1,
    )

    rebuilt = _maybe_rebuild_encoder_bond_adapter_model(model, args, device='cpu')

    assert rebuilt.config.encoder_bond_adapter_layers == 1
    assert rebuilt.config.encoder_bond_adapter_feature_dim == 16
    assert rebuilt.config.encoder_bond_adapter_dropout == 0.1
    assert rebuilt.encoder_bond_adapter is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_rebuild_encoder_metadata_adapter_model_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(
        encoder_metadata_adapter_layers=1,
        encoder_metadata_adapter_feature_dim=16,
        encoder_metadata_adapter_dropout=0.1,
    )

    rebuilt = _maybe_rebuild_encoder_metadata_adapter_model(model, args, device='cpu')

    assert rebuilt.config.encoder_metadata_adapter_layers == 1
    assert rebuilt.config.encoder_metadata_adapter_feature_dim == 16
    assert rebuilt.config.encoder_metadata_adapter_dropout == 0.1
    assert rebuilt.encoder_metadata_adapter is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_rebuild_encoder_metadata_adapter_model_preserves_initial_predictions():
    df = _tiny_training_df()
    metadata = MetadataConfig.from_dataframe(df, precursor_mz_max=1000.0)
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(df, graph_config, metadata)))
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=100, hidden_dim=16, metadata_dim=8),
    )
    model.eval()
    with torch.no_grad():
        base_pred = model(batch)
    args = SimpleNamespace(
        encoder_metadata_adapter_layers=1,
        encoder_metadata_adapter_feature_dim=16,
        encoder_metadata_adapter_dropout=0.1,
    )

    rebuilt = _maybe_rebuild_encoder_metadata_adapter_model(model, args, device='cpu')
    rebuilt.eval()
    with torch.no_grad():
        rebuilt_pred = rebuilt(batch)

    assert rebuilt.encoder_metadata_adapter is not None
    for key, base_value in base_pred.items():
        if isinstance(base_value, torch.Tensor) and key in rebuilt_pred:
            assert torch.allclose(rebuilt_pred[key], base_value)


def test_bond_break_logits_start_as_neutral_auxiliary_scores():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_bond_break_layers=2,
        )
    )
    node_feats = torch.randn(4, 5)
    formula_features = torch.randn(2, 8)
    context_features = torch.randn(2, 8)
    collision_features = torch.randn(2, 8)
    fragments = {
        'bond_atom_index': torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        'bond_ptr': torch.tensor([0, 1, 2], dtype=torch.long),
        'bond_features': torch.tensor(
            [[1.0 / 3.0, 0.5, 0.0], [2.0 / 3.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
    }

    logits = head._bond_break_logits(
        node_feats,
        formula_features,
        context_features,
        collision_features,
        fragments,
    )

    assert logits.shape == (2,)
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_rebuild_fragment_bond_break_model_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(fragment_bond_break_layers=1)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_bond_break_layers == 1
    assert rebuilt.head.bond_break_pair_scorer is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_fragment_action_primary_delta_starts_as_noop_residual():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=2,
        )
    )
    node_feats = torch.randn(4, 5)
    formula_features = torch.randn(2, 8)
    context_features = torch.randn(2, 8)
    collision_features = torch.randn(2, 8)
    fragment_descriptor = torch.randn(2, FRAGMENT_FEATURE_DIM)
    formula_batch = torch.zeros(2, dtype=torch.long)
    base_logits = torch.randn(2)
    fragments = {
        'bond_atom_index': torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        'bond_ptr': torch.tensor([0, 1, 2], dtype=torch.long),
        'bond_features': torch.randn(2, BOND_BREAK_FEATURE_DIM),
    }

    delta = head._fragment_action_primary_delta(
        node_feats,
        formula_features,
        context_features,
        collision_features,
        fragment_descriptor,
        fragments,
        formula_batch,
        batch_size=1,
        base_logits=base_logits,
    )

    assert delta.shape == (2,)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_fragment_action_primary_ce_gate_starts_as_noop_residual():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=2,
            fragment_action_primary_ce_gate=True,
        )
    )
    node_feats = torch.randn(4, 5)
    formula_features = torch.randn(2, 8)
    context_features = torch.randn(2, 8)
    collision_features = torch.randn(2, 8)
    fragment_descriptor = torch.randn(2, FRAGMENT_FEATURE_DIM)
    formula_batch = torch.zeros(2, dtype=torch.long)
    base_logits = torch.randn(2)
    fragments = {
        'bond_atom_index': torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        'bond_ptr': torch.tensor([0, 1, 2], dtype=torch.long),
        'bond_features': torch.randn(2, BOND_BREAK_FEATURE_DIM),
    }

    delta = head._fragment_action_primary_delta(
        node_feats,
        formula_features,
        context_features,
        collision_features,
        fragment_descriptor,
        fragments,
        formula_batch,
        batch_size=1,
        base_logits=base_logits,
    )

    assert head.fragment_action_primary_ce_gate is not None
    assert delta.shape == (2,)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_fragment_action_geometry_features_are_invariant_event_scalars():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
            fragment_action_geometry_features=True,
        )
    )
    assert head.bond_break_geometry_features is True

    fragments = {
        'formula_batch': torch.zeros(2, dtype=torch.long),
        'atom_index': torch.tensor([0, 1, 1, 2], dtype=torch.long),
        'atom_ptr': torch.tensor([0, 2, 4], dtype=torch.long),
    }
    graph = {
        'positions': torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        'atomic_numbers': torch.tensor([6, 6, 8], dtype=torch.long),
        'batch': torch.zeros(3, dtype=torch.long),
    }

    features = head._fragment_action_geometry_event_features(
        fragments,
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        graph,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()
    assert torch.allclose(features[:, 0], torch.tensor([0.375, 0.375]))
    assert torch.all(features[:, 2] > 0)


def test_rebuild_fragment_action_geometry_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    old_residual_weight = (
        model.head.fragment_action_primary_residual[0].weight.detach().clone()
    )
    args = SimpleNamespace(fragment_action_geometry_features=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.bond_break_geometry_features is True
    assert rebuilt.config.fragment_action_geometry_features is True
    assert rebuilt.head.bond_break_geometry_features is True
    assert rebuilt.head.fragment_action_primary_pair_scorer is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)
    assert torch.allclose(
        rebuilt.head.fragment_action_primary_residual[0].weight,
        old_residual_weight,
    )


def test_rebuild_fragment_action_geometry_starts_as_noop_prediction():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    model.eval()
    with torch.no_grad():
        base_pred = model(batch)
    args = SimpleNamespace(fragment_action_geometry_features=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')
    rebuilt.eval()
    with torch.no_grad():
        rebuilt_pred = rebuilt(batch)

    assert rebuilt.config.bond_break_geometry_features is True
    assert torch.allclose(rebuilt_pred['logits'], base_pred['logits'], atol=1e-6)
    assert torch.allclose(rebuilt_pred['oos_logits'], base_pred['oos_logits'])


def test_fragment_action_primary_forward_uses_bond_provenance_after_residual_unzero():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=100,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    fragment_config = fragment_config_from_model_config(model.config)
    dataset = BinnedSpectrumDataset(
        _tiny_training_df(),
        graph_config=graph_config,
        metadata_config=metadata,
        mz_max=100.0,
        bin_width=1.0,
        include_fragments=True,
        fragment_config=fragment_config,
        require_spectrum=True,
    )
    batch = collate_spectrum_batch([dataset[0]])
    fragments = batch['fragments']
    assert fragments['bond_atom_index'].ndim == 2
    assert fragments['bond_ptr'].numel() == fragments['formula_batch'].numel() + 1
    assert fragments['bond_features'].shape[1] == BOND_BREAK_FEATURE_DIM

    model.eval()
    with torch.no_grad():
        base = model(batch)['logits']
        final_layer = model.head.fragment_action_primary_residual[-1]
        final_layer.weight.fill_(0.1)
        final_layer.bias.fill_(0.05)
        changed = model(batch)['logits']

    assert not torch.allclose(changed, base)


def test_fragment_action_primary_gradients_reach_pair_scorer_after_residual_opens():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        )
    )
    node_feats = torch.randn(4, 5, requires_grad=True)
    formula_features = torch.randn(2, 8)
    context_features = torch.randn(2, 8)
    collision_features = torch.randn(2, 8)
    fragment_descriptor = torch.randn(2, FRAGMENT_FEATURE_DIM)
    formula_batch = torch.zeros(2, dtype=torch.long)
    base_logits = torch.randn(2)
    fragments = {
        'bond_atom_index': torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        'bond_ptr': torch.tensor([0, 1, 2], dtype=torch.long),
        'bond_features': torch.randn(2, BOND_BREAK_FEATURE_DIM),
    }
    head.train()
    _ = head._fragment_action_primary_delta(
        node_feats,
        formula_features,
        context_features,
        collision_features,
        fragment_descriptor,
        fragments,
        formula_batch,
        batch_size=1,
        base_logits=base_logits,
    )
    final_layer = head.fragment_action_primary_residual[-1]
    with torch.no_grad():
        final_layer.weight.fill_(0.1)
        final_layer.bias.zero_()

    delta = head._fragment_action_primary_delta(
        node_feats,
        formula_features,
        context_features,
        collision_features,
        fragment_descriptor,
        fragments,
        formula_batch,
        batch_size=1,
        base_logits=base_logits,
    )
    loss = delta.square().sum()
    loss.backward()

    pair_grads = [
        param.grad
        for param in head.fragment_action_primary_pair_scorer.parameters()
        if param.grad is not None
    ]
    assert pair_grads
    assert any(torch.count_nonzero(grad).item() > 0 for grad in pair_grads)


def test_rebuild_fragment_action_primary_ce_gate_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    old_residual_weight = (
        model.head.fragment_action_primary_residual[0].weight.detach().clone()
    )
    args = SimpleNamespace(fragment_action_primary_ce_gate=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_action_primary_layers == 1
    assert rebuilt.config.fragment_action_primary_ce_gate is True
    assert rebuilt.head.fragment_action_primary_ce_gate is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)
    assert torch.allclose(
        rebuilt.head.fragment_action_primary_residual[0].weight,
        old_residual_weight,
    )


def test_rebuild_fragment_action_primary_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(fragment_action_primary_layers=2)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_action_primary_layers == 2
    assert rebuilt.config.bond_break_geometry_features is True
    assert rebuilt.config.fragment_action_geometry_features is True
    assert rebuilt.head.bond_break_geometry_features is True
    assert rebuilt.head.fragment_action_primary_pair_scorer is not None
    assert rebuilt.head.fragment_action_primary_residual is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_rebuild_existing_fragment_action_branch_keeps_geometry_default_when_unset():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    args = SimpleNamespace()

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt is model
    assert rebuilt.config.bond_break_geometry_features is False
    assert rebuilt.config.fragment_action_geometry_features is False
    assert rebuilt.head.bond_break_geometry_features is False


def test_rebuild_fragment_action_primary_can_disable_default_geometry():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    args = SimpleNamespace(
        fragment_action_primary_layers=2,
        bond_break_geometry_features=False,
    )

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_action_primary_layers == 2
    assert rebuilt.config.bond_break_geometry_features is False
    assert rebuilt.config.fragment_action_geometry_features is False
    assert rebuilt.head.bond_break_geometry_features is False
    assert rebuilt.head.fragment_action_primary_pair_scorer is not None


def test_action_bond_gnn_model_config_requests_bond_break_cache_fields():
    config = MiraFragConfig(num_bins=16, fragment_action_bond_gnn_layers=2)

    fragment_config = fragment_config_from_model_config(config)

    assert fragment_config.include_bond_breaks is True


def test_rebuild_fragment_action_bond_gnn_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    old_action_weight = (
        model.head.fragment_action_primary_residual[0].weight.detach().clone()
    )
    args = SimpleNamespace(fragment_action_bond_gnn_layers=2)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_action_primary_layers == 1
    assert rebuilt.config.fragment_action_bond_gnn_layers == 2
    assert rebuilt.head.fragment_action_bond_gnn_encoder is not None
    assert len(rebuilt.head.fragment_action_bond_gnn_layers_module) == 2
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)
    assert torch.allclose(
        rebuilt.head.fragment_action_primary_residual[0].weight,
        old_action_weight,
    )


def test_action_primary_model_config_requests_bond_break_cache_fields():
    config = MiraFragConfig(num_bins=16, fragment_action_primary_layers=2)

    fragment_config = fragment_config_from_model_config(config)

    assert fragment_config.include_bond_breaks is True


def test_ce_basis_features_expand_collision_input_channels():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            ce_basis_features=True,
        )
    )
    metadata_features = torch.tensor(
        [
            [0.5, -1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features = head._collision_energy_feature(metadata_features)

    assert features.shape == (2, 7)
    assert torch.allclose(features[:, 0], metadata_features[:, 1])
    assert torch.isfinite(features).all()


def test_ce_fourier_embedding_expands_collision_input_channels():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            ce_embedding='fourier',
            ce_fourier_frequencies=3,
        )
    )
    metadata_features = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    features = head._collision_energy_feature(metadata_features)

    assert features.shape == (2, 7)
    assert torch.allclose(features[:, 0], metadata_features[:, 1])
    assert torch.allclose(features[0, 1:4], torch.zeros(3))
    assert torch.allclose(features[0, 4:], torch.ones(3))
    assert torch.isfinite(features).all()


def test_rebuild_ce_basis_model_copies_scalar_collision_weight():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    with torch.no_grad():
        old_weight = torch.arange(8, dtype=torch.float32).view(8, 1)
        model.head.collision_encoder[0].weight.copy_(old_weight)
    args = SimpleNamespace(ce_basis_features=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    new_weight = rebuilt.head.collision_encoder[0].weight.detach()
    assert rebuilt.config.ce_basis_features is True
    assert rebuilt.head.ce_basis_features is True
    assert new_weight.shape == (8, 7)
    assert torch.allclose(new_weight[:, :1], old_weight)
    assert torch.allclose(new_weight[:, 1:], torch.zeros_like(new_weight[:, 1:]))


def test_rebuild_ce_fourier_model_copies_scalar_collision_weight():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    with torch.no_grad():
        old_weight = torch.arange(8, dtype=torch.float32).view(8, 1)
        model.head.collision_encoder[0].weight.copy_(old_weight)
    args = SimpleNamespace(
        ce_basis_features=None,
        ce_embedding='fourier',
        ce_fourier_frequencies=4,
    )

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    new_weight = rebuilt.head.collision_encoder[0].weight.detach()
    assert rebuilt.config.ce_basis_features is False
    assert rebuilt.config.ce_embedding == 'fourier'
    assert rebuilt.config.ce_fourier_frequencies == 4
    assert rebuilt.head.ce_embedding == 'fourier'
    assert new_weight.shape == (8, 9)
    assert torch.allclose(new_weight[:, :1], old_weight)
    assert torch.allclose(new_weight[:, 1:], torch.zeros_like(new_weight[:, 1:]))


def test_fragment_path_primary_bypasses_generic_formula_scorer():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_path_layers=2,
            fragment_path_primary=True,
        )
    )
    formula_features = torch.randn(3, 8)
    context_features = torch.randn(3, 8)
    collision_features = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.zeros(2, FRAGMENT_EDGE_FEATURE_DIM)
    edge_attr[:, 0] = 1.0

    path_logits = head._fragment_path_log_scores(
        formula_features,
        context_features,
        collision_features,
        edge_index,
        edge_attr,
    )

    assert path_logits.shape == (3,)
    assert torch.isfinite(path_logits).all()
    assert not torch.allclose(path_logits, torch.zeros_like(path_logits))


def test_candidate_suppression_gate_starts_as_noop():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            candidate_suppression_gate_hidden_dim=16,
            candidate_suppression_gate_initial_penalty=1e-6,
        )
    )
    candidate_features = torch.randn(4, 5 * 8)
    fragment_descriptor = torch.randn(4, FRAGMENT_FEATURE_DIM)
    formula_logits = torch.tensor([0.5, -0.2, 1.0, 0.1])
    formula_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    penalty = head._candidate_suppression_penalty(
        candidate_features,
        fragment_descriptor,
        formula_logits,
        formula_batch,
        batch_size=2,
    )

    assert penalty.shape == formula_logits.shape
    assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-7)


def test_candidate_suppression_gate_only_suppresses_after_opening():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            candidate_suppression_gate_hidden_dim=16,
            candidate_suppression_gate_initial_penalty=1e-6,
        )
    )
    assert head.candidate_suppression_gate is not None
    with torch.no_grad():
        final = head.candidate_suppression_gate[-1]
        final.bias.fill_(0.0)
    candidate_features = torch.randn(4, 5 * 8)
    fragment_descriptor = torch.randn(4, FRAGMENT_FEATURE_DIM)
    formula_logits = torch.tensor([0.5, -0.2, 1.0, 0.1])
    formula_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    penalty = head._candidate_suppression_penalty(
        candidate_features,
        fragment_descriptor,
        formula_logits,
        formula_batch,
        batch_size=2,
    )

    assert torch.all(penalty >= 0)
    assert torch.any(penalty > 0)


def test_rebuild_candidate_suppression_gate_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    model.eval()
    with torch.no_grad():
        base_pred = model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(
        candidate_suppression_gate_hidden_dim=16,
        candidate_suppression_gate_dropout=0.0,
        candidate_suppression_gate_initial_penalty=1e-6,
    )

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')
    rebuilt.eval()
    with torch.no_grad():
        gated_pred = rebuilt(batch)

    assert rebuilt.config.candidate_suppression_gate_hidden_dim == 16
    assert rebuilt.head.candidate_suppression_gate is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)
    assert torch.allclose(gated_pred['logits'], base_pred['logits'], atol=1e-6)
    assert torch.allclose(gated_pred['oos_logits'], base_pred['oos_logits'])


def test_fragment_action_path_delta_starts_as_noop_residual():
    head = FragmentSpectrumHead(
        MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_path_layers=2,
        )
    )
    fragment_descriptor = torch.randn(3, FRAGMENT_FEATURE_DIM)
    collision_inputs = torch.randn(3, 1)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.zeros(2, FRAGMENT_EDGE_FEATURE_DIM)
    edge_attr[:, 0] = 1.0
    formula_batch = torch.zeros(3, dtype=torch.long)
    base_logits = torch.randn(3)

    delta = head._fragment_action_path_delta(
        fragment_descriptor,
        collision_inputs,
        edge_index,
        edge_attr,
        formula_batch,
        batch_size=1,
        base_logits=base_logits,
    )

    assert delta.shape == (3,)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_rebuild_fragment_action_path_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(num_bins=32, hidden_dim=8, metadata_dim=4),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(fragment_action_path_layers=2)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_action_path_layers == 2
    assert rebuilt.head.fragment_action_residual is not None
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)


def test_rebuild_fragment_path_primary_keeps_checkpoint_path_layers_when_unset():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_path_layers=2,
        ),
    )
    args = SimpleNamespace(fragment_path_layers=None, fragment_path_primary=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_path_layers == 2
    assert rebuilt.config.fragment_path_primary is True


def test_rebuild_fragment_path_primary_preserves_existing_weights():
    metadata = MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})
    model = MiraFragModel(
        NonPersistentFakeMace(),
        metadata_config=metadata,
        config=MiraFragConfig(
            num_bins=32,
            hidden_dim=8,
            metadata_dim=4,
            fragment_path_layers=2,
        ),
    )
    graph_config = GraphConfig(atomic_numbers=(1, 6, 8), cutoff=5.0, seed=7)
    batch = next(iter(_tiny_loader(_tiny_training_df(), graph_config, metadata)))
    with torch.no_grad():
        model(batch)
    old_scorer_weight = model.head.scorer[0].weight.detach().clone()
    args = SimpleNamespace(fragment_path_layers=2, fragment_path_primary=True)

    rebuilt = _maybe_rebuild_fragment_bond_break_model(model, args, device='cpu')

    assert rebuilt.config.fragment_path_primary is True
    assert rebuilt.head.fragment_path_primary is True
    assert torch.allclose(rebuilt.head.scorer[0].weight, old_scorer_weight)
