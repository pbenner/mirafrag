import pytest
import torch
from torch import nn

from mirafrag.cli.train import _set_pair_action_only
from mirafrag.config import MiraFragConfig
from mirafrag.data import MetadataConfig
from mirafrag.heads.pair_features import (
    align_directional_pair_features,
    unique_candidate_pair_index,
)
from mirafrag.model import MiraFragModel, set_encoder_finetune_strategy


class PairCapableEncoder(nn.Module):
    supports_pair_representations = True

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 4)
        self.top = nn.Linear(4, 4)
        self.selected_layers = 0

    def configure_trainable_top_layers(self, count: int) -> None:
        self.selected_layers = int(count)
        if count:
            for param in self.parameters():
                param.requires_grad_(False)
            for param in self.top.parameters():
                param.requires_grad_(True)


def _metadata() -> MetadataConfig:
    return MetadataConfig(adduct_to_idx={'[M+H]+': 0}, instrument_to_idx={'HCD': 0})


def test_pair_index_is_unique_and_directional_features_align() -> None:
    fragments = {'bond_atom_index': torch.tensor([[3, 1], [1, 3], [2, 4]])}
    pair_index = unique_candidate_pair_index(fragments)
    assert torch.equal(pair_index, torch.tensor([[1, 3], [2, 4]]))

    pair_features = torch.tensor([[10.0, 11.0, 20.0, 21.0], [30.0, 31.0, 40.0, 41.0]])
    aligned = align_directional_pair_features(
        pair_features,
        pair_index,
        fragments['bond_atom_index'],
        num_nodes=5,
    )
    assert torch.equal(
        aligned,
        torch.tensor(
            [
                [20.0, 21.0, 10.0, 11.0],
                [10.0, 11.0, 20.0, 21.0],
                [30.0, 31.0, 40.0, 41.0],
            ]
        ),
    )


def test_pair_alignment_rejects_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match='valid graph atom rows'):
        align_directional_pair_features(
            torch.zeros(1, 4),
            torch.tensor([[0, 3]]),
            torch.tensor([[0, 3]]),
            num_nodes=3,
        )


def test_pair_action_only_trains_only_pair_branch_and_selected_encoder() -> None:
    encoder = PairCapableEncoder()
    model = MiraFragModel(
        encoder,
        metadata_config=_metadata(),
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            fragment_action_primary_layers=1,
            bond_break_pair_features=True,
            bond_break_pair_dim=4,
        ),
    )

    _set_pair_action_only(model, encoder_layers=1)

    assert model.pair_action_only
    assert encoder.selected_layers == 1
    assert all(
        param.requires_grad
        for name, param in model.head.named_parameters()
        if name.startswith('fragment_action_pair_')
    )
    assert all(
        not param.requires_grad
        for name, param in model.head.named_parameters()
        if not name.startswith('fragment_action_pair_')
    )
    assert all(param.requires_grad for param in encoder.top.parameters())
    assert all(not param.requires_grad for param in encoder.base.parameters())


def test_full_strategy_does_not_unfreeze_encoder_in_pair_action_only_mode() -> None:
    encoder = PairCapableEncoder()
    model = MiraFragModel(
        encoder,
        metadata_config=_metadata(),
        config=MiraFragConfig(
            num_bins=16,
            hidden_dim=8,
            metadata_dim=4,
            encoder_finetune_strategy='full',
            bond_break_pair_features=True,
            bond_break_pair_dim=4,
        ),
    )

    _set_pair_action_only(model, encoder_layers=0)
    set_encoder_finetune_strategy(model, 'full')

    assert all(not param.requires_grad for param in encoder.parameters())
