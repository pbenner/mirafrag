import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from mirafrag.data import BinnedSpectrumDataset, collate_spectrum_batch
from mirafrag.fragments import FragmentConfig


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
