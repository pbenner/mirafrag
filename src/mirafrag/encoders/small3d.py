from __future__ import annotations

import torch
from torch import nn

from mirafrag.encoders.aimnet import AIMNET2_ATOMIC_NUMBERS

SMALL3D_R_MAX = 5.0


class Small3DNodeEncoder(nn.Module):
    """
    Compact train-from-scratch 3D atom encoder.

    The encoder is deliberately much smaller than the foundation encoders. It uses
    atomic embeddings and radial message passing over atoms within a fixed cutoff,
    producing node features for the existing MiraFrag spectrum head.
    """

    uses_molecular_charge = False

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_radial: int = 16,
        cutoff: float = SMALL3D_R_MAX,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive.')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive.')
        if num_radial <= 0:
            raise ValueError('num_radial must be positive.')
        if cutoff <= 0:
            raise ValueError('cutoff must be positive.')

        atomic_numbers = torch.tensor(AIMNET2_ATOMIC_NUMBERS, dtype=torch.long)
        self.register_buffer('atomic_numbers', atomic_numbers, persistent=False)
        self.register_buffer('r_max', torch.tensor(float(cutoff)), persistent=False)
        centers = torch.linspace(0.0, float(cutoff), int(num_radial))
        self.register_buffer('radial_centers', centers, persistent=False)
        spacing = float(cutoff) / max(int(num_radial) - 1, 1)
        self.radial_gamma = 1.0 / max(spacing, 1e-6) ** 2

        self.atomic_embedding = nn.Embedding(
            int(atomic_numbers.max().item()) + 1, hidden_dim
        )
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_radial, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )
        self.node_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        graph: dict[str, torch.Tensor],
        *,
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_node_feats: bool = True,
        molecular_charge: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del training, compute_force, compute_virials, compute_stress
        del compute_node_feats, molecular_charge
        atomic_numbers = graph['atomic_numbers'].long()
        positions = graph['positions']
        batch = graph.get('batch')
        if batch is None:
            batch = torch.zeros(
                atomic_numbers.shape[0], dtype=torch.long, device=atomic_numbers.device
            )
        else:
            batch = batch.to(device=atomic_numbers.device, dtype=torch.long)

        h = self.atomic_embedding(atomic_numbers)
        src, dst, distances = self._radius_edges(positions, batch)
        if src.numel() == 0:
            return {'node_feats': self.output_norm(h)}
        radial = self._radial_basis(distances)
        inv_degree = self._inverse_degree(dst, h.shape[0], h.dtype)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps, strict=True):
            weights = edge_mlp(radial)
            messages = weights * h[src]
            aggregated = h.new_zeros(h.shape)
            aggregated.index_add_(0, dst, messages)
            aggregated = aggregated * inv_degree.unsqueeze(-1)
            h = h + node_mlp(aggregated)
        return {'node_feats': self.output_norm(h)}

    def _radius_edges(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sources: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        distances: list[torch.Tensor] = []
        cutoff = float(self.r_max.detach().cpu().item())
        for molecule_id in torch.unique(batch, sorted=True):
            atom_idx = torch.nonzero(batch == molecule_id, as_tuple=False).flatten()
            if atom_idx.numel() <= 1:
                continue
            molecule_positions = positions[atom_idx]
            distance_matrix = torch.cdist(molecule_positions, molecule_positions)
            mask = (distance_matrix <= cutoff) & (
                ~torch.eye(atom_idx.numel(), dtype=torch.bool, device=positions.device)
            )
            local_src, local_dst = torch.nonzero(mask, as_tuple=True)
            if local_src.numel() == 0:
                continue
            sources.append(atom_idx[local_src])
            destinations.append(atom_idx[local_dst])
            distances.append(distance_matrix[local_src, local_dst])
        if not sources:
            empty_idx = torch.empty(0, dtype=torch.long, device=positions.device)
            empty_dist = torch.empty(0, dtype=positions.dtype, device=positions.device)
            return empty_idx, empty_idx, empty_dist
        return (
            torch.cat(sources),
            torch.cat(destinations),
            torch.cat(distances),
        )

    def _radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        centers = self.radial_centers.to(device=distances.device, dtype=distances.dtype)
        return torch.exp(-self.radial_gamma * (distances.unsqueeze(-1) - centers) ** 2)

    def _inverse_degree(
        self,
        destination: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        degree = torch.zeros(num_nodes, dtype=dtype, device=destination.device)
        degree.index_add_(0, destination, torch.ones_like(destination, dtype=dtype))
        return degree.clamp_min(1.0).reciprocal()


def load_small3d_encoder(
    *,
    device: str | torch.device = 'cpu',
) -> Small3DNodeEncoder:
    """
    Build the compact train-from-scratch 3D encoder.
    """
    return Small3DNodeEncoder().to(device)
