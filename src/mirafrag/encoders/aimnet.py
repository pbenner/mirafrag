from __future__ import annotations

import torch
from torch import nn

AIMNET2_ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)


def load_aimnet_encoder(
    *,
    model: str | None = 'aimnet2',
    model_path: str | None = None,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    return AimnetNodeEncoder(
        model=model_path or model or 'aimnet2',
        device=device,
    )


class AimnetNodeEncoder(nn.Module):
    """Expose AIMNet2 hidden atom features through MiraFrag's encoder interface."""

    uses_molecular_charge = True

    def __init__(
        self,
        *,
        model: str = 'aimnet2',
        device: str | torch.device = 'cpu',
    ) -> None:
        super().__init__()
        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as exc:
            raise ImportError(
                'AIMNet support requires the aimnet package. Install the local '
                'aimnetcentral checkout or run `uv sync --extra aimnet`.'
            ) from exc

        self.model_name = str(model)
        self.calculator = AIMNet2Calculator(
            self.model_name,
            device=str(device),
            train=True,
        )
        self.model = self.calculator.model
        metadata = self.calculator.metadata or {}
        atomic_numbers = tuple(
            int(z) for z in metadata.get('implemented_species', AIMNET2_ATOMIC_NUMBERS)
        )
        if not atomic_numbers:
            atomic_numbers = AIMNET2_ATOMIC_NUMBERS
        self.register_buffer(
            'atomic_numbers',
            torch.tensor(atomic_numbers, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'r_max',
            torch.tensor(float(self.calculator.cutoff), dtype=torch.float32),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True) -> AimnetNodeEncoder:
        super().train(mode)
        self.model.train(mode)
        if hasattr(self.calculator, '_train'):
            self.calculator._train = bool(mode)
        return self

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
        del compute_force, compute_virials, compute_stress
        if bool(training) != self.training:
            self.train(bool(training))
        if not compute_node_feats:
            raise ValueError('AIMNet encoder is only used for node features.')

        device = self._device()
        self.calculator.device = str(device)
        positions = graph['positions'].to(device=device)
        atomic_numbers = graph['atomic_numbers'].to(device=device, dtype=torch.long)
        batch = graph.get('batch')
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
            batch_size = 1
        else:
            batch = batch.to(device=device, dtype=torch.long)
            batch_size = (
                int(graph['ptr'].numel() - 1)
                if 'ptr' in graph
                else int(batch.max().item()) + 1
            )

        if molecular_charge is None:
            charge = positions.new_zeros(batch_size)
        else:
            charge = molecular_charge.to(device=device, dtype=positions.dtype)
            if charge.numel() != batch_size:
                raise ValueError(
                    'AIMNet molecular_charge must have one value per molecule; '
                    f'got {charge.numel()} values for batch_size={batch_size}.'
                )
            charge = charge.reshape(batch_size)

        data = {
            'coord': positions,
            'numbers': atomic_numbers,
            'charge': charge,
            'mol_idx': batch,
        }
        prepared = self.calculator.prepare_input(data)
        if isinstance(self.model, torch.jit.ScriptModule):
            with torch.jit.optimized_execution(False):  # type: ignore[attr-defined]
                out = self.model(prepared)
        else:
            out = self.model(prepared)
        if 'aim' not in out:
            raise RuntimeError(
                'AIMNet model did not return hidden atom features `aim`.'
            )
        return {'node_feats': out['aim'][: positions.shape[0]]}

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.atomic_numbers.device
