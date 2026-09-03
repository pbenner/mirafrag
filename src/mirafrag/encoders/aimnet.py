from __future__ import annotations

import torch
from torch import nn

AIMNET2_ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)
AIMNET2_R_MAX = 5.0


def load_aimnet_encoder(
    *,
    model: str | None = 'aimnet2',
    model_path: str | None = None,
    device: str | torch.device = 'cpu',
) -> nn.Module:
    """
    Create an AIMNet node-feature encoder adapter.

    A local path takes precedence over the registry model name. The adapter exposes AIMNet hidden atom features through the same interface used by the MACE encoder.
    """
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
        """
        Load AIMNet2 and expose metadata needed by MiraFrag.

        The constructor initializes the AIMNet calculator in train-capable mode, registers supported atomic numbers and cutoff radius, and keeps the underlying AIMNet model available as a child module.
        """
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
        self.export_multipass_features = False
        self.node_feature_dim = self._infer_node_feature_dim()
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

    def _infer_node_feature_dim(self) -> int | None:
        """
        Infer AIMNet final atom-feature width from the last AIM MLP when possible.
        """
        mlps = getattr(self.model, 'mlps', None)
        if not mlps:
            return None
        try:
            last_mlp = mlps[-1]
        except Exception:
            return None
        for module in reversed(list(last_mlp.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
        return None

    def train(self, mode: bool = True) -> AimnetNodeEncoder:
        """
        Set training/evaluation mode on both adapter and AIMNet calculator.

        AIMNet's calculator keeps its own training flag, so this override keeps it synchronized with PyTorch module mode.
        """
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
        """
        Return AIMNet hidden atom features for a batched molecular graph.

        The graph must contain positions, atomic numbers, and optional batch/ptr tensors. Charge-aware inputs pass one molecular charge per graph, and the output is ``{'node_feats': tensor}`` aligned to the original atoms.
        """
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
        if self.export_multipass_features:
            out = self._forward_with_multipass_features(prepared)
        elif isinstance(self.model, torch.jit.ScriptModule):
            with torch.jit.optimized_execution(False):  # type: ignore[attr-defined]
                out = self.model(prepared)
        else:
            out = self.model(prepared)
        if 'aim' not in out:
            raise RuntimeError(
                'AIMNet model did not return hidden atom features `aim`.'
            )
        num_atoms = positions.shape[0]
        result = {'node_feats': out['aim'][:num_atoms]}
        charge_features = self._charge_features(out, num_atoms=num_atoms)
        if charge_features is not None:
            result['aimnet_charge_features'] = charge_features
        multipass_features = out.get('aimnet_multipass_features')
        if multipass_features is not None:
            result['aimnet_multipass_features'] = multipass_features[:num_atoms]
        return result

    def _forward_with_multipass_features(
        self, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Run AIMNet2 while retaining intermediate atom states before final pooling.
        """
        model = self.model
        required = (
            'prepare_input',
            'afv',
            'aev',
            'mlps',
            '_prepare_in_a',
            '_prepare_in_q',
            '_update_q',
        )
        if isinstance(model, torch.jit.ScriptModule) or not all(
            hasattr(model, name) for name in required
        ):
            raise RuntimeError(
                'AIMNet multipass features require an eager AIMNet2 model; '
                'scripted or unsupported AIMNet variants do not expose intermediate states.'
            )

        data = model.prepare_input(data)
        a = model.afv(data['numbers'])
        if getattr(model, 'd2features', False):
            a = a.unflatten(-1, (model.nfeature, model.nshifts_s))
        data['a'] = a

        if getattr(model, 'num_charge_channels', 1) == 2:
            data = model._preprocess_spin_polarized_charge(data)
        else:
            data['charge'] = data['charge'].unsqueeze(-1)

        data = model.aev(data)
        states: list[torch.Tensor] = []
        num_passes = len(model.mlps)
        for pass_idx, mlp in enumerate(model.mlps):
            if pass_idx == 0:
                mlp_input = model._prepare_in_a(data)
            else:
                mlp_input = torch.cat(
                    [model._prepare_in_a(data), model._prepare_in_q(data)], dim=-1
                )
            mlp_output = mlp(mlp_input)
            if data['_input_padded'].item():
                from aimnet import nbops

                mlp_output = nbops.mask_i_(mlp_output, data, mask_value=0.0)

            if pass_idx == 0:
                data = model._update_q(data, mlp_output, delta_q=False)
                states.append(data['a'].flatten(-2, -1))
            elif pass_idx < num_passes - 1:
                data = model._update_q(data, mlp_output, delta_q=True)
                states.append(data['a'].flatten(-2, -1))
            else:
                data['aim'] = mlp_output
                states.append(mlp_output)

        if getattr(model, 'num_charge_channels', 1) == 2:
            data = model._postprocess_spin_polarized_charge(data)
        else:
            data['charges'] = data['charges'].squeeze(-1)
            data['charge'] = data['charge'].squeeze(-1)

        for module in model.outputs.children():
            data = module(data)

        data['aimnet_multipass_features'] = torch.cat(states, dim=-1)
        return data

    @staticmethod
    def _charge_features(
        out: dict[str, torch.Tensor],
        *,
        num_atoms: int,
    ) -> torch.Tensor | None:
        """
        Return fixed-width AIMNet charge channels aligned to real input atoms.
        """
        charges = out.get('charges')
        if charges is None:
            return None
        charges = charges[:num_atoms].reshape(num_atoms, 1)
        pre_charges = out.get('charges_pre')
        if pre_charges is None:
            pre = torch.zeros_like(charges)
        else:
            pre = pre_charges[:num_atoms].reshape(num_atoms, 1)
        return torch.cat(
            [
                charges,
                charges.abs(),
                charges.clamp_min(0.0),
                (-charges).clamp_min(0.0),
                pre,
                charges - pre,
            ],
            dim=-1,
        )

    def _device(self) -> torch.device:
        """
        Return the current device of the wrapped AIMNet model.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.atomic_numbers.device
