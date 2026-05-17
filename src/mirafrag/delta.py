from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import torch
from torch import nn

try:  # pragma: no cover - depends on torch version
    from torch.func import functional_call as _functional_call
except ModuleNotFoundError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

_SANITIZE_TOKEN = '__DOT__'


def _sanitize(name: str) -> str:
    """
    Convert a parameter name into a ParameterDict-safe key.
    """
    return name.replace('.', _SANITIZE_TOKEN)


class TorchDeltaFineTuneWrapper(nn.Module):
    """Additive delta fine-tuning wrapper for a Torch module.

    This mirrors Equitrain's delta strategy: base parameters are frozen, each
    base parameter gets a same-shaped trainable delta tensor, and forward uses
    ``base_parameter + delta`` through ``torch.func.functional_call``.
    """

    def __init__(self, base_module: nn.Module) -> None:
        """
        Wrap a module with zero-initialized trainable delta parameters.

        All base parameters are frozen. For each named base parameter, a same-shaped delta parameter is registered and used during functional calls.
        """
        super().__init__()
        self.base_module = base_module

        for param in self.base_module.parameters():
            param.requires_grad_(False)

        self._delta_params = nn.ParameterDict()
        self._delta_entries: list[tuple[str, nn.Parameter, nn.Parameter]] = []
        for name, param in self.base_module.named_parameters():
            delta = nn.Parameter(torch.zeros_like(param))
            self._delta_params[_sanitize(name)] = delta
            self._delta_entries.append((name, param, delta))

    def delta_parameters(self) -> Iterator[nn.Parameter]:
        """
        Iterate over all trainable delta tensors.
        """
        return iter(self._delta_params.values())

    def named_delta_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """
        Iterate over delta tensors using the original base parameter names.
        """
        for name, _base_param, delta in self._delta_entries:
            yield name, delta

    def merged_parameter_dict(self) -> OrderedDict[str, torch.Tensor]:
        """
        Return base parameters plus deltas along with base buffers.

        The dictionary is suitable for ``torch.func.functional_call`` and represents the effective parameters used for one forward pass.
        """
        params = OrderedDict(
            (name, base_param + delta)
            for name, base_param, delta in self._delta_entries
        )
        for name, buffer in self.base_module.named_buffers():
            params[name] = buffer
        return params

    def forward(self, *args, **kwargs):
        """
        Run the wrapped module with base-plus-delta parameters.
        """
        return _functional_call(
            self.base_module,
            self.merged_parameter_dict(),
            args,
            kwargs,
            strict=False,
        )

    def merge_deltas_(self) -> nn.Module:
        """
        Add delta tensors into the base module in place and zero the deltas.

        The merged base module is returned, which is used when switching from delta fine-tuning to full or head-only strategies.
        """
        with torch.no_grad():
            for _name, base_param, delta in self._delta_entries:
                base_param.add_(delta)
                delta.zero_()
        return self.base_module

    def __getattr__(self, item):
        """
        Delegate unknown attributes to the wrapped base module.

        This keeps encoder attributes such as ``atomic_numbers`` and ``r_max`` visible through the wrapper.
        """
        if item in {'base_module', '_delta_params', '_delta_entries'}:
            return super().__getattr__(item)
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.base_module, item)
