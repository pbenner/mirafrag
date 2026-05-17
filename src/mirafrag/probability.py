from __future__ import annotations

import torch


def fragment_oos_log_probs(
    pred: dict[str, torch.Tensor | int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize fragment logits and OOS logits into log probabilities per spectrum.

    Each spectrum has its own softmax over its fragment candidates plus one optional OOS bucket. The function returns candidate log probabilities aligned to ``pred['logits']`` and one OOS log probability per spectrum.
    """
    logits = pred['logits']
    if not isinstance(logits, torch.Tensor):
        raise TypeError("pred['logits'] must be a tensor.")
    batch = pred['batch']
    if not isinstance(batch, torch.Tensor):
        raise TypeError("pred['batch'] must be a tensor.")
    batch = batch.long()
    batch_size = int(pred['batch_size'])
    raw_oos_logits = pred.get('oos_logits')
    if raw_oos_logits is not None:
        if not isinstance(raw_oos_logits, torch.Tensor):
            raise TypeError("pred['oos_logits'] must be a tensor.")
        oos_logits = raw_oos_logits.to(device=logits.device, dtype=logits.dtype)

    frag_log_probs = logits.new_empty(logits.shape)
    oos_log_probs = logits.new_empty(batch_size)
    for batch_idx in range(batch_size):
        mask = batch == batch_idx
        if raw_oos_logits is None:
            if bool(mask.any()):
                frag_log_probs[mask] = torch.log_softmax(logits[mask], dim=0)
                oos_log_probs[batch_idx] = -torch.inf
            else:
                oos_log_probs[batch_idx] = 0.0
            continue
        combined = torch.cat([logits[mask], oos_logits[batch_idx].reshape(1)])
        combined_log_probs = torch.log_softmax(combined, dim=0)
        if bool(mask.any()):
            frag_log_probs[mask] = combined_log_probs[:-1]
        oos_log_probs[batch_idx] = combined_log_probs[-1]
    return frag_log_probs, oos_log_probs
