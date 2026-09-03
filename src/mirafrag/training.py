from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mirafrag.checkpoint import save_checkpoint
from mirafrag.data import move_batch_to_device
from mirafrag.losses import (
    LOSS_REGISTRY,
    _fragment_only_log_probs,
    exclude_precursor_prediction_candidates,
    sparse_binned_cosine_similarity,
    sparse_decoupled_oos_probability,
    sparse_fragment_only_binned_cosine_similarity,
    sparse_oos_probability,
    spectrum_loss,
)
from mirafrag.model import MiraFragModel, set_encoder_finetune_strategy
from mirafrag.optim import (
    _build_scheduler,
    _current_lr,
    _optimizer_param_groups,
    _print_optimizer_groups,
    _scheduler_total_steps,
)


def run_epoch(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
    loss_name: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = 'step',
    desc: str | None = None,
    show_progress: bool = True,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    kl_weight: float = 0.7,
    coverage_weight: float = 0.1,
    target_power: float = 1.0,
    entropy_weight: float = 0.0,
    head_delta_regularization: float = 0.0,
    head_delta_reference: dict[str, torch.Tensor] | None = None,
    encoder_delta_regularization: float = 0.0,
    encoder_delta_reference: dict[str, torch.Tensor] | None = None,
    distill_spectra: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    distill_loss_weight: float = 0.0,
    distill_min_overlap_mass: float = 0.05,
) -> dict[str, float]:
    """
    Run one training or evaluation epoch.

    The loop moves batches to device, computes the configured loss, optionally updates optimizer and step scheduler, tracks example-weighted averages, and reports tqdm progress statistics.
    """
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError('scheduler_interval must be one of: epoch, step.')
    training = optimizer is not None
    model.train(training)
    total_examples = 0
    total_batches = len(loader)
    processed_batches = 0
    loss_sum = 0.0
    cosine_sum = 0.0
    oos_sum = 0.0
    distill_loss_sum = 0.0
    distill_active = distill_spectra is not None and float(distill_loss_weight) > 0.0
    objective_name = _objective_display_name(loss_name)
    progress = tqdm(
        loader,
        desc=desc,
        total=total_batches,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for raw_batch in progress:
        processed_batches += 1
        batch = move_batch_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            pred = model(batch)
            batch_size = int(pred['batch_size'])
            sample_weight = batch.get('sample_weight') if training else None
            loss = spectrum_loss(
                pred,
                batch,
                loss=loss_name,
                mass_tolerance=mass_tolerance,
                relative_mass_tolerance=relative_mass_tolerance,
                mass_tolerance_min_mz=mass_tolerance_min_mz,
                kl_weight=kl_weight,
                coverage_weight=coverage_weight,
                target_power=target_power,
                entropy_weight=entropy_weight,
                reduction='none' if sample_weight is not None else 'mean',
            )
            if sample_weight is not None:
                weights = sample_weight.to(device=loss.device, dtype=loss.dtype)
                if loss.ndim != 1 or weights.shape != loss.shape:
                    raise ValueError(
                        'sample_weight requires one unreduced loss per spectrum.'
                    )
                loss = (loss * weights).mean()
            distill_loss = loss.new_tensor(float('nan'))
            if training and distill_active:
                distill_pred = exclude_precursor_prediction_candidates(pred, batch)
                distill_loss = _teacher_projection_kl(
                    distill_pred,
                    batch,
                    distill_spectra,
                    min_overlap_mass=distill_min_overlap_mass,
                )
                loss = loss + float(distill_loss_weight) * distill_loss
            if training and head_delta_regularization > 0.0:
                if head_delta_reference is None:
                    raise ValueError(
                        'head_delta_reference is required when '
                        'head_delta_regularization is positive.'
                    )
                loss = loss + float(head_delta_regularization) * _delta_penalty(
                    model,
                    head_delta_reference,
                )
            if training and encoder_delta_regularization > 0.0:
                if encoder_delta_reference is None:
                    raise ValueError(
                        'encoder_delta_reference is required when '
                        'encoder_delta_regularization is positive.'
                    )
                loss = loss + float(encoder_delta_regularization) * _delta_penalty(
                    model,
                    encoder_delta_reference,
                )
        if training:
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_interval == 'step':
                scheduler.step()
        with torch.no_grad():
            loss_value = float(loss.detach().cpu())
            scored_pred = exclude_precursor_prediction_candidates(pred, batch)
            if loss_name in {
                'decoupled_kl',
                'fiora_decoupled_kl',
                'responsibility_decoupled_kl',
                'fragment_cosine',
                'fragment_sqrt_cosine',
            }:
                cosine = sparse_fragment_only_binned_cosine_similarity(
                    scored_pred, batch
                )
                oos_probability = sparse_decoupled_oos_probability(pred)
            else:
                cosine = sparse_binned_cosine_similarity(scored_pred, batch)
                oos_probability = sparse_oos_probability(pred)
            cosine_value = float(cosine.mean().cpu())
            oos_value = float(oos_probability.mean().cpu())
            distill_value = (
                float(distill_loss.detach().cpu())
                if training and distill_active
                else float('nan')
            )
            total_examples += batch_size
            loss_sum += loss_value * batch_size
            cosine_sum += cosine_value * batch_size
            oos_sum += oos_value * batch_size
            if training and distill_active:
                distill_loss_sum += distill_value * batch_size
            if show_progress:
                postfix = {
                    f'{objective_name}_avg': (
                        f'{loss_sum / max(total_examples, 1):.4f}'
                    ),
                    'cosine_avg': (f'{cosine_sum / max(total_examples, 1):.4f}'),
                    'oos_avg': f'{oos_sum / max(total_examples, 1):.4f}',
                    'lr': f'{_current_lr(optimizer):.2e}'
                    if optimizer is not None
                    else 'n/a',
                }
                if training and distill_active:
                    postfix['distill_avg'] = (
                        f'{distill_loss_sum / max(total_examples, 1):.4f}'
                    )
                progress.set_postfix(postfix)

    if processed_batches != total_batches:
        print(
            f'Warning: {desc or "epoch"} processed {processed_batches}/'
            f'{total_batches} batches.'
        )

    stats = {
        'loss': float(loss_sum / max(total_examples, 1)),
        'cosine': float(cosine_sum / max(total_examples, 1)),
        'oos_probability': float(oos_sum / max(total_examples, 1)),
    }
    if training and distill_active:
        stats['distill_loss'] = float(distill_loss_sum / max(total_examples, 1))
    return stats


def _teacher_projection_kl(
    pred: dict[str, Any],
    batch: dict[str, Any],
    teacher_spectra: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    min_overlap_mass: float = 0.05,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Distill teacher spectra onto the model's currently supported fragment bins.

    Teacher peaks outside the generated candidate support are ignored and the
    overlapping teacher mass is renormalized. This trains candidate scoring while
    avoiding a support-coverage penalty that the scorer cannot fix.
    """
    fragment_log_probs = _fragment_only_log_probs(pred)
    pred_bins = pred['bins'].long()
    pred_batch = pred['batch'].long()
    identifiers = batch.get('identifier') or []
    losses: list[torch.Tensor] = []
    for batch_idx, identifier in enumerate(identifiers):
        teacher = teacher_spectra.get(str(identifier))
        if teacher is None:
            continue
        mask = pred_batch == batch_idx
        if not bool(mask.any()):
            continue
        local_bins = pred_bins[mask]
        local_log_probs = fragment_log_probs[mask]
        unique_bins, inverse = torch.unique(
            local_bins, sorted=True, return_inverse=True
        )
        bin_log_probs = local_log_probs.new_full(
            (int(unique_bins.numel()),), -float('inf')
        )
        for local_idx in range(int(unique_bins.numel())):
            bin_log_probs[local_idx] = torch.logsumexp(
                local_log_probs[inverse == local_idx], dim=0
            )
        teacher_bins = teacher[0].to(device=pred_bins.device).long()
        teacher_values = teacher[1].to(
            device=local_log_probs.device, dtype=local_log_probs.dtype
        )
        if teacher_bins.numel() == 0:
            continue
        positions = torch.searchsorted(unique_bins, teacher_bins)
        in_bounds = positions < unique_bins.numel()
        valid = torch.zeros_like(in_bounds, dtype=torch.bool)
        if bool(in_bounds.any()):
            valid[in_bounds] = (
                unique_bins[positions[in_bounds]] == teacher_bins[in_bounds]
            )
        if not bool(valid.any()):
            continue
        positions = positions[valid]
        target = teacher_values[valid]
        overlap_mass = target.sum()
        if float(overlap_mass.detach().cpu()) < float(min_overlap_mass):
            continue
        target = target / overlap_mass.clamp_min(eps)
        losses.append(F.kl_div(bin_log_probs[positions], target, reduction='sum'))
    if not losses:
        return pred['logits'].sum() * 0.0
    return torch.stack(losses).mean()


def run_retrieval_epoch(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = 'step',
    desc: str | None = None,
    show_progress: bool = True,
    loss_weight: float = 1.0,
) -> dict[str, float]:
    """Run one auxiliary retrieval-ranking epoch over expanded candidate rows."""
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError('scheduler_interval must be one of: epoch, step.')
    training = optimizer is not None
    model.train(training)
    total_groups = 0
    loss_sum = 0.0
    top1_sum = 0.0
    progress = tqdm(
        loader,
        desc=desc,
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for raw_batch in progress:
        if bool(raw_batch.pop('_retrieval_empty', False)):
            continue
        raw_batch.pop('_retrieval_errors', None)
        batch = move_batch_to_device(raw_batch, device)
        if '_retrieval_group' not in batch or '_retrieval_is_true' not in batch:
            raise ValueError('retrieval batches require group and truth labels.')
        if training:
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context:
            pred = model(batch)
            scored_pred = exclude_precursor_prediction_candidates(pred, batch)
            scores = sparse_fragment_only_binned_cosine_similarity(scored_pred, batch)
            retrieval_logit = pred.get('retrieval_logit')
            if retrieval_logit is not None:
                scores = scores + retrieval_logit.to(
                    device=scores.device,
                    dtype=scores.dtype,
                )
            group = batch['_retrieval_group'].long()
            is_true = batch['_retrieval_is_true'].bool()
            losses: list[torch.Tensor] = []
            correct = 0
            for group_id in torch.unique(group, sorted=True):
                mask = group == group_id
                true_positions = torch.nonzero(is_true[mask], as_tuple=False).flatten()
                if true_positions.numel() == 0:
                    continue
                local_scores = scores[mask]
                label = true_positions[:1].to(
                    device=local_scores.device, dtype=torch.long
                )
                losses.append(F.cross_entropy(local_scores.unsqueeze(0), label))
                correct += int(torch.argmax(local_scores).item() == int(label.item()))
            if losses:
                loss = torch.stack(losses).mean()
            else:
                loss = scores.sum() * 0.0
        weighted_loss = loss * float(loss_weight)
        if training:
            weighted_loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_interval == 'step':
                scheduler.step()
        groups = len(losses)
        if groups:
            total_groups += groups
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value * groups
            top1_sum += float(correct)
            if show_progress:
                progress.set_postfix(
                    {
                        'retrieval_loss_avg': f'{loss_sum / max(total_groups, 1):.4f}',
                        'retrieval_top1_avg': f'{top1_sum / max(total_groups, 1):.4f}',
                        'lr': f'{_current_lr(optimizer):.2e}'
                        if optimizer is not None
                        else 'n/a',
                    }
                )
    return {
        'loss': float(loss_sum / max(total_groups, 1)),
        'top1': float(top1_sum / max(total_groups, 1)),
        'groups': float(total_groups),
    }


def train_model(
    model: MiraFragModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    retrieval_loader: DataLoader | None = None,
    retrieval_loss_weight: float = 0.0,
    distill_spectra: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    distill_loss_weight: float = 0.0,
    distill_min_overlap_mass: float = 0.05,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str | torch.device,
    head_lr: float | None = None,
    encoder_lr: float | None = None,
    head_weight_decay: float = 0.0,
    encoder_weight_decay: float | None = None,
    encoder_layer_lr_decay: float = 1.0,
    output: str | Path,
    loss_name: str = 'cosine',
    train_config: dict[str, Any] | None = None,
    graph_config: Any | None = None,
    show_progress: bool = True,
    scheduler_name: str = 'exponential',
    scheduler_interval: str = 'epoch',
    min_lr_ratio: float = 0.1,
    exponential_gamma: float = 0.8,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
    evaluate_initial: bool = False,
    mass_tolerance: float = 0.01,
    relative_mass_tolerance: bool = False,
    mass_tolerance_min_mz: float = 200.0,
    kl_weight: float = 0.7,
    coverage_weight: float = 0.1,
    target_power: float = 1.0,
    entropy_weight: float = 0.0,
    head_delta_regularization: float = 0.0,
    encoder_delta_regularization: float = 0.0,
    checkpoint_metric: str = 'val_loss',
    verbose_epoch_config: bool = False,
    swa: bool = False,
    swa_start_epoch: int | None = None,
    swa_lr: float | None = None,
    swa_anneal_epochs: int = 1,
) -> dict[str, list[float]]:
    """
    Train a MiraFrag model and save the best checkpoint by the selected metric.

    The routine materializes lazy head layers, builds optimizer and scheduler, optionally evaluates the initial checkpoint state, runs train/validation epochs, saves improving checkpoints, and writes a history JSON file.
    """
    valid_checkpoint_metrics = {
        'val_loss',
        'train_loss',
        'val_cosine',
        'train_cosine',
    }
    if checkpoint_metric not in valid_checkpoint_metrics:
        raise ValueError(
            f'checkpoint_metric must be one of: {sorted(valid_checkpoint_metrics)}.'
        )
    model.to(device)
    _materialize_lazy_modules(model, train_loader, device=device)
    set_encoder_finetune_strategy(
        model,
        getattr(model.config, 'encoder_finetune_strategy', 'head'),
    )
    if head_delta_regularization < 0.0:
        raise ValueError('head_delta_regularization must be non-negative.')
    if encoder_delta_regularization < 0.0:
        raise ValueError('encoder_delta_regularization must be non-negative.')
    if encoder_layer_lr_decay <= 0.0 or encoder_layer_lr_decay > 1.0:
        raise ValueError('encoder_layer_lr_decay must be in the interval (0, 1].')
    if distill_loss_weight < 0.0:
        raise ValueError('distill_loss_weight must be non-negative.')
    if distill_min_overlap_mass < 0.0:
        raise ValueError('distill_min_overlap_mass must be non-negative.')
    if distill_loss_weight > 0.0 and not distill_spectra:
        raise ValueError('distill_spectra is required when distill_loss_weight > 0.')
    head_delta_reference = (
        _head_delta_reference(model) if head_delta_regularization > 0.0 else None
    )
    encoder_delta_reference = (
        _encoder_delta_reference(model) if encoder_delta_regularization > 0.0 else None
    )
    resolved_head_lr = float(lr if head_lr is None else head_lr)
    resolved_encoder_lr = float(lr if encoder_lr is None else encoder_lr)
    resolved_encoder_weight_decay = float(
        weight_decay if encoder_weight_decay is None else encoder_weight_decay
    )
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(
            model,
            lr=lr,
            weight_decay=weight_decay,
            head_lr=resolved_head_lr,
            encoder_lr=resolved_encoder_lr,
            head_weight_decay=head_weight_decay,
            encoder_weight_decay=resolved_encoder_weight_decay,
            encoder_layer_lr_decay=encoder_layer_lr_decay,
        ),
        lr=lr,
        weight_decay=0.0,
    )
    _print_optimizer_groups(optimizer)
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        total_steps=_scheduler_total_steps(
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            scheduler_interval=scheduler_interval,
        ),
        min_lr_ratio=min_lr_ratio,
        exponential_gamma=exponential_gamma,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
    )
    swa_model: AveragedModel | None = None
    swa_scheduler: SWALR | None = None
    resolved_swa_start_epoch: int | None = None
    if swa:
        resolved_swa_start_epoch = (
            int(swa_start_epoch) if swa_start_epoch is not None else max(1, epochs // 2)
        )
        if resolved_swa_start_epoch < 1 or resolved_swa_start_epoch > epochs:
            raise ValueError('swa_start_epoch must be between 1 and epochs.')
        # Only average parameters: encoder metadata buffers include integer tensors
        # such as atomic numbers, which PyTorch SWA cannot average.
        swa_model = AveragedModel(model, device=device, use_buffers=False)
        if swa_lr is not None:
            if swa_lr <= 0:
                raise ValueError('swa_lr must be positive when set.')
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=float(swa_lr),
                anneal_epochs=max(1, int(swa_anneal_epochs)),
            )
        print(
            'SWA enabled: '
            f'start_epoch={resolved_swa_start_epoch} '
            f'swa_lr={swa_lr if swa_lr is not None else "scheduler"} '
            f'anneal_epochs={max(1, int(swa_anneal_epochs))}'
        )
    train_config = dict(train_config or {})
    train_config['loss'] = loss_name
    train_config['head_lr'] = resolved_head_lr
    train_config['encoder_lr'] = resolved_encoder_lr
    train_config['head_weight_decay'] = float(head_weight_decay)
    train_config['head_delta_regularization'] = float(head_delta_regularization)
    train_config['encoder_delta_regularization'] = float(encoder_delta_regularization)
    train_config['encoder_weight_decay'] = resolved_encoder_weight_decay
    train_config['encoder_layer_lr_decay'] = float(encoder_layer_lr_decay)
    train_config['checkpoint_metric'] = checkpoint_metric
    train_config['retrieval_loss_weight'] = float(retrieval_loss_weight)
    train_config['distill_loss_weight'] = float(distill_loss_weight)
    train_config['distill_min_overlap_mass'] = float(distill_min_overlap_mass)
    train_config['prediction_probability_mode'] = _prediction_probability_mode(
        loss_name
    )
    train_config['swa'] = bool(swa)
    if swa:
        train_config['swa_start_epoch'] = resolved_swa_start_epoch
        train_config['swa_lr'] = swa_lr
        train_config['swa_anneal_epochs'] = max(1, int(swa_anneal_epochs))
    history: dict[str, list[float]] = {
        'epoch': [],
        'train_loss': [],
        'train_cosine': [],
        'train_oos_probability': [],
        'train_retrieval_loss': [],
        'train_retrieval_top1': [],
        'train_distill_loss': [],
        'val_loss': [],
        'val_cosine': [],
        'val_oos_probability': [],
        'swa_val_loss': [],
        'swa_val_cosine': [],
        'swa_val_oos_probability': [],
        'swa_n_averaged': [],
    }
    best_checkpoint_score = _initial_checkpoint_score(checkpoint_metric)
    checkpoint_metric_name = checkpoint_metric
    objective_name = _objective_display_name(loss_name)

    if evaluate_initial and val_loader is not None:
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            loss_name=loss_name,
            desc=f'val epoch 0/{epochs}',
            show_progress=show_progress,
            mass_tolerance=mass_tolerance,
            relative_mass_tolerance=relative_mass_tolerance,
            mass_tolerance_min_mz=mass_tolerance_min_mz,
            kl_weight=kl_weight,
            coverage_weight=coverage_weight,
            target_power=target_power,
            entropy_weight=entropy_weight,
        )
        _append_history(
            history,
            epoch=0,
            train_stats=None,
            val_stats=val_stats,
        )
        _append_swa_history(history, swa_stats=None, n_averaged=0)
        print(
            'epoch=0 '
            f'val_{objective_name}={val_stats["loss"]:.5f} '
            f'val_cosine={val_stats["cosine"]:.5f} '
            f'val_oos={val_stats["oos_probability"]:.5f} '
            f'lr={_current_lr(optimizer):.2e}'
        )
        if _checkpoint_metric_source(checkpoint_metric) == 'val':
            best_checkpoint_score = _checkpoint_metric_value(
                val_stats,
                checkpoint_metric,
            )
            save_checkpoint(
                output,
                model,
                train_config=_checkpoint_train_config(train_config),
                graph_config=graph_config,
            )
            print(
                f'saved checkpoint to {output} '
                f'{checkpoint_metric_name}={best_checkpoint_score:.5f}'
            )

    for epoch in range(1, epochs + 1):
        epoch_lr = _current_lr(optimizer)
        in_swa_phase = (
            swa_model is not None
            and resolved_swa_start_epoch is not None
            and epoch >= resolved_swa_start_epoch
        )
        batch_scheduler = scheduler if scheduler_name != 'plateau' else None
        if in_swa_phase and swa_scheduler is not None:
            batch_scheduler = None
        if verbose_epoch_config:
            _print_epoch_config(
                epoch=epoch,
                epochs=epochs,
                lr=epoch_lr,
                weight_decay=_optimizer_weight_decay_summary(optimizer),
                dropout=_model_dropout(model),
                swa=bool(swa),
                in_swa_phase=in_swa_phase,
                swa_start_epoch=resolved_swa_start_epoch,
                swa_lr=swa_lr,
                swa_anneal_epochs=max(1, int(swa_anneal_epochs)),
                swa_n_averaged=_swa_n_averaged(swa_model)
                if swa_model is not None
                else 0,
            )
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=batch_scheduler,
            scheduler_interval=scheduler_interval,
            device=device,
            loss_name=loss_name,
            desc=f'train epoch {epoch}/{epochs}',
            show_progress=show_progress,
            mass_tolerance=mass_tolerance,
            relative_mass_tolerance=relative_mass_tolerance,
            mass_tolerance_min_mz=mass_tolerance_min_mz,
            kl_weight=kl_weight,
            coverage_weight=coverage_weight,
            target_power=target_power,
            entropy_weight=entropy_weight,
            head_delta_regularization=head_delta_regularization,
            head_delta_reference=head_delta_reference,
            encoder_delta_regularization=encoder_delta_regularization,
            encoder_delta_reference=encoder_delta_reference,
            distill_spectra=distill_spectra,
            distill_loss_weight=distill_loss_weight,
            distill_min_overlap_mass=distill_min_overlap_mass,
        )
        retrieval_stats = None
        if retrieval_loader is not None and retrieval_loss_weight > 0.0:
            retrieval_stats = run_retrieval_epoch(
                model,
                retrieval_loader,
                optimizer=optimizer,
                scheduler=batch_scheduler,
                scheduler_interval=scheduler_interval,
                device=device,
                desc=f'retrieval epoch {epoch}/{epochs}',
                show_progress=show_progress,
                loss_weight=float(retrieval_loss_weight),
            )
            train_stats = dict(train_stats)
            train_stats['loss'] = train_stats['loss'] + float(
                retrieval_loss_weight
            ) * float(retrieval_stats['loss'])
            train_stats['retrieval_loss'] = float(retrieval_stats['loss'])
            train_stats['retrieval_top1'] = float(retrieval_stats['top1'])

        val_stats = (
            run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_name=loss_name,
                desc=f'val epoch {epoch}/{epochs}',
                show_progress=show_progress,
                mass_tolerance=mass_tolerance,
                relative_mass_tolerance=relative_mass_tolerance,
                mass_tolerance_min_mz=mass_tolerance_min_mz,
                kl_weight=kl_weight,
                coverage_weight=coverage_weight,
                target_power=target_power,
                entropy_weight=entropy_weight,
            )
            if val_loader is not None
            else train_stats
        )

        _append_history(
            history,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
        )

        print(
            f'epoch={epoch} '
            f'train_{objective_name}={train_stats["loss"]:.5f} '
            f'train_cosine={train_stats["cosine"]:.5f} '
            f'train_oos={train_stats["oos_probability"]:.5f} '
            f'val_{objective_name}={val_stats["loss"]:.5f} '
            f'val_cosine={val_stats["cosine"]:.5f} '
            f'val_oos={val_stats["oos_probability"]:.5f} '
            + (
                f'train_retrieval_loss={train_stats["retrieval_loss"]:.5f} '
                f'train_retrieval_top1={train_stats["retrieval_top1"]:.5f} '
                if 'retrieval_loss' in train_stats
                else ''
            )
            + (
                f'train_distill_loss={train_stats["distill_loss"]:.5f} '
                if 'distill_loss' in train_stats
                else ''
            )
            + f'lr={epoch_lr:.2e}'
        )

        checkpoint_stats = _checkpoint_stats(
            train_stats=train_stats,
            val_stats=val_stats,
            checkpoint_metric=checkpoint_metric,
        )
        checkpoint_score = _checkpoint_metric_value(
            checkpoint_stats,
            checkpoint_metric,
        )
        if _checkpoint_improved(
            checkpoint_score,
            best_checkpoint_score,
            checkpoint_metric,
        ):
            best_checkpoint_score = checkpoint_score
            save_checkpoint(
                output,
                model,
                train_config=_checkpoint_train_config(train_config),
                graph_config=graph_config,
            )
            print(
                f'saved checkpoint to {output} '
                f'{checkpoint_metric_name}={best_checkpoint_score:.5f}'
            )

        swa_stats: dict[str, float] | None = None
        n_averaged = 0
        if in_swa_phase and swa_model is not None:
            swa_model.update_parameters(model)
            n_averaged = _swa_n_averaged(swa_model)
            if val_loader is not None:
                swa_stats = run_epoch(
                    swa_model,
                    val_loader,
                    optimizer=None,
                    device=device,
                    loss_name=loss_name,
                    desc=f'swa val epoch {epoch}/{epochs}',
                    show_progress=show_progress,
                    mass_tolerance=mass_tolerance,
                    relative_mass_tolerance=relative_mass_tolerance,
                    mass_tolerance_min_mz=mass_tolerance_min_mz,
                    kl_weight=kl_weight,
                    coverage_weight=coverage_weight,
                    target_power=target_power,
                    entropy_weight=entropy_weight,
                )
                print(
                    f'swa_epoch={epoch} '
                    f'val_{objective_name}={swa_stats["loss"]:.5f} '
                    f'val_cosine={swa_stats["cosine"]:.5f} '
                    f'val_oos={swa_stats["oos_probability"]:.5f} '
                    f'n_averaged={n_averaged}'
                )
                if _checkpoint_metric_source(checkpoint_metric) == 'val':
                    swa_checkpoint_score = _checkpoint_metric_value(
                        swa_stats,
                        checkpoint_metric,
                    )
                    if _checkpoint_improved(
                        swa_checkpoint_score,
                        best_checkpoint_score,
                        checkpoint_metric,
                    ):
                        best_checkpoint_score = swa_checkpoint_score
                        save_checkpoint(
                            output,
                            swa_model.module,
                            train_config=_checkpoint_train_config(
                                train_config,
                                swa_checkpoint=True,
                                swa_n_averaged=n_averaged,
                            ),
                            graph_config=graph_config,
                        )
                        print(
                            f'saved SWA checkpoint to {output} '
                            f'{checkpoint_metric_name}={best_checkpoint_score:.5f}'
                        )
        _append_swa_history(
            history,
            swa_stats=swa_stats,
            n_averaged=n_averaged,
        )

        if in_swa_phase and swa_scheduler is not None:
            swa_scheduler.step()
        elif scheduler_name == 'plateau' and scheduler is not None:
            scheduler.step(val_stats['loss'])
        elif scheduler is not None and scheduler_interval == 'epoch':
            scheduler.step()

    history_path = Path(str(output).replace('.pt', '.history.json'))
    with open(history_path, 'w') as fp:
        json.dump(history, fp, indent=2)
    return history


def _checkpoint_metric_source(checkpoint_metric: str) -> str:
    """
    Return whether a checkpoint metric reads training or validation statistics.
    """
    return checkpoint_metric.split('_', 1)[0]


def _checkpoint_metric_key(checkpoint_metric: str) -> str:
    """
    Return the statistic key used by a checkpoint metric.
    """
    return checkpoint_metric.split('_', 1)[1]


def _initial_checkpoint_score(checkpoint_metric: str) -> float:
    """
    Return the initial best value for minimizing losses or maximizing cosine.
    """
    return (
        -float('inf')
        if _checkpoint_metric_key(checkpoint_metric) == 'cosine'
        else float('inf')
    )


def _checkpoint_improved(
    value: float,
    best_value: float,
    checkpoint_metric: str,
) -> bool:
    """
    Return whether a checkpoint metric value improves on the previous best.
    """
    if _checkpoint_metric_key(checkpoint_metric) == 'cosine':
        return value >= best_value
    return value <= best_value


def _checkpoint_metric_value(
    stats: dict[str, float],
    checkpoint_metric: str,
) -> float:
    """
    Extract the loss or cosine value addressed by a checkpoint metric.
    """
    return float(stats[_checkpoint_metric_key(checkpoint_metric)])


def _checkpoint_stats(
    *,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    checkpoint_metric: str,
) -> dict[str, float]:
    """
    Select train or validation statistics for checkpoint comparison.
    """
    if _checkpoint_metric_source(checkpoint_metric) == 'train':
        return train_stats
    return val_stats


def _checkpoint_train_config(
    train_config: dict[str, Any],
    *,
    swa_checkpoint: bool = False,
    swa_n_averaged: int | None = None,
) -> dict[str, Any]:
    """
    Return checkpoint metadata with explicit live/SWA checkpoint identity.
    """
    config = dict(train_config)
    config['swa_checkpoint'] = bool(swa_checkpoint)
    if swa_n_averaged is not None:
        config['swa_n_averaged'] = int(swa_n_averaged)
    return config


def _head_delta_reference(model: MiraFragModel) -> dict[str, torch.Tensor]:
    """
    Snapshot trainable spectrum-head weight matrices for checkpoint-centered regularization.
    """
    refs = _module_delta_reference(model.head, prefix='head')
    if not refs:
        raise ValueError(
            'head_delta_regularization requires trainable spectrum-head weight matrices.'
        )
    return refs


def _encoder_delta_reference(model: MiraFragModel) -> dict[str, torch.Tensor]:
    """
    Snapshot trainable encoder weight matrices for checkpoint-centered regularization.
    """
    refs = _module_delta_reference(model.encoder, prefix='encoder')
    if not refs:
        raise ValueError(
            'encoder_delta_regularization requires trainable encoder weight matrices.'
        )
    return refs


def _module_delta_reference(
    module: torch.nn.Module,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    Snapshot trainable weight matrices from a module using full model parameter names.
    """
    refs: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters(prefix=prefix):
        if param.requires_grad and param.ndim >= 2:
            refs[name] = param.detach().clone()
    return refs


def _delta_penalty(
    model: MiraFragModel,
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Return mean squared deviation from checkpoint weights.
    """
    named_params = dict(model.named_parameters())
    total_penalty: torch.Tensor | None = None
    total_elements = 0
    for name, ref in reference.items():
        param = named_params.get(name)
        if param is None:
            raise ValueError(f'Delta reference parameter {name!r} is missing.')
        delta_sum = (
            (param - ref.to(device=param.device, dtype=param.dtype)).square().sum()
        )
        total_penalty = (
            delta_sum if total_penalty is None else total_penalty + delta_sum
        )
        total_elements += int(param.numel())
    if total_penalty is None or total_elements == 0:
        raise ValueError('Delta reference is empty.')
    return total_penalty / float(total_elements)


def _print_epoch_config(
    *,
    epoch: int,
    epochs: int,
    lr: float,
    weight_decay: str,
    dropout: float,
    swa: bool,
    in_swa_phase: bool,
    swa_start_epoch: int | None,
    swa_lr: float | None,
    swa_anneal_epochs: int,
    swa_n_averaged: int,
) -> None:
    """
    Print per-epoch hyperparameter context for validation tuning runs.
    """
    print(
        f'epoch_config epoch={epoch}/{epochs} '
        f'lr={lr:.2e} '
        f'weight_decay={weight_decay} '
        f'dropout={dropout:g} '
        f'swa={swa} '
        f'swa_active={in_swa_phase} '
        f'swa_start={swa_start_epoch if swa_start_epoch is not None else "n/a"} '
        f'swa_lr={swa_lr if swa_lr is not None else "scheduler"} '
        f'swa_anneal_epochs={swa_anneal_epochs} '
        f'swa_n_averaged={swa_n_averaged}'
    )


def _optimizer_weight_decay_summary(optimizer: torch.optim.Optimizer) -> str:
    """
    Return a compact per-parameter-group weight-decay summary.
    """
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get('name', f'group{index}'))
        weight_decay = float(group.get('weight_decay', 0.0))
        parts.append(f'{name}={weight_decay:.2e}')
    return ','.join(parts)


def _model_dropout(model: torch.nn.Module) -> float:
    """
    Return the configured MiraFrag head dropout value for epoch logging.
    """
    config = getattr(model, 'config', None)
    if config is not None and hasattr(config, 'dropout'):
        return float(config.dropout)
    dropouts = [
        module.p for module in model.modules() if isinstance(module, torch.nn.Dropout)
    ]
    return float(dropouts[0]) if dropouts else 0.0


def _swa_n_averaged(swa_model: AveragedModel) -> int:
    """
    Return the number of models already included in an SWA average.
    """
    return int(swa_model.n_averaged.detach().cpu().item())


def _prediction_probability_mode(loss_name: str) -> str:
    """
    Return the prediction probability semantics implied by a training loss.
    """
    return (
        'decoupled'
        if loss_name
        in {
            'decoupled_kl',
            'fiora_decoupled_kl',
            'responsibility_decoupled_kl',
            'fragment_cosine',
            'fragment_sqrt_cosine',
        }
        else 'joint'
    )


def _append_history(
    history: dict[str, list[float]],
    *,
    epoch: int,
    train_stats: dict[str, float] | None,
    val_stats: dict[str, float],
) -> None:
    """
    Append one epoch's train and validation statistics to the history dictionary.
    """
    history['epoch'].append(epoch)
    if train_stats is None:
        history['train_loss'].append(float('nan'))
        history['train_cosine'].append(float('nan'))
        history['train_oos_probability'].append(float('nan'))
        history.setdefault('train_retrieval_loss', []).append(float('nan'))
        history.setdefault('train_retrieval_top1', []).append(float('nan'))
        history.setdefault('train_distill_loss', []).append(float('nan'))
    else:
        history['train_loss'].append(float(train_stats['loss']))
        history['train_cosine'].append(float(train_stats['cosine']))
        history['train_oos_probability'].append(float(train_stats['oos_probability']))
        history.setdefault('train_retrieval_loss', []).append(
            float(train_stats.get('retrieval_loss', float('nan')))
        )
        history.setdefault('train_retrieval_top1', []).append(
            float(train_stats.get('retrieval_top1', float('nan')))
        )
        history.setdefault('train_distill_loss', []).append(
            float(train_stats.get('distill_loss', float('nan')))
        )
    history['val_loss'].append(float(val_stats['loss']))
    history['val_cosine'].append(float(val_stats['cosine']))
    history['val_oos_probability'].append(float(val_stats['oos_probability']))


def _append_swa_history(
    history: dict[str, list[float]],
    *,
    swa_stats: dict[str, float] | None,
    n_averaged: int,
) -> None:
    """
    Append SWA validation statistics or NaNs when no SWA model was evaluated.
    """
    if swa_stats is None:
        history['swa_val_loss'].append(float('nan'))
        history['swa_val_cosine'].append(float('nan'))
        history['swa_val_oos_probability'].append(float('nan'))
    else:
        history['swa_val_loss'].append(float(swa_stats['loss']))
        history['swa_val_cosine'].append(float(swa_stats['cosine']))
        history['swa_val_oos_probability'].append(float(swa_stats['oos_probability']))
    history['swa_n_averaged'].append(float(n_averaged))


def _objective_display_name(loss_name: str) -> str:
    """
    Return the progress-display name for a registered loss.
    """
    loss_spec = LOSS_REGISTRY.get(loss_name)
    if loss_spec is None:
        return 'loss'
    return loss_spec.display_name


def _materialize_lazy_modules(
    model: MiraFragModel,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> None:
    """
    Run one dummy forward pass so lazy modules create their parameters before optimizer setup.
    """
    try:
        first_item = loader.dataset[0]
    except IndexError as exc:
        raise ValueError('Cannot train MiraFrag with an empty DataLoader.') from exc
    collate_fn = loader.collate_fn or default_collate
    raw_batch = collate_fn([first_item])
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(move_batch_to_device(raw_batch, device))
    model.train(was_training)
