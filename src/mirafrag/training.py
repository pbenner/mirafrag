from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mirafrag.checkpoint import save_checkpoint
from mirafrag.data import move_batch_to_device
from mirafrag.losses import (
    LOSS_REGISTRY,
    sparse_binned_cosine_similarity,
    sparse_decoupled_oos_probability,
    sparse_fragment_only_binned_cosine_similarity,
    sparse_oos_probability,
    spectrum_loss,
)
from mirafrag.model import MiraFragModel
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
            )
        if training:
            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_interval == 'step':
                scheduler.step()
        with torch.no_grad():
            loss_value = float(loss.detach().cpu())
            if loss_name == 'decoupled_kl':
                cosine = sparse_fragment_only_binned_cosine_similarity(pred, batch)
                oos_probability = sparse_decoupled_oos_probability(pred)
            else:
                cosine = sparse_binned_cosine_similarity(pred, batch)
                oos_probability = sparse_oos_probability(pred)
            cosine_value = float(cosine.mean().cpu())
            oos_value = float(oos_probability.mean().cpu())
            total_examples += batch_size
            loss_sum += loss_value * batch_size
            cosine_sum += cosine_value * batch_size
            oos_sum += oos_value * batch_size
            if show_progress:
                progress.set_postfix(
                    {
                        f'{objective_name}_avg': (
                            f'{loss_sum / max(total_examples, 1):.4f}'
                        ),
                        'cosine_avg': (f'{cosine_sum / max(total_examples, 1):.4f}'),
                        'oos_avg': f'{oos_sum / max(total_examples, 1):.4f}',
                        'lr': f'{_current_lr(optimizer):.2e}'
                        if optimizer is not None
                        else 'n/a',
                    }
                )

    if processed_batches != total_batches:
        print(
            f'Warning: {desc or "epoch"} processed {processed_batches}/'
            f'{total_batches} batches.'
        )

    return {
        'loss': float(loss_sum / max(total_examples, 1)),
        'cosine': float(cosine_sum / max(total_examples, 1)),
        'oos_probability': float(oos_sum / max(total_examples, 1)),
    }


def train_model(
    model: MiraFragModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str | torch.device,
    output: str | Path,
    loss_name: str = 'cosine',
    train_config: dict[str, Any] | None = None,
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
    checkpoint_metric: str = 'val_loss',
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
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(
            model,
            lr=lr,
            weight_decay=weight_decay,
        ),
        lr=lr,
        weight_decay=weight_decay,
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
        swa_model = AveragedModel(model, device=device, use_buffers=True)
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
    train_config['checkpoint_metric'] = checkpoint_metric
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
        )
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
            f'lr={epoch_lr:.2e}'
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


def _swa_n_averaged(swa_model: AveragedModel) -> int:
    """
    Return the number of models already included in an SWA average.
    """
    return int(swa_model.n_averaged.detach().cpu().item())


def _prediction_probability_mode(loss_name: str) -> str:
    """
    Return the prediction probability semantics implied by a training loss.
    """
    return 'decoupled' if loss_name == 'decoupled_kl' else 'joint'


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
    else:
        history['train_loss'].append(float(train_stats['loss']))
        history['train_cosine'].append(float(train_stats['cosine']))
        history['train_oos_probability'].append(float(train_stats['oos_probability']))
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
