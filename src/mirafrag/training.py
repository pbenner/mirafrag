from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from mirafrag.checkpoint import save_checkpoint
from mirafrag.data import move_batch_to_device
from mirafrag.losses import (
    LOSS_REGISTRY,
    sparse_binned_cosine_similarity,
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
            cosine_value = float(
                sparse_binned_cosine_similarity(pred, batch).mean().cpu()
            )
            oos_value = float(sparse_oos_probability(pred).mean().cpu())
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
) -> dict[str, list[float]]:
    """
    Train a MiraFrag model and save the best checkpoint by validation loss.

    The routine materializes lazy head layers, builds optimizer and scheduler, optionally evaluates the initial checkpoint state, runs train/validation epochs, saves improving checkpoints, and writes a history JSON file.
    """
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
    history: dict[str, list[float]] = {
        'epoch': [],
        'train_loss': [],
        'train_cosine': [],
        'train_oos_probability': [],
        'val_loss': [],
        'val_cosine': [],
        'val_oos_probability': [],
    }
    best_val = float('inf')
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
        best_val = val_stats['loss']
        print(
            'epoch=0 '
            f'val_{objective_name}={val_stats["loss"]:.5f} '
            f'val_cosine={val_stats["cosine"]:.5f} '
            f'val_oos={val_stats["oos_probability"]:.5f} '
            f'lr={_current_lr(optimizer):.2e}'
        )
        save_checkpoint(output, model, train_config=train_config)
        print(f'saved checkpoint to {output} val_{objective_name}={best_val:.5f}')

    for epoch in range(1, epochs + 1):
        epoch_lr = _current_lr(optimizer)
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_name != 'plateau' else None,
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

        if val_stats['loss'] <= best_val:
            best_val = val_stats['loss']
            save_checkpoint(output, model, train_config=train_config)
            print(f'saved checkpoint to {output} val_{objective_name}={best_val:.5f}')

        if scheduler_name == 'plateau' and scheduler is not None:
            scheduler.step(val_stats['loss'])
        elif scheduler is not None and scheduler_interval == 'epoch':
            scheduler.step()

    history_path = Path(str(output).replace('.pt', '.history.json'))
    with open(history_path, 'w') as fp:
        json.dump(history, fp, indent=2)
    return history


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
