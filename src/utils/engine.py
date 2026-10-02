"""Single-epoch training, validation, and inference routines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.config import TrainConfig
from utils.metrics import sigmoid


class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCE with optional label smoothing and positive-class weighting."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        if self.label_smoothing > 0.0:
            targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()


def build_loss(
    train_config: TrainConfig,
    device: torch.device,
    pos_weight: float | None = None,
) -> nn.Module:
    """Build BCE loss; omit positive-class weighting in none mode."""

    weight_tensor: Tensor | None = None
    if train_config.pos_weight_mode == "dynamic":
        if pos_weight is None or not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError(f"Dynamic pos_weight must be positive and finite: {pos_weight}")
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    elif train_config.pos_weight_mode != "none":
        raise ValueError(f"Unsupported pos_weight_mode: {train_config.pos_weight_mode}")
    return SmoothedBCEWithLogitsLoss(
        label_smoothing=train_config.label_smoothing,
        pos_weight=weight_tensor,
    )


def _autocast_context(device: torch.device, enabled: bool) -> Callable:
    return lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    train_config: TrainConfig,
    model_ema: Any | None = None,
) -> float:
    """Train one epoch with AMP, gradient accumulation, and optional EMA."""

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=False)
    autocast = _autocast_context(device, train_config.amp)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
            loss = criterion(logits, targets, sample_weight=sample_weight)
            scaled_loss = loss / train_config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        is_update_step = (step + 1) % train_config.accumulation_steps == 0
        is_last_step = step + 1 == len(loader)
        if is_update_step or is_last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_grad_norm,
            )
            old_scale = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = old_scale is not None and scaler.get_scale() < old_scale
            if model_ema is not None and not step_was_skipped:
                model_ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach().item()) * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    description: str = "Valid",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return validation loss, positive-class probabilities, and targets."""

    model.eval()
    running_loss = 0.0
    logits_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
            loss = criterion(logits, targets)
        running_loss += float(loss.item()) * images.size(0)
        logits_list.append(logits.float().cpu().numpy())
        target_list.append(targets.cpu().numpy())
    probabilities = sigmoid(np.concatenate(logits_list))
    targets = np.concatenate(target_list).astype(np.int64)
    return running_loss / len(loader.dataset), probabilities, targets


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    description: str,
) -> np.ndarray:
    """Predict positive-class probabilities for unlabeled images."""

    model.eval()
    logits_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
        logits_list.append(logits.float().cpu().numpy())
    return sigmoid(np.concatenate(logits_list))
