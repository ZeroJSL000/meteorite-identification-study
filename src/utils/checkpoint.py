"""Extract and load compatible model checkpoint tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """Extract a state dictionary from a pretrained or training checkpoint."""

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint content must be a state dictionary.")
    for key in (
        "state_dict_ema",
        "state_dict",
        "model",
        "model_state_dict",
        "ema_state_dict",
    ):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_weights_flexible(model: nn.Module, path: Path) -> tuple[int, list[str]]:
    """Load compatible tensors and skip layers with mismatched shapes."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()

    compatible_state = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value

    if not compatible_state:
        raise ValueError(f"No compatible tensors were found in checkpoint: {path}")

    skipped = sorted(set(source_state) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), skipped
