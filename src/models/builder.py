"""Build timm backbones and the legacy local CSWin model."""

from __future__ import annotations

from pathlib import Path

import timm
from torch import nn

from utils.config import ModelConfig
from utils.checkpoint import load_weights_flexible

# --------------------------------------------------------
# Optional local model registration. Standard timm models remain available when
# the CSWin module cannot be imported.
# --------------------------------------------------------
try:
    from models.cswin.cswin import CSWin_96_24322_base_384
except ImportError:
    CSWin_96_24322_base_384 = None


def build_model(
    config: ModelConfig,
    pretrained_path: Path | None = None,
    require_pretrained: bool = False,
) -> nn.Module:
    """Create a single-logit image classifier."""

    # ==========================================
    # Route the legacy CSWin architecture.
    # ==========================================
    if config.name == "cswin_base_384":
        if CSWin_96_24322_base_384 is None:
            raise RuntimeError("CSWin source is unavailable; check models/cswin/.")
        model = CSWin_96_24322_base_384(
            pretrained=False,
            num_classes=1,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            img_size=384  # Keep the architecture at its intended resolution.
        )

    # Add any future local architectures through an explicit branch here.

    # ==========================================
    # All other architectures use the timm registry.
    # ==========================================
    else:
        model = timm.create_model(
            config.name,
            pretrained=False,
            num_classes=1,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
        )

    # ==========================================
    # Load all compatible tensors from the optional pretrained checkpoint.
    # ==========================================
    if pretrained_path is not None and pretrained_path.is_file():
        loaded_count, skipped = load_weights_flexible(model, pretrained_path)
        print(
            f"Loaded {loaded_count} pretrained tensors from {pretrained_path}; "
            f"skipped {len(skipped)} incompatible tensors."
        )
    elif require_pretrained:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {pretrained_path}. "
            "Place it in weights/ before training."
        )

    return model
