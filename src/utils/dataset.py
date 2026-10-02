"""Binary image datasets and aspect-preserving augmentation pipelines."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from utils.config import AugmentConfig


def _pad_to_square(image_size: int) -> A.BasicTransform:
    """Pad to a square while preserving aspect ratio across library versions."""

    try:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=1.0,
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1.0,
        )


def _aspect_safe_resize(image_size: int) -> list[A.BasicTransform]:
    """Resize the longest edge and pad without distorting the object shape."""

    return [
        A.LongestMaxSize(max_size=image_size, p=1.0),
        _pad_to_square(image_size),
    ]


def _shift_scale_rotate(probability: float) -> A.BasicTransform:
    try:
        return A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=probability,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=probability,
        )


def _image_compression() -> A.BasicTransform:
    try:
        return A.ImageCompression(quality_range=(75, 98), p=1.0)
    except TypeError:
        return A.ImageCompression(quality_lower=75, quality_upper=98, p=1.0)


def _coarse_dropout(image_size: int, config: AugmentConfig) -> A.BasicTransform:
    """Occlude local shortcuts such as rulers and fingers when configured."""

    min_size = max(1, int(image_size * 0.03))
    max_height = min(config.coarse_dropout_max_height, image_size)
    max_width = min(config.coarse_dropout_max_width, image_size)
    if config.coarse_dropout_max_holes <= 0:
        raise ValueError("coarse_dropout_max_holes must be positive.")
    try:
        return A.CoarseDropout(
            num_holes_range=(1, config.coarse_dropout_max_holes),
            hole_height_range=(min_size, max_height),
            hole_width_range=(min_size, max_width),
            fill=0,
            p=config.coarse_dropout_probability,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=config.coarse_dropout_max_holes,
            max_height=max_height,
            max_width=max_width,
            min_height=min_size,
            min_width=min_size,
            fill_value=0,
            p=config.coarse_dropout_probability,
        )


def build_train_transforms(image_size: int, config: AugmentConfig) -> A.Compose:
    """Build training transforms; minimal mode avoids extra augmentation."""

    if config.mode == "minimal":
        return build_valid_transforms(image_size)

    transforms: list[A.BasicTransform] = [
        *_aspect_safe_resize(image_size),
        _shift_scale_rotate(config.geometry_probability),
        A.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
            hue=0.12,
            p=0.75,
        ),
        A.RandomGamma(gamma_limit=(70, 135), p=0.45),
        A.HueSaturationValue(
            hue_shift_limit=18,
            sat_shift_limit=32,
            val_shift_limit=22,
            p=0.5,
        ),
        A.CLAHE(
            clip_limit=(1.0, 2.5),
            tile_grid_size=(8, 8),
            p=config.clahe_probability,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(p=1.0),
                _image_compression(),
            ],
            p=config.degradation_probability,
        ),
    ]
    if config.coarse_dropout_enabled:
        transforms.append(_coarse_dropout(image_size, config))
    transforms.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(transforms)


def build_valid_transforms(image_size: int, tta_name: str = "identity") -> A.Compose:
    """Use aspect-preserving resize, padding, and normalization for evaluation."""

    if tta_name != "identity":
        raise ValueError("Inference only supports identity test-time augmentation.")
    return A.Compose(
        [
            *_aspect_safe_resize(image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


class BinaryImageDataset(Dataset):
    """Load CSV-listed images as single-label binary classification samples."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        transform: A.Compose,
        id_col: str = "id",
        label_col: str | None = "label",
        image_dir_col: str | None = None,
        sample_weight_col: str | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.id_col = id_col
        self.label_col = label_col
        self.image_dir_col = image_dir_col
        self.sample_weight_col = sample_weight_col
        if id_col not in self.frame:
            raise KeyError(f"Image ID column is missing: {id_col}")
        if label_col is not None and label_col not in self.frame:
            raise KeyError(f"Label column is missing: {label_col}")
        if image_dir_col is not None and image_dir_col not in self.frame:
            raise KeyError(f"Image directory column is missing: {image_dir_col}")
        if sample_weight_col is not None and sample_weight_col not in self.frame:
            raise KeyError(f"Sample weight column is missing: {sample_weight_col}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        image_id = str(self.frame.at[index, self.id_col])
        image_dir = self.image_dir
        if self.image_dir_col is not None:
            image_dir = Path(str(self.frame.at[index, self.image_dir_col]))
        image_path = image_dir / image_id
        try:
            with Image.open(image_path) as image:
                image_array = np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Failed to read image: {image_path}") from error
        sample: dict[str, Tensor | str | int] = {
            "image": self.transform(image=image_array)["image"],
            "id": image_id,
            "index": index,
        }
        if self.label_col is not None:
            sample["target"] = torch.tensor(
                float(self.frame.at[index, self.label_col]),
                dtype=torch.float32,
            )
        if self.sample_weight_col is not None:
            sample["sample_weight"] = torch.tensor(
                float(self.frame.at[index, self.sample_weight_col]),
                dtype=torch.float32,
            )
        return sample
