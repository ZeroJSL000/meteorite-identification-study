"""Configure training, validation, and inference from YAML and environment values."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as error:
    raise ImportError("PyYAML is required; install the project dependencies.") from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class PathConfig:
    """Paths for project inputs, outputs, and experiment logs."""

    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    )
    weights_dir: Path = field(
        default_factory=lambda: Path(os.getenv("WEIGHTS_DIR", PROJECT_ROOT / "weights"))
    )
    outputs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
    )
    docs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
    )
    train_image_dir: Path = field(init=False)
    test_image_dir: Path = field(init=False)
    train_csv: Path = field(init=False)
    sample_submission_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.train_image_dir = Path(
            os.getenv("TRAIN_IMAGE_DIR", self.data_dir / "train_images")
        )
        self.test_image_dir = Path(
            os.getenv("TEST_IMAGE_DIR", self.data_dir / "test_images")
        )
        self.train_csv = Path(
            os.getenv("TRAIN_CSV", self.data_dir / "train_labels.csv")
        )
        self.sample_submission_csv = Path(
            os.getenv(
                "SAMPLE_SUBMISSION_CSV",
                self.data_dir / "sample_submission.csv",
            )
        )


@dataclass
class DataConfig:
    """Column names, image resolution, and DataLoader settings."""

    id_col: str = "id"
    label_col: str = "label"
    image_size: int = 384
    num_workers: int = field(default_factory=lambda: _env_int("NUM_WORKERS", 8))


@dataclass
class ModelConfig:
    """Configuration for a timm classifier with GAP and one output logit."""

    name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        )
    )
    pooling: str = "gap"
    drop_rate: float = field(default_factory=lambda: _env_float("DROP_RATE", 0.0))
    drop_path_rate: float = field(
        default_factory=lambda: _env_float("DROP_PATH_RATE", 0.0)
    )
    use_pretrained: bool = field(
        default_factory=lambda: _env_bool("USE_PRETRAINED", True)
    )
    pretrained_file: str = field(
        default_factory=lambda: os.getenv(
            "PRETRAINED_FILE",
            "convnextv2_base.fcmae_ft_in22k_in1k.pth",
        )
    )


@dataclass
class AugmentConfig:
    """Training augmentations; validation and inference preserve image shape."""

    mode: str = "strong"
    color_jitter: float = 0.45
    clahe_probability: float = 0.6
    geometry_probability: float = 0.5
    degradation_probability: float = 0.35
    coarse_dropout_enabled: bool = True
    coarse_dropout_probability: float = 0.45
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height: int = 64
    coarse_dropout_max_width: int = 64
    tta_names: tuple[str, ...] = ("identity",)


@dataclass
class TrainConfig:
    """Training settings for BCE, layerwise learning rates, and AdamW."""

    seed: int = field(default_factory=lambda: _env_int("SEED", 2026))
    folds: int = field(default_factory=lambda: _env_int("N_FOLDS", 5))
    epochs: int = field(default_factory=lambda: _env_int("EPOCHS", 24))
    batch_size: int = field(default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 8))
    valid_batch_size: int = field(
        default_factory=lambda: _env_int("VALID_BATCH_SIZE", 16)
    )
    accumulation_steps: int = field(
        default_factory=lambda: _env_int("GRAD_ACCUMULATION_STEPS", 2)
    )
    learning_rate: float = field(
        default_factory=lambda: _env_float("LEARNING_RATE", 3.0e-5)
    )
    backbone_lr_factor: float = field(
        default_factory=lambda: _env_float("BACKBONE_LR_FACTOR", 0.1)
    )
    min_learning_rate: float = field(
        default_factory=lambda: _env_float("MIN_LEARNING_RATE", 2.0e-7)
    )
    weight_decay: float = field(
        default_factory=lambda: _env_float("WEIGHT_DECAY", 0.05)
    )
    patience: int = field(default_factory=lambda: _env_int("PATIENCE", 7))
    max_grad_norm: float = 1.0
    amp: bool = field(default_factory=lambda: _env_bool("AMP", True))
    pos_weight_mode: str = "none"
    label_smoothing: float = 0.0
    ema_enabled: bool = False
    ema_decay: float = 0.999
    loss_name: str = "BCEWithLogitsLoss"
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"


@dataclass
class InferenceConfig:
    """Settings for probability averaging across fold models."""

    batch_size: int = field(default_factory=lambda: _env_int("INFER_BATCH_SIZE", 24))
    threshold: float = field(
        default_factory=lambda: _env_float("FALLBACK_THRESHOLD", 0.5)
    )
    target_positive_count: int | None = None
    expected_rows: int | None = None


@dataclass
class PseudoLabelConfig:
    """Use optional external test predictions as low-weight pseudo-labels."""

    enabled: bool = False
    csv: Path | None = None
    weight: float = 0.40


@dataclass
class ExperimentConfig:
    """Complete configuration for one experiment."""

    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "EXPERIMENT_NAME",
            "convnextv2_base_384_ema_smooth",
        )
    )
    config_path: Path | None = None
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)

    @property
    def experiment_output_dir(self) -> Path:
        return self.paths.outputs_dir / self.experiment_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.paths.weights_dir / "checkpoints" / self.experiment_name

    @property
    def pretrained_path(self) -> Path:
        return self.paths.weights_dir / self.model.pretrained_file

    @property
    def experiment_log_path(self) -> Path:
        return self.paths.docs_dir / "experiment_log.md"

    def create_output_dirs(self) -> None:
        self.paths.weights_dir.mkdir(parents=True, exist_ok=True)
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.docs_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert nested configuration values to JSON-compatible types."""

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return normalize(asdict(self))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _apply_yaml(config: ExperimentConfig, payload: dict[str, Any], path: Path) -> None:
    """Apply YAML values to the experiment dataclasses."""

    config.config_path = path
    if "experiment_name" in payload:
        config.experiment_name = str(payload["experiment_name"])

    model = payload.get("model", {}) or {}
    if not isinstance(model, dict):
        raise TypeError("model config must be a mapping.")
    if "backbone" in model:
        config.model.name = str(model["backbone"])
    if "name" in model:
        config.model.name = str(model["name"])
    if "image_size" in model:
        config.data.image_size = int(model["image_size"])
    if "pooling" in model:
        config.model.pooling = str(model["pooling"]).lower()
    if config.model.pooling != "gap":
        raise ValueError("Only pooling: gap is supported by this pipeline.")
    if "pretrained_file" in model:
        config.model.pretrained_file = str(model["pretrained_file"])
    elif config.model.name == "cswin_base_384":
        config.model.pretrained_file = "cswin_base_384.pth"
    if "drop_rate" in model:
        config.model.drop_rate = float(model["drop_rate"])
    if "drop_path_rate" in model:
        config.model.drop_path_rate = float(model["drop_path_rate"])

    train = payload.get("train", {}) or {}
    if not isinstance(train, dict):
        raise TypeError("train config must be a mapping.")
    if "batch_size" in train:
        config.train.batch_size = int(train["batch_size"])
    if "valid_batch_size" in train:
        config.train.valid_batch_size = int(train["valid_batch_size"])
    if "epochs" in train:
        config.train.epochs = int(train["epochs"])
    if "base_lr" in train:
        config.train.learning_rate = float(train["base_lr"])
    if "learning_rate" in train:
        config.train.learning_rate = float(train["learning_rate"])
    if "backbone_lr_factor" in train:
        config.train.backbone_lr_factor = float(train["backbone_lr_factor"])
    if "pos_weight_mode" in train:
        config.train.pos_weight_mode = str(train["pos_weight_mode"]).lower()
    if "label_smoothing" in train:
        config.train.label_smoothing = float(train["label_smoothing"])
    if not 0.0 <= config.train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing must be in [0, 1).")
    if config.train.pos_weight_mode not in {"dynamic", "none"}:
        raise ValueError("train.pos_weight_mode must be dynamic or none.")
    for key, attr, caster in (
        ("folds", "folds", int),
        ("seed", "seed", int),
        ("accumulation_steps", "accumulation_steps", int),
        ("weight_decay", "weight_decay", float),
        ("min_lr", "min_learning_rate", float),
        ("min_learning_rate", "min_learning_rate", float),
        ("patience", "patience", int),
        ("max_grad_norm", "max_grad_norm", float),
    ):
        if key in train:
            setattr(config.train, attr, caster(train[key]))

    ema = payload.get("ema", {}) or train.get("ema", {}) or {}
    if not isinstance(ema, dict):
        raise TypeError("ema config must be a mapping.")
    if "enabled" in ema:
        config.train.ema_enabled = bool(ema["enabled"])
    if "decay" in ema:
        config.train.ema_decay = float(ema["decay"])
    if not 0.0 < config.train.ema_decay < 1.0:
        raise ValueError("ema.decay must be in (0, 1).")

    augmentation = payload.get("augmentation", {}) or {}
    if not isinstance(augmentation, dict):
        raise TypeError("augmentation config must be a mapping.")
    if "mode" in augmentation:
        config.augment.mode = str(augmentation["mode"]).lower()
    if config.augment.mode not in {"strong", "minimal"}:
        raise ValueError("augmentation.mode must be strong or minimal.")
    coarse = augmentation.get("coarse_dropout", {}) or {}
    if not isinstance(coarse, dict):
        raise TypeError("augmentation.coarse_dropout must be a mapping.")
    if "enabled" in coarse:
        config.augment.coarse_dropout_enabled = bool(coarse["enabled"])
    if "probability" in coarse:
        config.augment.coarse_dropout_probability = float(coarse["probability"])
    if "max_holes" in coarse:
        config.augment.coarse_dropout_max_holes = int(coarse["max_holes"])
    if "max_height" in coarse:
        config.augment.coarse_dropout_max_height = int(coarse["max_height"])
    if "max_width" in coarse:
        config.augment.coarse_dropout_max_width = int(coarse["max_width"])

    data = payload.get("data", {}) or {}
    if not isinstance(data, dict):
        raise TypeError("data config must be a mapping.")
    if "num_workers" in data:
        config.data.num_workers = int(data["num_workers"])
    if "id_col" in data:
        config.data.id_col = str(data["id_col"])
    if "label_col" in data:
        config.data.label_col = str(data["label_col"])

    inference = payload.get("inference", {}) or {}
    if not isinstance(inference, dict):
        raise TypeError("inference config must be a mapping.")
    if "batch_size" in inference:
        config.inference.batch_size = int(inference["batch_size"])
    if "threshold" in inference:
        config.inference.threshold = float(inference["threshold"])
    if "target_positive_count" in inference:
        value = inference["target_positive_count"]
        config.inference.target_positive_count = None if value is None else int(value)
    if config.inference.target_positive_count is not None and config.inference.target_positive_count < 0:
        raise ValueError("inference.target_positive_count must be nonnegative.")
    if "expected_rows" in inference:
        value = inference["expected_rows"]
        config.inference.expected_rows = None if value is None else int(value)
    if config.inference.expected_rows is not None and config.inference.expected_rows <= 0:
        raise ValueError("inference.expected_rows must be positive.")

    pseudo_label = payload.get("pseudo_label", {}) or {}
    if not isinstance(pseudo_label, dict):
        raise TypeError("pseudo_label config must be a mapping.")
    if "enabled" in pseudo_label:
        config.pseudo_label.enabled = bool(pseudo_label["enabled"])
    if "csv" in pseudo_label:
        config.pseudo_label.csv = _as_path(pseudo_label["csv"])
    if "weight" in pseudo_label:
        config.pseudo_label.weight = float(pseudo_label["weight"])
    if config.pseudo_label.enabled and config.pseudo_label.csv is None:
        raise ValueError("pseudo_label.csv is required when pseudo_label.enabled=true.")
    if not 0.0 < config.pseudo_label.weight <= 1.0:
        raise ValueError("pseudo_label.weight must be in (0, 1].")

    paths = payload.get("paths", {}) or {}
    if not isinstance(paths, dict):
        raise TypeError("paths config must be a mapping.")
    for key in ("data_dir", "weights_dir", "outputs_dir", "docs_dir"):
        if key in paths:
            setattr(
                config.paths, key, _as_path(paths[key]) or getattr(config.paths, key)
            )
    config.paths.__post_init__()


def resolve_image_dir(directory: Path, expected_image: str | None = None) -> Path:
    """Resolve image folders that contain a repeated directory level."""

    candidates = (directory, directory / directory.name)
    for candidate in candidates:
        if expected_image and (candidate / expected_image).is_file():
            return candidate
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.[jJpP][pPnN][gG]")):
            return candidate
    raise FileNotFoundError(
        f"Image directory or sample {expected_image!r} not found; checked: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def load_config(
    config_path: str | Path | None = None,
    validate_paths: bool = False,
) -> ExperimentConfig:
    """Build a configuration and optionally override it with a YAML file."""

    config = ExperimentConfig()
    path = _as_path(config_path)
    if path is not None:
        _apply_yaml(config, _load_yaml(path), path)
    config.create_output_dirs()
    if validate_paths:
        required_files = (
            config.paths.train_csv,
            config.paths.sample_submission_csv,
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Required data files are missing: {missing}")
        if config.model.use_pretrained and not config.pretrained_path.is_file():
            raise FileNotFoundError(
                f"Offline pretrained weights not found: {config.pretrained_path}. "
                "See README.md for checkpoint preparation."
            )
    return config
