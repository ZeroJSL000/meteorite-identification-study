"""Build a submission from mean probabilities across fold checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, load_config, resolve_image_dir
from utils.dataset import BinaryImageDataset, build_valid_transforms
from utils.engine import predict_probabilities
from models.builder import build_model
from utils.checkpoint import load_weights_flexible
from utils.ranking import select_top_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble inference across folds")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def build_test_loader(
    frame: pd.DataFrame,
    image_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    """Build the aspect-preserving test data loader."""

    dataset = BinaryImageDataset(
        frame=frame,
        image_dir=image_dir,
        transform=build_valid_transforms(config.data.image_size),
        id_col=config.data.id_col,
        label_col=None,
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.inference.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.data.num_workers > 0,
    }
    if config.data.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def apply_target_positive_count(
    submission: pd.DataFrame,
    probabilities: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Assign positive labels to the highest-ranked probabilities."""

    target_count = config.inference.target_positive_count
    if target_count is None:
        return submission
    if target_count > len(submission):
        raise ValueError(
            f"target_positive_count={target_count} exceeds {len(submission)} rows."
        )
    calibrated = submission.copy()
    labels = select_top_count(probabilities, target_count)
    calibrated[config.data.label_col] = labels
    path = config.experiment_output_dir / "submission_count_calibrated.csv"
    calibrated.to_csv(path, index=False)
    print(
        f"Positive-count calibration applied: target_positive={target_count}, "
        f"target_negative={len(calibrated) - target_count}; saved={path}"
    )
    return calibrated


def load_optimal_threshold(
    metadata: dict[str, object], config: ExperimentConfig
) -> float:
    """Read the OOF threshold, tolerating paths from archived metadata."""

    candidate_paths = [config.experiment_output_dir / "optimal_threshold.json"]
    path_value = metadata.get("optimal_threshold_file")
    if path_value:
        candidate_paths.append(Path(str(path_value)))

    checked: list[str] = []
    for threshold_path in candidate_paths:
        checked.append(str(threshold_path))
        if not threshold_path.is_file():
            continue
        with threshold_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("experiment_name") != metadata.get("experiment_name"):
            raise ValueError("Threshold file and checkpoint experiment names differ.")
        threshold = float(payload["optimal_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid OOF threshold: {threshold}")
        return threshold

    oof_metrics = metadata.get("oof_metrics")
    if isinstance(oof_metrics, dict) and "threshold" in oof_metrics:
        threshold = float(oof_metrics["threshold"])
        if 0.0 < threshold < 1.0:
            print(
                "Local OOF threshold file was not found; using checkpoint metadata "
                f"threshold={threshold:.4f}."
            )
            return threshold

    raise FileNotFoundError(
        "No OOF threshold file or threshold in checkpoint metadata was found. "
        f"Checked: {checked}"
    )


def run_prediction(
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> None:
    """Average fold probabilities and apply optional count calibration."""

    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata was not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    config.model.name = metadata["model_name"]
    config.data.image_size = int(metadata["image_size"])
    checkpoint_files = metadata["checkpoint_files"]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files are listed in {metadata_path}")
    threshold = load_optimal_threshold(metadata, config)

    sample_submission = pd.read_csv(config.paths.sample_submission_csv)
    required_columns = {config.data.id_col, config.data.label_col}
    if not required_columns.issubset(sample_submission.columns):
        raise KeyError(f"Submission template requires columns: {sorted(required_columns)}")
    if sample_submission.empty:
        raise ValueError("Submission template contains no images.")
    expected_rows = config.inference.expected_rows
    if expected_rows is not None and len(sample_submission) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} submission rows, found {len(sample_submission)}. "
            "Review the evaluation stage and count calibration configuration."
        )
    first_image = str(sample_submission.iloc[0][config.data.id_col])
    test_image_dir = resolve_image_dir(config.paths.test_image_dir, first_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability_sum = np.zeros(len(sample_submission), dtype=np.float64)
    print(f"Device: {device}; test images: {test_image_dir}; OOF threshold: {threshold:.2f}")

    loader = build_test_loader(sample_submission, test_image_dir, config, device)
    for filename in checkpoint_files:
        checkpoint_path = checkpoint_dir / filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Fold checkpoint is missing: {checkpoint_path}")
        model = build_model(config.model).to(device)
        loaded_count, _ = load_weights_flexible(model, checkpoint_path)
        if loaded_count != len(model.state_dict()):
            raise ValueError(
                f"Checkpoint {filename} only matched {loaded_count} of "
                f"{len(model.state_dict())} model tensors."
            )
        print(f"Loaded {filename}: {loaded_count} matching parameters")
        probability_sum += predict_probabilities(
            model,
            loader,
            device,
            config.train.amp,
            description=filename,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = probability_sum / len(checkpoint_files)
    model_labels = (probabilities >= threshold).astype(int)
    model_submission = sample_submission.copy()
    model_submission[config.data.label_col] = model_labels
    probability_frame = model_submission[[config.data.id_col]].copy()
    probability_frame["probability"] = probabilities
    probability_frame["prediction"] = model_submission[config.data.label_col]
    model_submission_path = config.experiment_output_dir / "submission_model_only.csv"
    probability_path = config.experiment_output_dir / "submission_probabilities.csv"
    model_submission.to_csv(model_submission_path, index=False)
    probability_frame.to_csv(probability_path, index=False)

    submission = model_submission
    submission = apply_target_positive_count(submission, probabilities, config)
    submission_path = config.experiment_output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(
        f"Submission saved: {submission_path}; "
        f"model-only submission: {model_submission_path}; "
        f"predicted positives: {int(submission[config.data.label_col].sum())}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
        config.create_output_dirs()
    checkpoint_dir = args.checkpoint_dir or config.checkpoint_dir
    run_prediction(config, checkpoint_dir)


if __name__ == "__main__":
    main()
