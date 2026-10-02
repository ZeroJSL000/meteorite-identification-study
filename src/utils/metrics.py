"""Binary classification metrics and F1-based threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class BinaryMetrics:
    """Scores measured at one binary decision threshold."""

    f1: float
    precision: float
    recall: float
    threshold: float
    positive_predictions: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to positive-class probabilities with clipping."""

    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    """Compute F1, precision, and recall at the given threshold."""

    predictions = (probabilities >= threshold).astype(np.int64)
    return BinaryMetrics(
        f1=float(f1_score(targets, predictions, zero_division=0)),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        recall=float(recall_score(targets, predictions, zero_division=0)),
        threshold=float(threshold),
        positive_predictions=int(predictions.sum()),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.40,
    threshold_max: float = 0.60,
    step: float = 0.01,
) -> BinaryMetrics:
    """Select the F1-maximizing threshold on a fixed probability grid.

    A fixed grid avoids tuning to a presumed test-set class count. The narrow
    default range limits extreme thresholds from noisy out-of-fold predictions.
    """

    targets = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if targets.size == 0 or targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must be nonempty and aligned.")
    if np.unique(targets).size < 2:
        raise ValueError("Threshold search requires both positive and negative targets.")
    if not 0.0 <= threshold_min < threshold_max <= 1.0 or step <= 0:
        raise ValueError("Threshold range or step is invalid.")

    candidates = np.round(
        np.arange(threshold_min, threshold_max + step / 2.0, step),
        decimals=10,
    )
    best = compute_binary_metrics(targets, probabilities, float(candidates[0]))
    for threshold in candidates[1:]:
        current = compute_binary_metrics(targets, probabilities, float(threshold))
        if current.f1 > best.f1:
            best = current
        elif np.isclose(current.f1, best.f1) and abs(current.threshold - 0.5) < abs(
            best.threshold - 0.5
        ):
            best = current
    return best


def optimize_f1_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> BinaryMetrics:
    """Compatibility wrapper for F1 threshold optimization."""

    return find_optimal_threshold(targets, probabilities)
