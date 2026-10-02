"""Deterministic positive-count calibration for a fixed submission cohort."""

from __future__ import annotations

import math
from collections.abc import Sequence


def select_top_count(probabilities: Sequence[float], positive_count: int) -> list[int]:
    """Label the highest-ranked rows positive, breaking ties by input order."""

    if not 0 <= positive_count <= len(probabilities):
        raise ValueError("positive_count must be between zero and the row count")
    if any(not math.isfinite(float(value)) for value in probabilities):
        raise ValueError("probabilities must be finite")

    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    labels = [0] * len(probabilities)
    for index in order[:positive_count]:
        labels[index] = 1
    return labels
