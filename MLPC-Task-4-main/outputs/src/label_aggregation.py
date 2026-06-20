"""Label aggregation utilities for MLPC 2026 Task 4."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def aggregate_annotations(
    annotation_tensor: np.ndarray,
    min_vote_fraction: float = 0.5,
) -> np.ndarray:
    """Convert aligned annotation overlaps [T, C, A] to binary labels [T, C].

    Each annotator votes positive for a class in a segment when the stored
    overlap value is greater than zero. The final class label is positive when
    at least half of available annotators voted positive.
    """
    if annotation_tensor.ndim != 3:
        raise ValueError(
            f"Expected annotation tensor with shape [T, C, A], got {annotation_tensor.shape}."
        )
    n_annotators = annotation_tensor.shape[2]
    if n_annotators < 1:
        raise ValueError("Cannot aggregate annotations without annotators.")

    votes = annotation_tensor > 0
    required_votes = max(1, math.ceil(n_annotators * min_vote_fraction))
    return (votes.sum(axis=2) >= required_votes).astype(np.uint8)


def label_summary(labels: np.ndarray) -> Dict[str, float]:
    """Return compact density statistics for a binary multi-label matrix."""
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape [N, C], got {labels.shape}.")
    n_samples, n_classes = labels.shape
    positives = int(labels.sum())
    return {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "positive_labels": positives,
        "label_density": float(positives / max(1, n_samples * n_classes)),
        "segments_with_any_label": int((labels.sum(axis=1) > 0).sum()),
    }
