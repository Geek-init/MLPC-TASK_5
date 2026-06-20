"""Evaluation helpers for multi-label classification."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities or scores to binary predictions."""
    return (probabilities >= threshold).astype(np.uint8)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
) -> Dict[str, float | str]:
    """Compute summary multi-label metrics."""
    return {
        "model": model_name,
        "split": split,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "true_label_density": float(y_true.sum() / max(1, y_true.size)),
        "pred_label_density": float(y_pred.sum() / max(1, y_pred.size)),
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Iterable[str],
    model_name: str,
    split: str,
) -> pd.DataFrame:
    """Return per-class precision, recall, F1, and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split,
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def file_level_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a stable file-level score for case study selection."""
    return float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
