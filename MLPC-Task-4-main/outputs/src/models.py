"""Model factories and prediction helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier


def make_linear_logistic_model(config: Dict[str, object], seed: int) -> OneVsRestClassifier:
    """Create a linear logistic classifier trained with SGD."""
    base = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config.get("alpha", 1e-4)),
        class_weight=config.get("class_weight", "balanced"),
        max_iter=int(config.get("max_iter", 50)),
        tol=float(config.get("tol", 2e-3)),
        random_state=seed,
        n_jobs=1,
    )
    return OneVsRestClassifier(base, n_jobs=4)


def make_linear_ridge_model(config: Dict[str, object], seed: int) -> OneVsRestClassifier:
    """Create a fast one-vs-rest linear ridge classifier."""
    base = RidgeClassifier(
        alpha=float(config.get("alpha", 1.0)),
        class_weight=config.get("class_weight", "balanced"),
        random_state=seed,
    )
    return OneVsRestClassifier(base, n_jobs=4)


def make_random_forest_model(config: Dict[str, object], seed: int) -> RandomForestClassifier:
    """Create a multi-output random forest classifier."""
    max_depth = config.get("max_depth", 18)
    if max_depth == "none":
        max_depth = None
    return RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 60)),
        max_depth=max_depth,
        min_samples_leaf=int(config.get("min_samples_leaf", 2)),
        max_features=str(config.get("max_features", "sqrt")),
        max_samples=config.get("max_samples", 0.6),
        class_weight=config.get("class_weight", "balanced_subsample"),
        random_state=seed,
        n_jobs=-1,
    )


def predict_probabilities(model, x: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for sklearn multi-label models."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
    else:
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))

    if isinstance(probabilities, list):
        columns = []
        for class_prob in probabilities:
            if class_prob.ndim == 1:
                columns.append(class_prob)
            elif class_prob.shape[1] == 1:
                columns.append(np.zeros(class_prob.shape[0], dtype=np.float32))
            else:
                columns.append(class_prob[:, 1])
        return np.vstack(columns).T.astype(np.float32)
    return np.asarray(probabilities, dtype=np.float32)


def empirical_frequency_predictions(
    y_train: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random baseline using training label prevalence."""
    prevalence = y_train.mean(axis=0)
    rng = np.random.default_rng(seed)
    probabilities = np.tile(prevalence.reshape(1, -1), (n_samples, 1)).astype(np.float32)
    predictions = (rng.random((n_samples, y_train.shape[1])) < prevalence).astype(np.uint8)
    return predictions, probabilities
