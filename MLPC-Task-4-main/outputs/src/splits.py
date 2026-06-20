"""Recording and collector aware splits for MLPC 2026 Task 4."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}


def make_group_split(
    recording_info: pd.DataFrame,
    class_names: Iterable[str],
    group_col: str = "collector_id",
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
    n_iter: int = 800,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Assign whole collector groups to train/validation/test.

    A deterministic random search is used because sklearn has no built-in
    stratified multi-label group split. The selected split minimizes class,
    segment, and recording distribution drift from the target ratios.
    """
    class_names = list(class_names)
    class_cols = [f"label_{name}" for name in class_names]
    splits = list(split_ratios.keys())
    ratios = np.array([split_ratios[name] for name in splits], dtype=np.float64)
    ratios = ratios / ratios.sum()

    grouped = (
        recording_info.groupby(group_col)
        .agg(
            n_recordings=("filename", "count"),
            n_segments=("n_segments", "sum"),
            **{col: (col, "sum") for col in class_cols},
        )
        .reset_index()
    )
    group_ids = grouped[group_col].to_numpy()
    n_groups = len(group_ids)
    n_train = int(round(n_groups * ratios[0]))
    n_validation = int(round(n_groups * ratios[1]))
    n_train = min(max(1, n_train), n_groups - 2)
    n_validation = min(max(1, n_validation), n_groups - n_train - 1)

    total_class = grouped[class_cols].sum().to_numpy(dtype=np.float64)
    total_segments = float(grouped["n_segments"].sum())
    total_recordings = float(grouped["n_recordings"].sum())

    rng = np.random.default_rng(seed)
    best_score = np.inf
    best_assignment: Dict[object, str] | None = None
    best_stats: Dict[str, object] = {}

    for _ in range(n_iter):
        order = rng.permutation(n_groups)
        assignment = {}
        for pos, group_idx in enumerate(order):
            if pos < n_train:
                split = "train"
            elif pos < n_train + n_validation:
                split = "validation"
            else:
                split = "test"
            assignment[group_ids[group_idx]] = split

        stats = _score_assignment(grouped, assignment, group_col, class_cols, splits, ratios)
        penalty = 0.0
        class_counts = stats["class_counts"]
        if (class_counts == 0).any():
            penalty += 5.0 * float((class_counts == 0).sum())
        score = (
            _squared_error(stats["class_props"], ratios)
            + 0.25 * _squared_error(stats["segment_props"], ratios)
            + 0.25 * _squared_error(stats["recording_props"], ratios)
            + penalty
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment
            best_stats = stats

    if best_assignment is None:
        raise RuntimeError("Could not create a split assignment.")

    split_info = recording_info.copy()
    split_info["split"] = split_info[group_col].map(best_assignment)
    diagnostics = {
        "group_col": group_col,
        "collector_grouping_used": True,
        "n_groups": int(n_groups),
        "score": float(best_score),
        "split_ratios": dict(split_ratios),
        "all_classes_present_each_split": bool((best_stats["class_counts"] > 0).all()),
        "class_counts": {
            split: {
                class_name: int(best_stats["class_counts"][split_idx, class_idx])
                for class_idx, class_name in enumerate(class_names)
            }
            for split_idx, split in enumerate(splits)
        },
    }
    return split_info, diagnostics


def _score_assignment(
    grouped: pd.DataFrame,
    assignment: Mapping[object, str],
    group_col: str,
    class_cols: List[str],
    splits: List[str],
    ratios: np.ndarray,
) -> Dict[str, np.ndarray]:
    assigned = grouped.copy()
    assigned["split"] = assigned[group_col].map(assignment)

    class_counts = []
    segment_counts = []
    recording_counts = []
    for split in splits:
        subset = assigned[assigned["split"] == split]
        class_counts.append(subset[class_cols].sum().to_numpy(dtype=np.float64))
        segment_counts.append(float(subset["n_segments"].sum()))
        recording_counts.append(float(subset["n_recordings"].sum()))
    class_counts_array = np.vstack(class_counts)
    total_class = np.maximum(1.0, class_counts_array.sum(axis=0, keepdims=True))
    class_props = class_counts_array / total_class
    segment_props = np.array(segment_counts, dtype=np.float64) / max(1.0, sum(segment_counts))
    recording_props = np.array(recording_counts, dtype=np.float64) / max(1.0, sum(recording_counts))
    return {
        "class_counts": class_counts_array,
        "class_props": class_props,
        "segment_props": segment_props,
        "recording_props": recording_props,
    }


def _squared_error(values: np.ndarray, target: np.ndarray) -> float:
    if values.ndim == 2:
        target = target.reshape(-1, 1)
    return float(np.mean((values - target) ** 2))


def add_segment_splits(segment_info: pd.DataFrame, recording_splits: pd.DataFrame) -> pd.DataFrame:
    """Copy split labels from recording table to segment table."""
    split_map = recording_splits.set_index("filename")["split"].to_dict()
    output = segment_info.copy()
    output["split"] = output["filename"].map(split_map)
    if output["split"].isna().any():
        missing = output.loc[output["split"].isna(), "filename"].unique()[:5]
        raise ValueError(f"Missing split assignment for files: {missing}")
    return output


def verify_no_leakage(recording_splits: pd.DataFrame, group_col: str = "collector_id") -> Dict[str, bool]:
    """Check that recordings and collectors are unique to one split."""
    file_split_counts = recording_splits.groupby("filename")["split"].nunique()
    group_split_counts = recording_splits.groupby(group_col)["split"].nunique()
    return {
        "recording_leakage": bool((file_split_counts > 1).any()),
        "collector_leakage": bool((group_split_counts > 1).any()),
    }


def split_summary(
    recording_splits: pd.DataFrame,
    segment_info: pd.DataFrame,
    y: np.ndarray,
    class_names: Iterable[str],
) -> pd.DataFrame:
    """Create a split summary table with recording, segment, and label counts."""
    class_names = list(class_names)
    rows = []
    for split in ["train", "validation", "test"]:
        rec_subset = recording_splits[recording_splits["split"] == split]
        seg_subset = segment_info[segment_info["split"] == split]
        idx = seg_subset["global_index"].to_numpy(dtype=int)
        y_split = y[idx]
        row = {
            "split": split,
            "recordings": int(len(rec_subset)),
            "collectors": int(rec_subset["collector_id"].nunique()),
            "segments": int(len(seg_subset)),
            "positive_labels": int(y_split.sum()),
            "segments_with_any_label": int((y_split.sum(axis=1) > 0).sum()),
            "label_density": float(y_split.sum() / max(1, y_split.size)),
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"label_{class_name}"] = int(y_split[:, class_idx].sum())
        rows.append(row)
    return pd.DataFrame(rows)
