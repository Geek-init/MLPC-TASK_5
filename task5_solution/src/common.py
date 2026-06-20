"""Shared utilities for the MLPC 2026 Task 5 (SED) non-bonus solution.

This module centralises everything the numbered pipeline scripts reuse:
constants extracted from the provided baseline, dataset paths, npz feature/label
loading, on-disk caching of heavy matrices, the baseline-faithful
seed-42 validation split, SED inference + interval conversion, median-filter
post-processing, and a thin wrapper around the *official* evaluate.py.

Nothing here modifies the original baseline or Task 4 files; evaluate.py is
imported read-only from challenge_baseline/.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = SRC_DIR.parent                       # task5_solution/
ROOT = SOLUTION_DIR.parent                          # MLPC/
DATASET_DIR = ROOT / "MLPC2026_challenge"
BASELINE_DIR = ROOT / "challenge_baseline"

PATH_TRAIN = DATASET_DIR / "train"
PATH_VAL = DATASET_DIR / "validation"
PATH_TEST = DATASET_DIR / "test"

CACHE_DIR = SOLUTION_DIR / "cache"
RESULTS_DIR = SOLUTION_DIR / "results"
FIGURES_DIR = SOLUTION_DIR / "figures"
SUBMISSION_DIR = SOLUTION_DIR / "submission"

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR, SUBMISSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Constants (reused verbatim from challenge_baseline/challenge_baseline.ipynb)
# --------------------------------------------------------------------------- #
SEED = 42

FEATURE_NAMES: List[str] = [
    "zcr_mean", "zcr_std", "zcr_min", "zcr_max",
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    "mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max",
    "mfcc_d_mean", "mfcc_d_std", "mfcc_d_min", "mfcc_d_max",
    "mfcc_d2_mean", "mfcc_d2_std", "mfcc_d2_min", "mfcc_d2_max",
    "flux_mean", "flux_std", "flux_min", "flux_max",
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    "bandwidth_mean", "bandwidth_std", "bandwidth_min", "bandwidth_max",
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_low_min", "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean", "energy_std", "energy_min", "energy_max",
    "power_mean", "power_std", "power_min", "power_max",
]

# 15 target classes, sorted alphabetically to match the .npz annotation axis order.
CLASS_NAMES: List[str] = [
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch", "microwave",
    "phone_ringing", "running_water", "toilet_flushing", "vacuum_cleaner",
    "wardrobe_drawer_open_close", "window_open_close",
]

SEGMENT_LENGTH = 1.0   # each feature vector covers a 1-second window
HOP_SIZE = 0.5         # segments extracted with 50% overlap

# Baseline trains a decision tree on a 50k random subsample (notebook cell 18).
BASELINE_MAX_TRAINING_SEGMENTS = 50_000

# --------------------------------------------------------------------------- #
# Official evaluator (imported read-only from the provided baseline folder)
# --------------------------------------------------------------------------- #
def _load_official_evaluate():
    import sys
    spec = importlib.util.spec_from_file_location(
        "official_evaluate", str(BASELINE_DIR / "evaluate.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__ in sys.modules.
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_official_evaluate()

# --------------------------------------------------------------------------- #
# Feature / label extraction (baseline-faithful)
# --------------------------------------------------------------------------- #
def build_feature_matrix(data: dict) -> np.ndarray:
    """Concatenate the named features from a loaded .npz dict -> (N_seg, 960)."""
    arrays = []
    for feat_name in FEATURE_NAMES:
        feat = data[feat_name]
        if feat.ndim == 1:
            feat = feat[:, np.newaxis]
        arrays.append(feat.astype(np.float32))
    return np.concatenate(arrays, axis=1)


def get_segment_labels(data: dict) -> np.ndarray:
    """Binary multilabel targets from the annotations tensor (N, C, A).

    Baseline rule: an annotator votes positive when overlap > 0; the class is
    positive when *more than half* of annotators agree (votes > A // 2).
    """
    annotations = data["annotations"]              # (N, C, A)
    binary = (annotations > 0).astype(int)         # (N, C, A)
    votes = binary.sum(axis=2)                     # (N, C)
    n_annotators = binary.shape[2]
    return (votes > (n_annotators // 2)).astype(np.uint8)


def list_npz(split_dir: Path) -> List[str]:
    """Sorted list of .npz feature files for a split (matches baseline glob)."""
    import glob
    return sorted(glob.glob(str(split_dir / "audio_features" / "*.npz")))


# --------------------------------------------------------------------------- #
# Baseline-faithful split of the provided validation set
# --------------------------------------------------------------------------- #
def validation_split() -> Tuple[List[str], List[str]]:
    """Reproduce the baseline split (notebook cell 14) exactly.

    rng = default_rng(42); shuffle sorted validation npz; first half ->
    development validation, second half -> non-hidden test.
    """
    val_files = list_npz(PATH_VAL)
    rng = np.random.default_rng(seed=SEED)
    val_files_shuffled = rng.permutation(val_files).tolist()
    n_val = len(val_files_shuffled) // 2
    dev_val = val_files_shuffled[:n_val]
    nonhidden_test = val_files_shuffled[n_val:]
    return dev_val, nonhidden_test


def to_wav(npz_path: str) -> str:
    return os.path.basename(npz_path).replace(".npz", ".wav")


# --------------------------------------------------------------------------- #
# Caching of heavy matrices
# --------------------------------------------------------------------------- #
def _cache_paths(split: str) -> Dict[str, Path]:
    return {
        "X": CACHE_DIR / f"{split}_X.npy",
        "Y": CACHE_DIR / f"{split}_Y.npy",
        "starts": CACHE_DIR / f"{split}_starts.npy",
        "index": CACHE_DIR / f"{split}_index.json",
    }


def build_split_cache(split: str, file_list: List[str], with_labels: bool) -> None:
    """Load every npz for a split, concatenate features (+labels), cache to disk.

    Stores X (N,960) float32, optional Y (N,15) uint8, starts (N,) float64, and
    an ordered index [[wav_name, row_start, count], ...] so per-file segments
    can be sliced back out for SED inference.
    """
    paths = _cache_paths(split)
    X_list, Y_list, starts_list, index = [], [], [], []
    row = 0
    for i, fp in enumerate(file_list):
        data = dict(np.load(fp, allow_pickle=True))
        X = build_feature_matrix(data)
        starts = np.asarray(data["start_time"], dtype=np.float64)
        n = X.shape[0]
        X_list.append(X)
        starts_list.append(starts)
        if with_labels:
            Y_list.append(get_segment_labels(data))
        index.append([to_wav(fp), row, n])
        row += n
        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i + 1}/{len(file_list)} files", flush=True)

    X_all = np.vstack(X_list).astype(np.float32)
    starts_all = np.concatenate(starts_list).astype(np.float64)
    np.save(paths["X"], X_all)
    np.save(paths["starts"], starts_all)
    if with_labels:
        Y_all = np.vstack(Y_list).astype(np.uint8)
        np.save(paths["Y"], Y_all)
    with open(paths["index"], "w") as fh:
        json.dump(index, fh)
    print(f"  [{split}] cached X={X_all.shape}, "
          f"{'Y=' + str(Y_all.shape) if with_labels else 'no labels'}, "
          f"files={len(index)}", flush=True)


def load_split_cache(
    split: str, with_labels: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[list]]:
    """Return (X, Y_or_None, starts, index) for a cached split."""
    paths = _cache_paths(split)
    X = np.load(paths["X"])
    starts = np.load(paths["starts"])
    with open(paths["index"]) as fh:
        index = json.load(fh)
    Y = np.load(paths["Y"]) if with_labels and paths["Y"].exists() else None
    return X, Y, starts, index


def subset_index(index: List[list], wav_names: set) -> List[list]:
    """Filter an index to a set of .wav filenames (preserves file order)."""
    return [entry for entry in index if entry[0] in wav_names]


# --------------------------------------------------------------------------- #
# Preprocessing + closed-form ridge
#
# Implementation note: on this arm64 macOS box numpy links against Accelerate.
# Plain BLAS gemm / LAPACK solve are fast, but sklearn's RidgeClassifier solver
# hangs and SGD/lbfgs logistic are very slow, while any joblib n_jobs>1 spawns
# loky workers that deadlock. We therefore standardize with numpy and fit ridge
# in closed form (weighted normal equations) -- identical to RidgeClassifier
# but fast and dependency-light. The features contain no NaNs (audited), so no
# imputation is needed.
# --------------------------------------------------------------------------- #
class Standardizer:
    """Numpy z-score standardizer fit on training data only (no BLAS, no hang)."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu_ = X.mean(axis=0).astype(np.float64)
        sd = X.std(axis=0).astype(np.float64)
        self.sd_ = np.where(sd == 0.0, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def ridge_grams(X: np.ndarray, Y: np.ndarray, class_weight="balanced"):
    """Per-class weighted normal-equation pieces for ridge with an unpenalized
    intercept. Returns (Grams, RHS, D) where each Gram is (D+1, D+1) = Xa^T W Xa
    and each RHS is (D+1,) = Xa^T (w * y_signed); Xa is X augmented with a ones
    column. Computed once, reused across alphas (alpha only shifts the diagonal).
    """
    N, D = X.shape
    Xa = np.empty((N, D + 1), dtype=np.float32)
    Xa[:, :D] = X
    Xa[:, D] = 1.0
    Grams, RHS = [], []
    for c in range(Y.shape[1]):
        pos = Y[:, c] > 0
        y = np.where(pos, 1.0, -1.0).astype(np.float32)
        if class_weight == "balanced":
            npos = max(1, int(pos.sum()))
            nneg = max(1, N - npos)
            w = np.where(pos, N / (2.0 * npos), N / (2.0 * nneg)).astype(np.float32)
        else:
            w = np.ones(N, dtype=np.float32)
        Xw = Xa * np.sqrt(w)[:, None]
        Grams.append((Xw.T @ Xw).astype(np.float64))
        RHS.append((Xa.T @ (w * y)).astype(np.float64))
    return Grams, RHS, D


class ClosedFormRidge:
    """One-vs-rest ridge classifier solved in closed form (numpy).

    Regresses signed targets {-1,+1} per class with class-balanced sample
    weights and an unpenalized intercept; decision_function returns the raw
    regression scores (combine with a sigmoid for probability-like values).
    """

    def __init__(self, alpha: float = 1.0, class_weight="balanced"):
        self.alpha = float(alpha)
        self.class_weight = class_weight
        self.W_ = None   # (D, C)
        self.b_ = None   # (C,)

    @staticmethod
    def _solve(Grams, RHS, D, alpha):
        P = np.eye(D + 1)
        P[D, D] = 0.0  # do not penalize the intercept
        W_aug = np.column_stack([np.linalg.solve(G + alpha * P, r)
                                 for G, r in zip(Grams, RHS)])  # (D+1, C)
        return W_aug[:D].astype(np.float64), W_aug[D].astype(np.float64)

    @classmethod
    def from_grams(cls, Grams, RHS, D, alpha, class_weight="balanced"):
        obj = cls(alpha=alpha, class_weight=class_weight)
        obj.W_, obj.b_ = cls._solve(Grams, RHS, D, alpha)
        return obj

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ClosedFormRidge":
        Grams, RHS, D = ridge_grams(X, Y, self.class_weight)
        self.W_, self.b_ = self._solve(Grams, RHS, D, self.alpha)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) @ self.W_ + self.b_)


# --------------------------------------------------------------------------- #
# Model scoring helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def model_scores(model, X: np.ndarray) -> np.ndarray:
    """Return (N, 15) probability-like scores in [0, 1] for any model type.

    - RidgeClassifier (no predict_proba): sigmoid(decision_function).
    - Logistic/SGD log-loss OvR: predict_proba -> (N, 15).
    - RandomForest (native multioutput): predict_proba -> list of (N, 2).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):  # native multioutput RF
            cols = []
            for p in proba:
                if p.ndim == 1:
                    cols.append(p)
                elif p.shape[1] == 1:
                    cols.append(np.zeros(p.shape[0], dtype=np.float32))
                else:
                    cols.append(p[:, 1])
            return np.vstack(cols).T.astype(np.float32)
        return np.asarray(proba, dtype=np.float32)
    scores = model.decision_function(X)
    return sigmoid(scores).astype(np.float32)


# --------------------------------------------------------------------------- #
# SED inference: per-second predictions -> onset/offset intervals
# --------------------------------------------------------------------------- #
def median_filter_binary(binary: np.ndarray, window: int) -> np.ndarray:
    """Per-class temporal median filter over a (T, C) binary matrix.

    window == 1 is a no-op (baseline). Odd windows >1 remove isolated
    false-positive blips and fill 1-frame gaps. Edges use reflection.
    """
    if window <= 1:
        return binary
    from scipy.ndimage import median_filter
    out = np.empty_like(binary)
    for c in range(binary.shape[1]):
        out[:, c] = median_filter(binary[:, c], size=window, mode="nearest")
    return out


def predictions_to_intervals(
    predictions: np.ndarray, start_times: np.ndarray, filename: str
) -> List[Dict]:
    """Merge consecutive active whole-second segments into intervals.

    Identical logic to the baseline notebook's predictions_to_intervals.
    """
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_preds = predictions[:, cls_idx]
        in_event = False
        onset = None
        for t, pred in zip(start_times, cls_preds):
            if pred == 1 and not in_event:
                onset = float(t)
                in_event = True
            elif pred == 0 and in_event:
                rows.append({"filename": filename, "annotation": cls_name,
                             "onset": onset, "offset": float(t)})
                in_event = False
        if in_event:
            rows.append({"filename": filename, "annotation": cls_name,
                         "onset": onset,
                         "offset": float(start_times[-1]) + SEGMENT_LENGTH})
    return rows


def generate_predictions(
    probs: np.ndarray,
    starts: np.ndarray,
    index: List[list],
    threshold: float,
    median_window: int = 1,
) -> pd.DataFrame:
    """Convert cached per-segment probabilities into an event-interval DataFrame.

    `probs`/`starts` are the full cached arrays for a split; `index` selects the
    files (and their row ranges) to predict on. Only whole-second segments are
    kept (matching the baseline), then thresholded, optionally median-filtered,
    and merged into intervals.
    """
    all_rows = []
    for wav, start, count in index:
        p = probs[start:start + count]
        st = starts[start:start + count]
        whole_second_mask = np.isclose(st % 1.0, 0.0)
        binary = (p[whole_second_mask] >= threshold).astype(int)
        binary = median_filter_binary(binary, median_window)
        all_rows.extend(predictions_to_intervals(binary, st[whole_second_mask], wav))
    cols = ["filename", "annotation", "onset", "offset"]
    if not all_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(all_rows)[cols]


# --------------------------------------------------------------------------- #
# Evaluation via the official evaluate.py
# --------------------------------------------------------------------------- #
_ANN_CACHE: Dict[str, pd.DataFrame] = {}


def load_annotations(split: str) -> pd.DataFrame:
    """Load (and cache) the raw annotations.csv for 'train' or 'validation'."""
    if split not in _ANN_CACHE:
        path = (PATH_VAL if split == "validation" else PATH_TRAIN) / "annotations.csv"
        _ANN_CACHE[split] = pd.read_csv(path)
    return _ANN_CACHE[split]


def build_gt_segments(wav_names: set, ann_df: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote aggregate the GT for `wav_names` and build its 1-second
    segment frame. Compute once per split and reuse across configs/thresholds."""
    ann_split = ann_df[ann_df["filename"].isin(wav_names)].copy()
    gt = EVAL.aggregate_ground_truth_annotations(ann_split)
    return EVAL.build_segment_frame_from_intervals(gt, name="ground_truth")


def evaluate_with_gt(
    pred_df: pd.DataFrame, wav_names: set, gt_segments: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Segment-based Macro F1 against a *precomputed* GT segment frame."""
    pred_segments = EVAL.build_segment_frame_from_intervals(pred_df, name="predictions")
    if len(pred_segments) > 0:
        pred_files = pred_segments.index.get_level_values("filename")
        pred_segments = pred_segments[pred_files.isin(wav_names)]
    return EVAL.calculate_f1_score(gt_segments, pred_segments)


def evaluate_against_annotations(
    pred_df: pd.DataFrame, wav_names: set, ann_df: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Segment-based Macro F1 via the official script (baseline cell 26 logic).

    Filters the raw annotations to `wav_names`, majority-vote aggregates the
    ground truth, builds 1-second segment frames for GT and predictions, and
    returns (macro_f1, per_class_table). Convenience wrapper around
    build_gt_segments + evaluate_with_gt for one-off evaluations.
    """
    gt_segments = build_gt_segments(wav_names, ann_df)
    return evaluate_with_gt(pred_df, wav_names, gt_segments)


def macro_f1_over_all_classes(per_class: pd.DataFrame) -> float:
    """Macro F1 averaged over all 15 classes (missing classes count as F1=0).

    The official calculate_f1_score averages only over classes present in the
    GT/pred union. For a stable, comparable headline number we also report the
    mean over the full fixed class list.
    """
    f1_by_class = dict(zip(per_class["annotation"], per_class["f1"]))
    return float(np.mean([f1_by_class.get(c, 0.0) for c in CLASS_NAMES]))
