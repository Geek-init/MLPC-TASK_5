"""Phase 0 audit: confirm the Task 5 dataset structure and key facts.

Writes a human-readable audit to results/dataset_audit.txt and a machine
summary to results/dataset_audit.json. Does not touch heavy matrices.
"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    emit("=" * 70)
    emit("MLPC 2026 Task 5 — Dataset Audit")
    emit("=" * 70)
    emit(f"Dataset dir : {C.DATASET_DIR}")
    emit(f"Baseline dir: {C.BASELINE_DIR}")
    emit("")

    # Split file counts ----------------------------------------------------- #
    train_files = C.list_npz(C.PATH_TRAIN)
    val_files = C.list_npz(C.PATH_VAL)
    test_files = C.list_npz(C.PATH_TEST)
    emit(f"train      : {len(train_files):>5} npz   "
         f"metadata={'yes' if (C.PATH_TRAIN/'metadata.csv').exists() else 'NO'}  "
         f"annotations={'yes' if (C.PATH_TRAIN/'annotations.csv').exists() else 'NO'}")
    emit(f"validation : {len(val_files):>5} npz   "
         f"metadata={'yes' if (C.PATH_VAL/'metadata.csv').exists() else 'NO'}  "
         f"annotations={'yes' if (C.PATH_VAL/'annotations.csv').exists() else 'NO'}")
    emit(f"test       : {len(test_files):>5} npz   "
         f"metadata={'yes' if (C.PATH_TEST/'metadata.csv').exists() else 'NO'}  "
         f"annotations={'yes' if (C.PATH_TEST/'annotations.csv').exists() else 'NO'}  (hidden)")
    emit("")

    # Confirm hidden test has no labels ------------------------------------- #
    test_sample = dict(np.load(test_files[0], allow_pickle=True))
    test_has_labels = "annotations" in test_sample
    emit(f"Hidden test 'annotations' key present: {test_has_labels}  "
         f"(expected False -> no labels)")
    emit("")

    # Feature dimensionality & segment layout ------------------------------- #
    train_sample = dict(np.load(train_files[0], allow_pickle=True))
    X = C.build_feature_matrix(train_sample)
    Y = C.get_segment_labels(train_sample)
    starts = np.asarray(train_sample["start_time"], dtype=float)
    emit(f"Example train file : {os.path.basename(train_files[0])}")
    emit(f"  segments         : {X.shape[0]}")
    emit(f"  feature dim      : {X.shape[1]}  (expected 960)")
    emit(f"  label classes    : {Y.shape[1]}  (expected 15)")
    emit(f"  annotators (A)   : {train_sample['annotations'].shape[2]}")
    emit(f"  start_time[:6]   : {starts[:6]}  (hop = {C.HOP_SIZE}s)")
    emit("")

    # Baseline-faithful split ----------------------------------------------- #
    dev_val, nonhidden = C.validation_split()
    emit(f"Validation split (seed {C.SEED}):")
    emit(f"  development validation : {len(dev_val)} files")
    emit(f"  non-hidden test        : {len(nonhidden)} files")
    emit(f"  first dev-val file     : {C.to_wav(dev_val[0])}")
    emit(f"  first non-hidden file  : {C.to_wav(nonhidden[0])}")
    emit("")

    # Class distribution over the full training set (annotations.csv) ------- #
    ann = C.load_annotations("train") if (C.PATH_TRAIN / "annotations.csv").exists() else None
    emit("Annotation rows per class (train annotations.csv):")
    class_counts = {}
    if ann is not None:
        vc = ann["annotation"].value_counts()
        for cls in C.CLASS_NAMES:
            class_counts[cls] = int(vc.get(cls, 0))
            emit(f"  {cls:<28} {class_counts[cls]:>6}")
    emit("")
    emit("Audit complete.")

    # Persist --------------------------------------------------------------- #
    (C.RESULTS_DIR / "dataset_audit.txt").write_text("\n".join(lines))
    summary = {
        "n_train": len(train_files),
        "n_validation": len(val_files),
        "n_test_hidden": len(test_files),
        "hidden_test_has_labels": bool(test_has_labels),
        "feature_dim": int(X.shape[1]),
        "n_classes": int(Y.shape[1]),
        "seed": C.SEED,
        "n_dev_val": len(dev_val),
        "n_nonhidden_test": len(nonhidden),
        "class_names": C.CLASS_NAMES,
        "feature_names": C.FEATURE_NAMES,
        "segment_length": C.SEGMENT_LENGTH,
        "hop_size": C.HOP_SIZE,
        "train_class_annotation_counts": class_counts,
    }
    (C.RESULTS_DIR / "dataset_audit.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {C.RESULTS_DIR/'dataset_audit.txt'}")
    print(f"Saved: {C.RESULTS_DIR/'dataset_audit.json'}")


if __name__ == "__main__":
    main()
