"""Phase 2: reproduce the provided decision-tree SED baseline.

Faithful to challenge_baseline.ipynb:
  * same seed-42 rng sequence (validation permutation, then 50k subsample),
  * DecisionTreeClassifier(max_depth=20, max_features='sqrt', random_state=42)
    wrapped in MultiOutputClassifier, trained on raw (unscaled) features,
  * SED inference on whole-second segments, merged into intervals,
  * evaluated on the non-hidden test split with the official evaluate.py.

Nothing is tuned. Outputs go to results/ and the fitted model to cache/models/.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

import common as C


def main() -> None:
    log: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    t0 = time.time()
    emit("=== Phase 2: Baseline reproduction (decision trees) ===")

    # --- Reproduce the baseline rng sequence exactly ----------------------- #
    # Notebook: rng=default_rng(42); rng.permutation(val_files) [cell 14];
    # then rng.choice(N, 50000, replace=False) [cell 18].
    rng = np.random.default_rng(seed=C.SEED)
    _ = rng.permutation(C.list_npz(C.PATH_VAL))      # consume state like the split does

    X_train, Y_train, _, _ = C.load_split_cache("train", with_labels=True)
    emit(f"Loaded train: X={X_train.shape}, Y={Y_train.shape}")

    n = X_train.shape[0]
    if n > C.BASELINE_MAX_TRAINING_SEGMENTS:
        idx = rng.choice(n, size=C.BASELINE_MAX_TRAINING_SEGMENTS, replace=False)
        X_train, Y_train = X_train[idx], Y_train[idx]
        emit(f"Subsampled to {X_train.shape[0]} segments (baseline default).")

    # --- Train the decision-tree baseline ---------------------------------- #
    emit("Training MultiOutputClassifier(DecisionTreeClassifier)...")
    base = DecisionTreeClassifier(max_depth=20, max_features="sqrt", random_state=C.SEED)
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X_train, Y_train)
    emit(f"Fitted {len(clf.estimators_)} per-class trees in {time.time() - t0:.1f}s.")

    (C.CACHE_DIR / "models").mkdir(exist_ok=True)
    joblib.dump(clf, C.CACHE_DIR / "models" / "baseline_dt.joblib")

    # --- Predictions + evaluation ------------------------------------------ #
    split = json.loads((C.CACHE_DIR / "validation_split.json").read_text())
    dev_set = set(split["dev_val_wav"])
    nht_set = set(split["nonhidden_test_wav"])

    Xv, _, starts_v, index_v = C.load_split_cache("validation", with_labels=True)
    ann = C.load_annotations("validation")

    # Binary per-segment predictions for the whole validation cache, sliced per file.
    preds_bin = clf.predict(Xv).astype(np.float32)   # (Nval, 15), values in {0,1}

    def eval_subset(wav_set):
        idx = C.subset_index(index_v, wav_set)
        pred_df = C.generate_predictions(preds_bin, starts_v, idx, threshold=0.5, median_window=1)
        macro, per_class = C.evaluate_against_annotations(pred_df, wav_set, ann)
        return pred_df, macro, per_class

    pred_dev, macro_dev, _ = eval_subset(dev_set)
    pred_nht, macro_nht, per_class_nht = eval_subset(nht_set)
    macro_nht_all15 = C.macro_f1_over_all_classes(per_class_nht)

    emit("")
    emit(f"Development validation Macro F1 : {macro_dev:.4f}")
    emit(f"Non-hidden test     Macro F1    : {macro_nht:.4f}   "
         f"(all-15-class mean: {macro_nht_all15:.4f})")
    emit(f"Non-hidden intervals predicted  : {len(pred_nht)} across "
         f"{pred_nht['filename'].nunique() if len(pred_nht) else 0} files")
    emit("")
    emit("Per-class (non-hidden test):")
    emit(per_class_nht.to_string(index=False))

    # --- Persist ----------------------------------------------------------- #
    overall = {
        "model": "decision_tree_baseline",
        "config": {"max_depth": 20, "max_features": "sqrt", "random_state": C.SEED,
                   "train_segments": int(X_train.shape[0])},
        "dev_val_macro_f1": round(macro_dev, 6),
        "nonhidden_test_macro_f1_official": round(macro_nht, 6),
        "nonhidden_test_macro_f1_all15": round(macro_nht_all15, 6),
        "n_intervals_nonhidden": int(len(pred_nht)),
    }
    (C.RESULTS_DIR / "baseline_overall.json").write_text(json.dumps(overall, indent=2))
    per_class_nht.to_csv(C.RESULTS_DIR / "baseline_per_class.csv", index=False)
    pred_nht.to_csv(C.RESULTS_DIR / "baseline_predictions_nonhidden.csv", index=False)

    emit("")
    emit(f"Total runtime {time.time() - t0:.1f}s")
    (C.RESULTS_DIR / "baseline_run_log.txt").write_text("\n".join(log))
    print(f"\nSaved baseline_overall.json / baseline_per_class.csv / "
          f"baseline_predictions_nonhidden.csv / baseline_run_log.txt")


if __name__ == "__main__":
    main()
