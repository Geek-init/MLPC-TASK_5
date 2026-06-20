"""Phase 6: final hidden-test predictions.

Honest reporting (non-hidden test numbers) comes from the strict
train / dev-val / non-hidden-test protocol in Phases 2-4. For the *submission*
CSV, the selected configuration (model + threshold + median window) is retrained
on train + the full provided validation labels, because the hidden test split is
entirely separate and has no labels. This is recorded in the README and report.

Output: submission/predictions_hidden_test.csv with columns
filename,annotation,onset,offset (.wav filenames, no index column).
"""
from __future__ import annotations

import json
import time

import numpy as np

import common as C
from importlib import import_module

# reuse the model factory from the Phase 3 script (filename starts with a digit)
build_model = import_module("03_run_classical_models").build_model


def main() -> None:
    t0 = time.time()
    print("=== Phase 6: final hidden-test predictions ===", flush=True)

    meta = json.loads((C.CACHE_DIR / "models" / "best_classical_meta.json").read_text())
    name, params, thr = meta["model_name"], meta["params"], meta["threshold"]
    window = meta.get("median_window", 1)
    print(f"Final config: {name} {params} threshold={thr} median_window={window}", flush=True)

    # --- Final training set: train + full validation ---------------------- #
    Xtr, Ytr, _, _ = C.load_split_cache("train", with_labels=True)
    Xva, Yva, _, _ = C.load_split_cache("validation", with_labels=True)
    X_full = np.vstack([Xtr, Xva])
    Y_full = np.vstack([Ytr, Yva])
    del Xtr, Xva
    print(f"Final training set (train + full validation): X={X_full.shape}", flush=True)

    std = C.Standardizer().fit(X_full)
    X_full_t = std.transform(X_full)
    del X_full

    # Ridge (closed form) scales to all of train+val cheaply. liblinear logistic
    # is very slow on this Accelerate-BLAS box, so its final fit uses a capped
    # subsample of train+val (documented in the report). RF likewise capped.
    fit_X, fit_Y = X_full_t, Y_full
    # liblinear logistic does not scale on this Accelerate-BLAS box, so its final
    # fit uses a subsample matching the size used during model selection (12k).
    FINAL_CAP = {"logistic": 12_000, "random_forest": 60_000}
    if name in FINAL_CAP and X_full_t.shape[0] > FINAL_CAP[name]:
        s = np.random.default_rng(C.SEED).choice(X_full_t.shape[0], FINAL_CAP[name], replace=False)
        fit_X, fit_Y = X_full_t[s], Y_full[s]
        print(f"NOTE: final {name} fit on a {FINAL_CAP[name]}-segment subsample of "
              f"train+val (matches selection-time training size; solver is slow here).", flush=True)

    model = build_model(name, params)
    model.fit(fit_X, fit_Y)
    print(f"Trained final model in {time.time()-t0:.1f}s", flush=True)

    # --- Predict on every hidden test file -------------------------------- #
    Xte, _, starts_te, index_te = C.load_split_cache("test", with_labels=False)
    Xte_t = std.transform(Xte)
    probs = C.model_scores(model, Xte_t)
    pred_df = C.generate_predictions(probs, starts_te, index_te, thr, median_window=window)

    out = C.SUBMISSION_DIR / "predictions_hidden_test.csv"
    pred_df.to_csv(out, index=False)

    n_files_total = len(index_te)
    files_with_pred = pred_df["filename"].nunique() if len(pred_df) else 0
    print(f"Hidden test files            : {n_files_total}")
    print(f"Files with >=1 prediction    : {files_with_pred}")
    print(f"Files with no prediction     : {n_files_total - files_with_pred}")
    print(f"Total predicted intervals    : {len(pred_df)}")
    print(f"Saved: {out}")
    print(f"Total runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
