"""Phase 3: classical (non-deep, except MLP-allowed) classifiers for SED.

Starts from the Task 4 best idea -- one-vs-rest linear ridge (alpha=2.0,
threshold=0.50) -- and adapts it to the Task 5 SED pipeline, then tunes:

  A. Ridge (closed-form, primary):
        alpha in {0.1,0.5,1,2,5,10} x threshold in {0,0.25,0.5,0.75}
  B. Logistic regression (liblinear OvR, subsampled):
        C in {0.1,1.0} x class_weight {None,balanced} x threshold {0.2,0.3,0.4,0.5}
  C. RandomForest (optional, subsampled, n_jobs=1):
        n_estimators=100 x max_depth {20,None} x threshold {0.3,0.5}

Features are standardized on TRAIN ONLY (no imputation needed -- audited zero
NaNs). Ridge is solved with numpy normal equations because sklearn's iterative
solvers / joblib process-parallelism hang on this Accelerate-BLAS arm64 box.
Tuning uses the development-validation split; the single best config is then
evaluated once on the non-hidden test split and persisted for later stages.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import common as C

# sklearn's iterative liblinear is very slow on this Accelerate-BLAS arm64 box,
# so logistic is fit on a small subsample (it appears in the comparison table;
# the closed-form ridge is the primary, fully-tuned model). RF uses trees (no
# BLAS) and is reliable with n_jobs=1.
RUN_RF = True
RF_SUBSAMPLE = 40_000
LOGREG_SUBSAMPLE = 12_000


def build_model(name: str, params: dict):
    """Factory used here and by Phase 6 (final retrain). All n_jobs=1 (no loky)."""
    if name == "ridge":
        return C.ClosedFormRidge(alpha=params["alpha"], class_weight="balanced")
    if name == "logistic":
        # dual=False (primal) is much faster than the liblinear default when
        # n_samples >> n_features (40k >> 960).
        return OneVsRestClassifier(
            LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                               solver="liblinear", dual=False, max_iter=200,
                               random_state=C.SEED),
            n_jobs=1)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=C.SEED, n_jobs=1)
    raise ValueError(name)


def main() -> None:
    t0 = time.time()
    print("=== Phase 3: classical classifiers ===", flush=True)

    X_train, Y_train, _, _ = C.load_split_cache("train", with_labels=True)
    Xv, _, starts_v, index_v = C.load_split_cache("validation", with_labels=True)

    std = C.Standardizer().fit(X_train)
    Xtr = std.transform(X_train)
    Xval = std.transform(Xv)
    del X_train, Xv
    print(f"Standardized on train only. Xtr={Xtr.shape}, Xval={Xval.shape} "
          f"({time.time()-t0:.1f}s)", flush=True)

    split = json.loads((C.CACHE_DIR / "validation_split.json").read_text())
    dev_set, nht_set = set(split["dev_val_wav"]), set(split["nonhidden_test_wav"])
    dev_idx = C.subset_index(index_v, dev_set)
    nht_idx = C.subset_index(index_v, nht_set)
    ann = C.load_annotations("validation")
    dev_gt = C.build_gt_segments(dev_set, ann)
    nht_gt = C.build_gt_segments(nht_set, ann)
    print(f"Precomputed GT segment frames ({time.time()-t0:.1f}s)", flush=True)

    rng = np.random.default_rng(C.SEED)
    rows = []
    best = {"dev_macro": -1.0}

    def consider(name, params, probs_dev, thresholds, fitted):
        nonlocal best
        pstr = ",".join(f"{k}={v}" for k, v in params.items())
        for thr in thresholds:
            pred_df = C.generate_predictions(probs_dev, starts_v, dev_idx, thr, median_window=1)
            macro, per_class = C.evaluate_with_gt(pred_df, dev_set, dev_gt)
            rows.append({"model": name, "params": pstr, "threshold": thr,
                         "dev_macro_f1": round(macro, 6),
                         "dev_macro_f1_all15": round(C.macro_f1_over_all_classes(per_class), 6)})
            if macro > best["dev_macro"]:
                best = {"dev_macro": macro, "model_name": name, "params": dict(params),
                        "threshold": thr, "fitted": fitted}

    # ---- A. Ridge (closed form): Grams computed ONCE, reused across alphas - #
    tg = time.time()
    Grams, RHS, D = C.ridge_grams(Xtr, Y_train, class_weight="balanced")
    print(f"  ridge: built weighted Gram matrices in {time.time()-tg:.1f}s", flush=True)
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = C.ClosedFormRidge.from_grams(Grams, RHS, D, alpha, class_weight="balanced")
        probs_dev = C.model_scores(model, Xval)
        consider("ridge", {"alpha": alpha}, probs_dev, [0.0, 0.25, 0.5, 0.75], model)
        bdev = max(r["dev_macro_f1"] for r in rows if r["model"] == "ridge" and r["params"] == f"alpha={alpha}")
        print(f"    alpha={alpha:<5} best-thr dev Macro F1={bdev:.4f}", flush=True)
    del Grams, RHS

    # ---- B. Logistic regression (liblinear OvR, subsampled) --------------- #
    sub = rng.choice(Xtr.shape[0], size=min(LOGREG_SUBSAMPLE, Xtr.shape[0]), replace=False)
    for Cval in [1.0]:
        for cw in [None, "balanced"]:
            tc = time.time()
            model = build_model("logistic", {"C": Cval, "class_weight": cw})
            model.fit(Xtr[sub], Y_train[sub])
            probs_dev = C.model_scores(model, Xval)
            consider("logistic", {"C": Cval, "class_weight": cw}, probs_dev,
                     [0.2, 0.3, 0.4, 0.5], model)
            print(f"  logistic C={Cval} cw={cw}: fit+eval {time.time()-tc:.1f}s", flush=True)

    # ---- C. RandomForest (optional, subsampled, n_jobs=1) ----------------- #
    if RUN_RF:
        rsub = rng.choice(Xtr.shape[0], size=min(RF_SUBSAMPLE, Xtr.shape[0]), replace=False)
        for md in [20, None]:
            tc = time.time()
            model = build_model("random_forest", {"n_estimators": 80, "max_depth": md})
            model.fit(Xtr[rsub], Y_train[rsub])
            probs_dev = C.model_scores(model, Xval)
            consider("random_forest", {"n_estimators": 100, "max_depth": md},
                     probs_dev, [0.3, 0.5], model)
            print(f"  random_forest depth={md}: fit+eval {time.time()-tc:.1f}s", flush=True)

    tuning = pd.DataFrame(rows).sort_values("dev_macro_f1", ascending=False).reset_index(drop=True)
    tuning.to_csv(C.RESULTS_DIR / "classifier_tuning_results.csv", index=False)

    # ---- Selected best on non-hidden test (evaluated once) ---------------- #
    bn, bp, bt = best["model_name"], best["params"], best["threshold"]
    bp_str = ",".join(f"{k}={v}" for k, v in bp.items())
    print(f"\nBest classical: {bn} ({bp_str}), threshold={bt}, "
          f"dev Macro F1={best['dev_macro']:.4f}", flush=True)

    probs_nht = C.model_scores(best["fitted"], Xval)
    pred_nht = C.generate_predictions(probs_nht, starts_v, nht_idx, bt, median_window=1)
    macro_nht, per_class_nht = C.evaluate_with_gt(pred_nht, nht_set, nht_gt)
    macro_nht_all15 = C.macro_f1_over_all_classes(per_class_nht)
    print(f"Non-hidden test Macro F1: {macro_nht:.4f} (all-15: {macro_nht_all15:.4f})", flush=True)

    overall = {
        "model": bn, "params": bp_str, "threshold": bt,
        "dev_val_macro_f1": round(best["dev_macro"], 6),
        "nonhidden_test_macro_f1_official": round(macro_nht, 6),
        "nonhidden_test_macro_f1_all15": round(macro_nht_all15, 6),
        "n_intervals_nonhidden": int(len(pred_nht)),
        "preprocessing": "z-score standardize (fit on train only); no imputation (0 NaNs)",
        "trained_on": ("full train split" if bn == "ridge"
                       else f"train subsample ({LOGREG_SUBSAMPLE if bn=='logistic' else RF_SUBSAMPLE})"),
    }
    (C.RESULTS_DIR / "best_classical_overall.json").write_text(json.dumps(overall, indent=2))
    per_class_nht.to_csv(C.RESULTS_DIR / "best_classical_per_class.csv", index=False)
    pred_nht.to_csv(C.RESULTS_DIR / "best_classical_predictions_nonhidden.csv", index=False)

    (C.CACHE_DIR / "models").mkdir(exist_ok=True)
    joblib.dump(std, C.CACHE_DIR / "models" / "standardizer.joblib")
    joblib.dump(best["fitted"], C.CACHE_DIR / "models" / "best_classical.joblib")
    (C.CACHE_DIR / "models" / "best_classical_meta.json").write_text(
        json.dumps({"model_name": bn, "params": bp, "threshold": bt}, indent=2))

    # ---- Markdown tuning summary ------------------------------------------ #
    lines = ["# Classifier tuning summary (development validation)", "",
             f"- Configs evaluated: {len(tuning)} (model x params x threshold)",
             "- Selection metric: segment-based Macro F1 on the development validation split",
             f"- **Best: {bn} ({bp_str}), threshold={bt} -> dev Macro F1 {best['dev_macro']:.4f}**",
             f"- Best on non-hidden test (evaluated once): Macro F1 {macro_nht:.4f}", "",
             "## Top configurations", "",
             "| model | params | threshold | dev Macro F1 |", "|---|---|---|---|"]
    for _, r in tuning.head(12).iterrows():
        lines.append(f"| {r['model']} | {r['params']} | {r['threshold']} | {r['dev_macro_f1']:.4f} |")
    lines += ["", "## Per-model best (dev Macro F1)", "", "| model | best dev Macro F1 |", "|---|---|"]
    for m in tuning["model"].unique():
        lines.append(f"| {m} | {tuning[tuning['model']==m]['dev_macro_f1'].max():.4f} |")
    (C.RESULTS_DIR / "classifier_tuning_summary.md").write_text("\n".join(lines))

    print(f"\nTotal runtime {time.time()-t0:.1f}s")
    print("Saved tuning results, best_classical_*, standardizer + model artifacts.")


if __name__ == "__main__":
    main()
