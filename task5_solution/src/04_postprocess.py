"""Phase 4: temporal post-processing via per-class median filtering.

Takes the selected best classical model from Phase 3 and applies a median
filter over its per-second binary predictions. Window 1 == no post-processing
(the comparison baseline). The window is selected on development validation,
then the chosen window is evaluated once on the non-hidden test split.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

import common as C

WINDOWS = [1, 3, 5]


def main() -> None:
    t0 = time.time()
    print("=== Phase 4: median-filter post-processing ===", flush=True)

    std = joblib.load(C.CACHE_DIR / "models" / "standardizer.joblib")
    model = joblib.load(C.CACHE_DIR / "models" / "best_classical.joblib")
    meta = json.loads((C.CACHE_DIR / "models" / "best_classical_meta.json").read_text())
    thr = meta["threshold"]
    print(f"Best classical: {meta['model_name']} {meta['params']} threshold={thr}", flush=True)

    Xv, _, starts_v, index_v = C.load_split_cache("validation", with_labels=True)
    Xval = std.transform(Xv)
    probs = C.model_scores(model, Xval)

    split = json.loads((C.CACHE_DIR / "validation_split.json").read_text())
    dev_set, nht_set = set(split["dev_val_wav"]), set(split["nonhidden_test_wav"])
    dev_idx = C.subset_index(index_v, dev_set)
    nht_idx = C.subset_index(index_v, nht_set)
    ann = C.load_annotations("validation")

    # --- Parameter study on development validation ------------------------- #
    rows = []
    best = {"dev_macro": -1.0}
    for w in WINDOWS:
        pred_df = C.generate_predictions(probs, starts_v, dev_idx, thr, median_window=w)
        macro, per_class = C.evaluate_against_annotations(pred_df, dev_set, ann)
        macro_all15 = C.macro_f1_over_all_classes(per_class)
        rows.append({"median_window": w, "dev_macro_f1": round(macro, 6),
                     "dev_macro_f1_all15": round(macro_all15, 6),
                     "note": "no post-processing" if w == 1 else "median filter"})
        print(f"  window={w}: dev Macro F1 {macro:.4f}", flush=True)
        if macro > best["dev_macro"]:
            best = {"dev_macro": macro, "window": w}
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "postprocessing_results.csv", index=False)

    best_w = best["window"]
    print(f"\nSelected window={best_w} (dev Macro F1 {best['dev_macro']:.4f})", flush=True)

    # --- Evaluate selected window once on non-hidden test ------------------ #
    pred_nht = C.generate_predictions(probs, starts_v, nht_idx, thr, median_window=best_w)
    macro_nht, per_class_nht = C.evaluate_against_annotations(pred_nht, nht_set, ann)
    macro_nht_all15 = C.macro_f1_over_all_classes(per_class_nht)

    # No-post-processing reference on non-hidden test (window 1) for the report.
    pred_nht_w1 = C.generate_predictions(probs, starts_v, nht_idx, thr, median_window=1)
    macro_nht_w1, _ = C.evaluate_against_annotations(pred_nht_w1, nht_set, ann)

    print(f"Non-hidden test Macro F1: window1={macro_nht_w1:.4f} -> "
          f"window{best_w}={macro_nht:.4f}", flush=True)

    overall = {
        "model": meta["model_name"], "params": meta["params"], "threshold": thr,
        "post_processing": "per-class median filter",
        "selected_window": best_w,
        "dev_val_macro_f1": round(best["dev_macro"], 6),
        "nonhidden_test_macro_f1_no_postproc": round(macro_nht_w1, 6),
        "nonhidden_test_macro_f1_official": round(macro_nht, 6),
        "nonhidden_test_macro_f1_all15": round(macro_nht_all15, 6),
        "n_intervals_nonhidden": int(len(pred_nht)),
    }
    (C.RESULTS_DIR / "best_postprocessed_overall.json").write_text(json.dumps(overall, indent=2))
    per_class_nht.to_csv(C.RESULTS_DIR / "best_postprocessed_per_class.csv", index=False)
    pred_nht.to_csv(C.RESULTS_DIR / "best_postprocessed_predictions_nonhidden.csv", index=False)

    meta["median_window"] = int(best_w)
    (C.CACHE_DIR / "models" / "best_classical_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nTotal runtime {time.time()-t0:.1f}s")
    print("Saved postprocessing_results.csv + best_postprocessed_*.")


if __name__ == "__main__":
    main()
