"""Phase 5: qualitative error analysis on non-hidden test recordings.

Selects three illustrative files (a success case, a false-positive / noisy case,
and a missed-event / timing case), then renders for each a stacked figure:

  1. precomputed mel feature representation (segment means; NOT raw waveform)
  2. ground truth (whole-second, majority vote over annotators)
  3. baseline decision-tree predictions
  4. best classical model predictions
  5. best classical + median-filter post-processed predictions

Ground truth and per-second predictions come from the cached aligned arrays;
the mel heatmap is read from the .npz (we have no raw .wav). A notes file
summarises correct detections, misses, false detections, timing shifts and
likely reasons.
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


def whole_second(starts):
    return np.isclose(starts % 1.0, 0.0)


def per_class_counts(pred_ws, gt_ws):
    tp = ((pred_ws == 1) & (gt_ws == 1)).sum(axis=0)
    fp = ((pred_ws == 1) & (gt_ws == 0)).sum(axis=0)
    fn = ((pred_ws == 0) & (gt_ws == 1)).sum(axis=0)
    return tp, fp, fn


def main() -> None:
    t0 = time.time()
    print("=== Phase 5: qualitative error analysis ===", flush=True)

    std = joblib.load(C.CACHE_DIR / "models" / "standardizer.joblib")
    model = joblib.load(C.CACHE_DIR / "models" / "best_classical.joblib")
    baseline = joblib.load(C.CACHE_DIR / "models" / "baseline_dt.joblib")
    meta = json.loads((C.CACHE_DIR / "models" / "best_classical_meta.json").read_text())
    thr, window = meta["threshold"], meta.get("median_window", 1)

    Xv, Yv, starts_v, index_v = C.load_split_cache("validation", with_labels=True)
    Xval_t = std.transform(Xv)

    probs_cls = C.model_scores(model, Xval_t)            # classical, scaled features
    preds_bl = baseline.predict(Xv).astype(int)          # baseline, raw features
    npz_paths = {C.to_wav(p): p for p in C.list_npz(C.PATH_VAL)}

    split = json.loads((C.CACHE_DIR / "validation_split.json").read_text())
    nht_set = set(split["nonhidden_test_wav"])
    nht_index = C.subset_index(index_v, nht_set)

    # --- Score every non-hidden file at whole-second resolution ------------ #
    records = []
    cache = {}
    for wav, start, count in nht_index:
        st = starts_v[start:start + count]
        ws = whole_second(st)
        gt = Yv[start:start + count][ws]
        cls = (probs_cls[start:start + count][ws] >= thr).astype(int)
        clsp = C.median_filter_binary(cls, window)
        bl = preds_bl[start:start + count][ws]
        tp, fp, fn = per_class_counts(clsp, gt)
        TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        records.append({"wav": wav, "support": int(gt.sum()), "TP": TP, "FP": FP,
                        "FN": FN, "f1": f1, "recall": recall, "n_ws": int(ws.sum())})
        cache[wav] = {"start": start, "count": count, "ws": ws, "gt": gt,
                      "cls": cls, "clsp": clsp, "bl": bl, "st": st[ws]}
    rec = pd.DataFrame(records)

    # --- Select three distinct, illustrative files ------------------------- #
    chosen, used = {}, set()

    def pick(df):
        for w in df["wav"]:
            if w not in used:
                used.add(w)
                return w
        return None

    success = pick(rec[(rec.support >= 10) & (rec.n_ws >= 8)].sort_values("f1", ascending=False))
    fp_noisy = pick(rec[(rec.FP >= 5)].assign(ratio=rec.FP / (rec.TP + 1)).sort_values("ratio", ascending=False))
    missed = pick(rec[(rec.support >= 10) & (rec.recall <= 0.4)].sort_values("FN", ascending=False))
    # fall backs in case a category is empty
    for label, w in [("success", success), ("false_positive", fp_noisy), ("missed", missed)]:
        if w is None:
            w = pick(rec.sort_values("support", ascending=False))
        chosen[label] = w
    print("Selected:", chosen, flush=True)

    # --- Render figures + collect notes ------------------------------------ #
    notes = ["# Qualitative error analysis notes", "",
             f"Model: {meta['model_name']} {meta['params']}, threshold={thr}, "
             f"median_window={window}.",
             "Ground truth = whole-second majority vote over annotators (from the "
             "aligned .npz annotation tensor). Visual is a *precomputed mel feature "
             "representation* (segment means), not a raw waveform (no .wav available).",
             ""]
    case_files = [("success", "error_case_1.png"),
                  ("false_positive", "error_case_2.png"),
                  ("missed", "error_case_3.png")]

    for (label, fig_name) in case_files:
        wav = chosen[label]
        c = cache[wav]
        data = dict(np.load(npz_paths[wav], allow_pickle=True))
        starts_all = np.asarray(data["start_time"], float)
        mel = np.log1p(data["melspect_mean"].T)             # (128, N_all)
        gt, bl, cls, clsp, stw = c["gt"], c["bl"], c["cls"], c["clsp"], c["st"]
        Cn = len(C.CLASS_NAMES)

        t_edges_all = np.append(starts_all, starts_all[-1] + C.HOP_SIZE)
        t_edges_ws = np.append(stw, float(stw[-1]) + C.SEGMENT_LENGTH)
        mel_edges = np.arange(mel.shape[0] + 1) - 0.5
        cls_edges = np.arange(Cn + 1) - 0.5

        fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.4, 1.4]})
        fig.suptitle(f"Error case ({label}): {wav}", fontsize=13, fontweight="bold")

        pcm = axes[0].pcolormesh(t_edges_all, mel_edges, mel, cmap="magma", shading="flat")
        fig.colorbar(pcm, ax=axes[0], label="log(1+mean energy)", pad=0.01)
        axes[0].set_ylabel("Mel bin")
        axes[0].set_title("Precomputed mel feature representation (segment means, hop=0.5s)")

        for ax, mat, title, cmap in [
            (axes[1], gt, "Ground truth (majority vote)", "Greens"),
            (axes[2], bl, "Baseline (decision trees)", "Blues"),
            (axes[3], cls, "Best classical model", "Oranges"),
            (axes[4], clsp, f"Best classical + median filter (w={window})", "Purples"),
        ]:
            ax.pcolormesh(t_edges_ws, cls_edges, mat.T, cmap=cmap, vmin=0, vmax=1, shading="flat")
            ax.set_yticks(range(Cn))
            ax.set_yticklabels(C.CLASS_NAMES, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_ylabel("Class")
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        fig.savefig(C.FIGURES_DIR / fig_name, dpi=150)
        plt.close(fig)

        # textual notes from post-processed predictions vs GT
        tp, fp, fn = per_class_counts(clsp, gt)
        correct = [C.CLASS_NAMES[i] for i in range(Cn) if tp[i] > 0]
        missed_cl = [C.CLASS_NAMES[i] for i in range(Cn) if tp[i] == 0 and fn[i] > 0]
        false_cl = [C.CLASS_NAMES[i] for i in range(Cn) if tp[i] == 0 and fp[i] > 0]
        timing = []
        for i in range(Cn):
            if tp[i] > 0:
                g = np.where(gt[:, i] == 1)[0]
                p = np.where(clsp[:, i] == 1)[0]
                if len(g) and len(p):
                    do = (p[0] - g[0])
                    ddur = (len(p) - len(g))
                    if abs(do) >= 1 or abs(ddur) >= 2:
                        timing.append(f"{C.CLASS_NAMES[i]} (onset {'+' if do>=0 else ''}{do}s, "
                                      f"duration {'+' if ddur>=0 else ''}{ddur}s)")

        notes += [f"## {label} — {wav}  (figure: {fig_name})",
                  f"- Segment support (GT positives): {int(gt.sum())} over {gt.shape[0]} seconds",
                  f"- Correctly detected classes: {', '.join(correct) if correct else 'none'}",
                  f"- Missed classes (present in GT, not detected): {', '.join(missed_cl) if missed_cl else 'none'}",
                  f"- False detections (predicted, absent in GT): {', '.join(false_cl) if false_cl else 'none'}",
                  f"- Timing shifts: {'; '.join(timing) if timing else 'minor / none'}",
                  f"- Likely reasons: " + likely_reason(label, correct, missed_cl, false_cl),
                  ""]

    (C.RESULTS_DIR / "error_analysis_notes.md").write_text("\n".join(notes))
    rec.sort_values("f1", ascending=False).to_csv(C.RESULTS_DIR / "per_file_diagnostics.csv", index=False)
    print(f"Saved 3 figures + error_analysis_notes.md ({time.time()-t0:.1f}s)")


def likely_reason(label, correct, missed_cl, false_cl):
    bits = []
    if label == "success":
        bits.append("loud, sustained sources (e.g. running water, vacuum, microwave) give "
                    "stable spectral features that linear scores separate well")
    if missed_cl:
        bits.append(f"missed classes are often short or rare ({', '.join(missed_cl[:3])}); "
                    "1-second resolution and class imbalance hurt recall")
    if false_cl:
        bits.append(f"false detections ({', '.join(false_cl[:3])}) come from acoustically "
                    "similar transients and the balanced class weights raising recall at the "
                    "cost of precision")
    if not bits:
        bits.append("boundary disagreement at 1-second resolution and annotator label noise")
    return "; ".join(bits) + "."


if __name__ == "__main__":
    main()
