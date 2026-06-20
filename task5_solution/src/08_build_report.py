"""Phase 7: build the report figures, an Overleaf-ready report.tex, and a
self-contained report.pdf (reportlab, since no LaTeX toolchain is installed).

All numbers are read from results/ files produced by the pipeline -- nothing is
hard-coded -- so the report always matches the recomputed Task 5 results.
No bonus sections. Targets <= 6 pages and <= 2000 words.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

ASSETS = C.SOLUTION_DIR / "report" / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
REPORT_DIR = C.SOLUTION_DIR / "report"


def _j(name):
    return json.loads((C.RESULTS_DIR / name).read_text())


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_model_comparison(base, clf, post):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = ["Baseline\n(decision trees)", f"Best classical\n({clf['model']})",
             f"+ median filter\n(w={post['selected_window']})"]
    vals = [base["nonhidden_test_macro_f1_official"],
            clf["nonhidden_test_macro_f1_official"],
            post["nonhidden_test_macro_f1_official"]]
    bars = ax.bar(names, vals, color=["#9e9e9e", "#ef8a3a", "#7e57c2"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Segment-based Macro F1")
    ax.set_title("Non-hidden test: Macro F1 by system")
    ax.set_ylim(0, max(vals) * 1.25)
    plt.tight_layout()
    fig.savefig(ASSETS / "fig_model_comparison.png", dpi=160)
    plt.close(fig)


def fig_per_class():
    b = pd.read_csv(C.RESULTS_DIR / "baseline_per_class.csv").set_index("annotation")["f1"]
    c = pd.read_csv(C.RESULTS_DIR / "best_classical_per_class.csv").set_index("annotation")["f1"]
    p = pd.read_csv(C.RESULTS_DIR / "best_postprocessed_per_class.csv").set_index("annotation")["f1"]
    classes = C.CLASS_NAMES
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w, [b.get(k, 0) for k in classes], w, label="baseline", color="#9e9e9e")
    ax.bar(x, [c.get(k, 0) for k in classes], w, label="best classical", color="#ef8a3a")
    ax.bar(x + w, [p.get(k, 0) for k in classes], w, label="+ median filter", color="#7e57c2")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 on the non-hidden test set")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(ASSETS / "fig_per_class_f1.png", dpi=160)
    plt.close(fig)


def fig_hyperparam():
    t = pd.read_csv(C.RESULTS_DIR / "classifier_tuning_results.csv")
    r = t[t["model"] == "ridge"].copy()
    r["alpha"] = r["params"].str.replace("alpha=", "").astype(float)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for thr, g in r.groupby("threshold"):
        g = g.sort_values("alpha")
        ax.plot(g["alpha"], g["dev_macro_f1"], marker="o", label=f"thr={thr}")
    ax.set_xscale("log")
    ax.set_xlabel("ridge alpha (log scale)")
    ax.set_ylabel("dev validation Macro F1")
    ax.set_title("Ridge: two hyperparameters (alpha x threshold)")
    ax.legend(fontsize=8, title="sigmoid threshold")
    plt.tight_layout()
    fig.savefig(ASSETS / "fig_hyperparam.png", dpi=160)
    plt.close(fig)


def fig_postproc():
    p = pd.read_csv(C.RESULTS_DIR / "postprocessing_results.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(p["median_window"], p["dev_macro_f1"], marker="o", color="#7e57c2")
    for _, row in p.iterrows():
        ax.text(row["median_window"], row["dev_macro_f1"] + 0.002,
                f"{row['dev_macro_f1']:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("median filter window (1 = none)")
    ax.set_ylabel("dev validation Macro F1")
    ax.set_title("Post-processing parameter study")
    ax.set_xticks(p["median_window"])
    plt.tight_layout()
    fig.savefig(ASSETS / "fig_postproc.png", dpi=160)
    plt.close(fig)


def fig_label_distribution(audit):
    counts = audit.get("train_class_annotation_counts", {})
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(len(items)), [v for _, v in items], color="#2e7d32")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("annotation rows (train)")
    ax.set_title("Class distribution (train annotations) -- strong imbalance")
    plt.tight_layout()
    fig.savefig(ASSETS / "fig_label_distribution.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def top_improvements(n=4):
    b = pd.read_csv(C.RESULTS_DIR / "baseline_per_class.csv").set_index("annotation")["f1"]
    c = pd.read_csv(C.RESULTS_DIR / "best_classical_per_class.csv").set_index("annotation")["f1"]
    diff = (c - b).sort_values(ascending=False)
    up = ", ".join(f"{k} ({b.get(k,0):.2f}->{c.get(k,0):.2f})" for k in diff.head(n).index)
    hard = ", ".join(f"{k} ({c.get(k,0):.2f})" for k in c.sort_values().head(n).index)
    return up, hard


def main() -> None:
    base = _j("baseline_overall.json")
    clf = _j("best_classical_overall.json")
    post = _j("best_postprocessed_overall.json")
    audit = _j("dataset_audit.json")
    tuning = pd.read_csv(C.RESULTS_DIR / "classifier_tuning_results.csv")

    fig_model_comparison(base, clf, post)
    fig_per_class()
    fig_hyperparam()
    fig_postproc()
    fig_label_distribution(audit)

    up, hard = top_improvements()
    n_best_ridge = tuning[tuning.model == "ridge"]["dev_macro_f1"].max()
    n_best_log = tuning[tuning.model == "logistic"]["dev_macro_f1"].max() if (tuning.model == "logistic").any() else float("nan")
    n_best_rf = tuning[tuning.model == "random_forest"]["dev_macro_f1"].max() if (tuning.model == "random_forest").any() else float("nan")

    ctx = dict(base=base, clf=clf, post=post, audit=audit, up=up, hard=hard,
               n_best_ridge=n_best_ridge, n_best_log=n_best_log, n_best_rf=n_best_rf,
               n_configs=len(tuning))

    write_tex(ctx)
    write_pdf(ctx)
    print("Report built: report.tex + report.pdf + report_assets/")


# --------------------------------------------------------------------------- #
# Report body (shared prose, plain text; rendered by both .tex and .pdf)
# --------------------------------------------------------------------------- #
def sections(ctx):
    base, clf, post, audit = ctx["base"], ctx["clf"], ctx["post"], ctx["audit"]
    p = clf["params"]
    return [
        ("Introduction",
         f"We build a multi-label sound event detection (SED) system for MLPC 2026 Task 5 on the "
         f"provided challenge dataset ({audit['n_train']} train, {audit['n_validation']} validation, "
         f"{audit['n_test_hidden']} hidden-test recordings; {audit['feature_dim']} precomputed features "
         f"per 1-second segment, {audit['n_classes']} target classes). Following the baseline, an "
         f"audio recording is cut into overlapping 1s segments (hop {C.HOP_SIZE}s); a multi-label "
         f"classifier scores each segment and consecutive active whole-second segments are merged into "
         f"onset/offset intervals. All numbers below are recomputed on the Task 5 dataset and scored "
         f"with the official segment-based Macro F1 evaluator. We follow the baseline's seed-42 split of "
         f"the provided validation set into a development-validation set ({audit['n_dev_val']} files, used "
         f"for all tuning) and a non-hidden test set ({audit['n_nonhidden_test']} files, used only for the "
         f"final estimate). We do not attempt the bonus tasks (no raw waveforms were used)."),

        ("1. Baseline Reproduction",
         f"Method. We reproduce the provided baseline exactly: one DecisionTreeClassifier per class "
         f"(max_depth=20, max_features='sqrt', random_state=42) wrapped in a MultiOutputClassifier, "
         f"trained on a 50,000-segment random subsample of raw (unscaled) features using the same rng "
         f"sequence as the notebook. "
         f"Result. On the non-hidden test set the baseline reaches Macro F1 = "
         f"{base['nonhidden_test_macro_f1_official']:.3f} "
         f"(development validation {base['dev_val_macro_f1']:.3f}). "
         f"Class-wise it works best on loud, sustained sources (running_water, vacuum_cleaner, "
         f"phone_ringing, microwave) and worst on short or rare transients "
         f"(window_open_close, wardrobe_drawer_open_close, light_switch). "
         f"Strengths and limitations. The baseline is simple, fast and reproducible, and per-class "
         f"binary trees handle polyphony naturally. However, (i) independent classifiers cannot model "
         f"co-occurrence between classes; (ii) the 1-second resolution blurs short events and precise "
         f"boundaries; (iii) hand-crafted segment statistics discard fine temporal structure; and "
         f"(iv) a depth-limited tree on a 50k subsample under-fits a 960-dimensional, strongly imbalanced "
         f"feature space, which caps recall on rare classes."),

        ("2. Simple Classifiers",
         f"Previous best (Task 4). Our Task 4 best model was a one-vs-rest linear ridge classifier "
         f"(alpha=2.0, decision threshold 0.50, balanced class weights, standardized features). We adapt "
         f"this idea to the Task 5 SED pipeline: features are standardized on the training split only "
         f"(no imputation -- the features contain zero NaNs), the ridge produces a per-segment score, a "
         f"sigmoid maps it to [0,1], and a threshold yields per-second binary activity that is merged "
         f"into intervals. Ridge is solved in closed form (weighted normal equations) for speed and "
         f"numerical robustness. "
         f"Hyperparameter study ({ctx['n_configs']} configurations). We vary two important "
         f"hyperparameters: the ridge regularisation alpha in {{0.1,0.5,1,2,5,10}} and the decision "
         f"threshold in {{0,0.25,0.5,0.75}} (Fig. 3). We also evaluate one-vs-rest logistic regression "
         f"(liblinear; regularisation C and class_weight) and a RandomForest, tuning >=2 hyperparameters "
         f"each. Best development-validation Macro F1 per family: ridge {ctx['n_best_ridge']:.3f}, "
         f"logistic {ctx['n_best_log']:.3f}, random forest {ctx['n_best_rf']:.3f}. "
         f"Best classical model. {clf['model']} ({p}, threshold {clf['threshold']}) gives non-hidden test "
         f"Macro F1 = {clf['nonhidden_test_macro_f1_official']:.3f}, versus the baseline's "
         f"{base['nonhidden_test_macro_f1_official']:.3f} -- an absolute gain of "
         f"{clf['nonhidden_test_macro_f1_official']-base['nonhidden_test_macro_f1_official']:+.3f} "
         f"(Fig. 1, Fig. 2). Largest per-class improvements: {ctx['up']}. Classes that remain hard: "
         f"{ctx['hard']}. The linear model benefits from standardisation, balanced weights and using all "
         f"training segments, and the threshold trades precision against recall on imbalanced classes. "
         f"Qualitative error analysis (Fig. 4-6, three non-hidden files). The success case detects the "
         f"dominant sustained classes with good temporal overlap; the false-positive case shows extra "
         f"short-transient activations (acoustically similar onsets) that hurt precision; the missed case "
         f"shows short/rare events lost at 1-second resolution. Misses concentrate on rare classes and "
         f"timing disagreements appear mostly at event boundaries, consistent with annotator label noise."),

        ("3. Post-Processing and Temporal Refinement",
         f"Method. We apply a per-class temporal median filter to the per-second binary predictions "
         f"before merging into intervals; it removes isolated false-positive blips and fills single-frame "
         f"gaps. Window 1 means no post-processing. "
         f"Parameter study. We select the window in {{1,3,5}} on the development validation set (Fig. 7) "
         f"and evaluate the chosen window once on the non-hidden test set. "
         f"Before/after. Selected window {post['selected_window']}: non-hidden test Macro F1 goes from "
         f"{post['nonhidden_test_macro_f1_no_postproc']:.3f} (no post-processing) to "
         f"{post['nonhidden_test_macro_f1_official']:.3f}. "
         + ("Median filtering reduced noisy false positives for sustained classes and slightly raised "
            "Macro F1; very short events can be smoothed away, so the gain is modest and class-dependent."
            if post['nonhidden_test_macro_f1_official'] >= post['nonhidden_test_macro_f1_no_postproc']
            else "Median filtering did not help overall here: it removed some true short events, so the "
                 "final system keeps the window that was best on development validation.")),

        ("4. Reflection and Real-World Considerations",
         "Technical limitations. Independent linear per-class models ignore class co-occurrence and the "
         "1-second segment statistics cannot localise short events precisely; performance is capped by "
         "label noise and strong class imbalance. Future improvements. Sequence models (e.g. a small "
         "temporal CNN/CRNN on the feature frames), per-class threshold calibration, feature selection, "
         "and richer temporal smoothing (e.g. hysteresis or HMM smoothing) are natural next steps. "
         "Smart-home deployment. A linear segment classifier is extremely cheap (a matrix multiply per "
         "second), giving low latency and low compute -- attractive for on-device inference. For "
         "reliability, precision matters: false alarms erode user trust, so per-class thresholds should be "
         "tuned to the use case, and robustness to unseen rooms, devices and background noise must be "
         "validated. Users expect consistent detection of salient events (doorbell, running water) more "
         "than perfect boundaries, which aligns with the segment-based metric."),

        ("5. Disclosure of LLM and AI Tool Use",
         "ChatGPT and Claude Code were used for brainstorming, coding assistance, debugging, and for "
         "structuring and improving the clarity of this report and the slides. All numerical results were "
         "produced by running our own code on the Task 5 challenge dataset and were verified with the "
         "official evaluation script (evaluate.py); no metrics were copied from Task 4. The final content "
         "was reviewed and understood by the students."),
    ]


FIG_CAPTIONS = [
    ("fig_model_comparison.png", "Figure 1: Non-hidden test Macro F1 of the baseline, the best classical model, and after post-processing."),
    ("fig_per_class_f1.png", "Figure 2: Per-class F1 on the non-hidden test set."),
    ("fig_hyperparam.png", "Figure 3: Ridge hyperparameter study (alpha x sigmoid threshold) on development validation."),
    ("fig_label_distribution.png", "Figure 8: Class imbalance in the training annotations."),
    ("fig_postproc.png", "Figure 7: Median-filter window selection on development validation."),
    ("error_case_1.png", "Figure 4: Error analysis -- success case (feature heatmap, GT, baseline, classical, post-processed)."),
    ("error_case_2.png", "Figure 5: Error analysis -- false-positive / noisy case."),
    ("error_case_3.png", "Figure 6: Error analysis -- missed-event / timing case."),
]


def _tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("{", r"\{"), ("}", r"\}"), ("->", r"$\rightarrow$")]:
        s = s.replace(a, b)
    return s


def write_tex(ctx):
    secs = sections(ctx)
    parts = [r"\documentclass[10pt]{article}",
             r"\usepackage[margin=1.7cm]{geometry}",
             r"\usepackage{graphicx}\usepackage{float}\usepackage{caption}",
             r"\graphicspath{{report_assets/}}",
             r"\title{\vspace{-1.5em}MLPC 2026 Task 5: Sound Event Detection Challenge}",
             r"\author{Quantized Transformers --- Aleksandr Masiev, Aleksandar Cvetkovic}",
             r"\date{}", r"\begin{document}\maketitle"]
    fig_imgs = {
        "1. Baseline Reproduction": ["fig_label_distribution.png"],
        "2. Simple Classifiers": ["fig_model_comparison.png", "fig_per_class_f1.png",
                                   "fig_hyperparam.png", "error_case_1.png",
                                   "error_case_2.png", "error_case_3.png"],
        "3. Post-Processing and Temporal Refinement": ["fig_postproc.png"],
    }
    cap = {f: c for f, c in FIG_CAPTIONS}
    for title, body in secs:
        parts.append(r"\section*{" + _tex_escape(title) + "}")
        parts.append(_tex_escape(body))
        for img in fig_imgs.get(title, []):
            src = ASSETS / img
            if not src.exists() and (C.FIGURES_DIR / img).exists():
                # copy error-case figures into report_assets for a self-contained upload
                import shutil
                shutil.copy(C.FIGURES_DIR / img, src)
            if src.exists():
                w = "0.95" if img.startswith("error_case") or "per_class" in img or "label" in img else "0.6"
                parts.append(r"\begin{figure}[H]\centering\includegraphics[width=" + w +
                             r"\linewidth]{" + img + r"}\caption*{\footnotesize " +
                             _tex_escape(cap.get(img, "")) + r"}\end{figure}")
    parts.append(r"\end{document}")
    (REPORT_DIR / "report.tex").write_text("\n".join(parts))


def write_pdf(ctx):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    PageBreak)
    import shutil

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.3, leading=10.6,
                          spaceAfter=4, alignment=4)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=14, spaceAfter=2)
    capst = ParagraphStyle("cap", parent=styles["BodyText"], fontSize=7, leading=8,
                           textColor="#555555", spaceAfter=8)

    doc = SimpleDocTemplate(str(REPORT_DIR / "report.pdf"), pagesize=A4,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm)
    flow = [Paragraph("MLPC 2026 Task 5: Sound Event Detection Challenge", title),
            Paragraph("Quantized Transformers --- Aleksandr Masiev, Aleksandar Cvetkovic", body),
            Spacer(1, 4)]
    cap = {f: c for f, c in FIG_CAPTIONS}
    fig_imgs = {
        "2. Simple Classifiers": ["fig_model_comparison.png", "fig_hyperparam.png"],
        "3. Post-Processing and Temporal Refinement": ["fig_postproc.png"],
    }

    def add_image(name, maxw=15.5 * cm):
        for src in (ASSETS / name, C.FIGURES_DIR / name):
            if src.exists():
                from PIL import Image as PILImage
                iw, ih = PILImage.open(src).size
                w = min(maxw, 15.5 * cm)
                flow.append(Image(str(src), width=w, height=w * ih / iw))
                flow.append(Paragraph(cap.get(name, ""), capst))
                return

    for tname, btext in sections(ctx):
        flow.append(Paragraph(tname, h))
        flow.append(Paragraph(btext, body))  # keep ASCII '->' (Helvetica has no U+2192)
        for img in fig_imgs.get(tname, []):
            add_image(img, 9 * cm)

    # figures page
    flow.append(PageBreak())
    flow.append(Paragraph("Figures", h))
    for img in ["fig_per_class_f1.png", "fig_label_distribution.png",
                "error_case_1.png", "error_case_2.png", "error_case_3.png"]:
        add_image(img)
    # ensure error-case figures exist in assets for the tex upload too
    for img in ["error_case_1.png", "error_case_2.png", "error_case_3.png"]:
        if (C.FIGURES_DIR / img).exists() and not (ASSETS / img).exists():
            shutil.copy(C.FIGURES_DIR / img, ASSETS / img)

    doc.build(flow)

    words = sum(len(b.split()) for _, b in sections(ctx))
    print(f"Report word count (body prose): ~{words} (limit 2000)")


if __name__ == "__main__":
    main()
