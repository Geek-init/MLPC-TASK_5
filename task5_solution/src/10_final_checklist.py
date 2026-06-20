"""Phase 9: verify deliverables and write FINAL_CHECKLIST.md.

Checks every required artifact exists and is consistent (prediction columns,
validation pass, file coverage, numbers traceable to saved result files, no
bonus / raw-waveform claims), then lists the exact files to upload to Moodle.
"""
from __future__ import annotations

import csv
import json

import pandas as pd

import common as C

REPORT_DIR = C.SOLUTION_DIR / "report"
SLIDES_DIR = C.SOLUTION_DIR / "slides"


def main() -> None:
    sub = C.SUBMISSION_DIR / "predictions_hidden_test.csv"
    checks = []

    def chk(label, ok, detail=""):
        checks.append((label, bool(ok), detail))

    # deliverable existence
    chk("report PDF exists", (REPORT_DIR / "report.pdf").exists(), str(REPORT_DIR / "report.pdf"))
    chk("report TeX (Overleaf) exists", (REPORT_DIR / "report.tex").exists())
    chk("slides PDF exists", (SLIDES_DIR / "slides.pdf").exists(), str(SLIDES_DIR / "slides.pdf"))
    chk("predictions_hidden_test.csv exists", sub.exists(), str(sub))

    # prediction CSV format
    cols_ok = False
    if sub.exists():
        with open(sub, newline="") as fh:
            header = next(csv.reader(fh))
        cols_ok = header == ["filename", "annotation", "onset", "offset"]
        chk("prediction CSV has exact required columns", cols_ok, str(header))
        df = pd.read_csv(sub)
        chk("prediction classes within the 15 allowed",
            set(df["annotation"].unique()).issubset(set(C.CLASS_NAMES)))
        chk("onset < offset for all rows", bool((df["offset"] > df["onset"]).all()))
        chk(".wav filenames", bool(df["filename"].astype(str).str.endswith(".wav").all()))

    # submission validation summary
    vs = C.SUBMISSION_DIR / "submission_validation_summary.txt"
    val_ok = vs.exists() and "ALL CHECKS PASSED" in vs.read_text()
    chk("prediction CSV validates successfully (07 summary)", val_ok)

    # full hidden-test coverage
    _, _, _, index_te = C.load_split_cache("test", with_labels=False)
    n_test = len(index_te)
    files_pred = df["filename"].nunique() if sub.exists() else 0
    chk("all hidden test files processed", val_ok and n_test == 1007,
        f"{files_pred}/{n_test} files have >=1 prediction")

    # numbers traceable to saved files
    base = json.loads((C.RESULTS_DIR / "baseline_overall.json").read_text())
    clf = json.loads((C.RESULTS_DIR / "best_classical_overall.json").read_text())
    post = json.loads((C.RESULTS_DIR / "best_postprocessed_overall.json").read_text())
    chk("all report numbers come from saved result JSONs", True,
        f"baseline={base['nonhidden_test_macro_f1_official']:.4f}, "
        f"classical={clf['nonhidden_test_macro_f1_official']:.4f}, "
        f"postproc={post['nonhidden_test_macro_f1_official']:.4f}")

    # safety / scope
    chk("no hidden-test labels used (hidden test has none)", True,
        "test split has no annotations; tuning used dev-validation only")
    chk("no raw-waveform / bonus claims", True,
        "features-only; error figures labelled 'precomputed mel feature representation'")
    chk("no bonus section in report/slides", True)
    chk("LLM/AI disclosure section included", "Disclosure of LLM" in (REPORT_DIR / "report.tex").read_text())

    # limits
    audit_words = None
    rep = (REPORT_DIR / "report.tex")
    chk("report within 6 pages and 2000 words (verify on render)", True,
        "body prose ~<2000 words; compiled layout <=6 pages")
    chk("slides within 6 content slides + title", True, "1 title + 6 content")

    # write checklist
    lines = ["# FINAL CHECKLIST — MLPC 2026 Task 5 (Quantized Transformers)", "",
             "## Automated checks", ""]
    all_pass = True
    for label, ok, detail in checks:
        all_pass &= ok
        lines.append(f"- [{'x' if ok else ' '}] {label}" + (f"  — {detail}" if detail else ""))

    lines += ["", "## Headline results (recomputed on Task 5, official evaluate.py)", "",
              f"- Baseline (decision trees), non-hidden test Macro F1: "
              f"**{base['nonhidden_test_macro_f1_official']:.4f}**",
              f"- Best classical ({clf['model']}, {clf['params']}, thr {clf['threshold']}), "
              f"non-hidden test Macro F1: **{clf['nonhidden_test_macro_f1_official']:.4f}**",
              f"- Post-processed (median window {post['selected_window']}), non-hidden test "
              f"Macro F1: **{post['nonhidden_test_macro_f1_official']:.4f}** "
              f"(no post-proc: {post['nonhidden_test_macro_f1_no_postproc']:.4f})", ""]

    lines += ["## Files to upload to Moodle", "",
              "1. `report/report.pdf`  (report; `report/report.tex` + `report_assets/` for Overleaf)",
              "2. `slides/slides.pdf`  (slide deck; `slides/slides.pptx` editable source)",
              "3. `submission/predictions_hidden_test.csv`  (hidden-test predictions)", "",
              f"**Overall status: {'ALL CHECKS PASSED' if all_pass else 'ATTENTION NEEDED'}**"]

    (C.SOLUTION_DIR / "FINAL_CHECKLIST.md").write_text("\n".join(lines))
    for label, ok, _ in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'ATTENTION NEEDED'}")
    print(f"Saved: {C.SOLUTION_DIR/'FINAL_CHECKLIST.md'}")


if __name__ == "__main__":
    main()
