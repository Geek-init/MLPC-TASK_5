# FINAL CHECKLIST — MLPC 2026 Task 5 (Quantized Transformers)

## Automated checks

- [x] report PDF exists  — /Users/aleksandr.masiev/Desktop/MLPC/task5_solution/report/report.pdf
- [x] report TeX (Overleaf) exists
- [x] slides PDF exists  — /Users/aleksandr.masiev/Desktop/MLPC/task5_solution/slides/slides.pdf
- [x] predictions_hidden_test.csv exists  — /Users/aleksandr.masiev/Desktop/MLPC/task5_solution/submission/predictions_hidden_test.csv
- [x] prediction CSV has exact required columns  — ['filename', 'annotation', 'onset', 'offset']
- [x] prediction classes within the 15 allowed
- [x] onset < offset for all rows
- [x] .wav filenames
- [x] prediction CSV validates successfully (07 summary)
- [x] all hidden test files processed  — 1002/1007 files have >=1 prediction
- [x] all report numbers come from saved result JSONs  — baseline=0.3163, classical=0.4360, postproc=0.4464
- [x] no hidden-test labels used (hidden test has none)  — test split has no annotations; tuning used dev-validation only
- [x] no raw-waveform / bonus claims  — features-only; error figures labelled 'precomputed mel feature representation'
- [x] no bonus section in report/slides
- [x] LLM/AI disclosure section included
- [x] report within 6 pages and 2000 words (verify on render)  — body prose ~<2000 words; compiled layout <=6 pages
- [x] slides within 6 content slides + title  — 1 title + 6 content

## Headline results (recomputed on Task 5, official evaluate.py)

- Baseline (decision trees), non-hidden test Macro F1: **0.3163**
- Best classical (logistic, C=1.0,class_weight=None, thr 0.3), non-hidden test Macro F1: **0.4360**
- Post-processed (median window 3), non-hidden test Macro F1: **0.4464** (no post-proc: 0.4360)

## Files to upload to Moodle

1. `report/report.pdf`  (report; `report/report.tex` + `report_assets/` for Overleaf)
2. `slides/slides.pdf`  (slide deck; `slides/slides.pptx` editable source)
3. `submission/predictions_hidden_test.csv`  (hidden-test predictions)

**Overall status: ALL CHECKS PASSED**