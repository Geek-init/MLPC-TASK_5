# MLPC 2026 Task 5 — SED solution (Quantized Transformers)

Non-bonus solution. Uses **only** the provided challenge dataset with precomputed
`.npz` audio features (no raw waveforms, no CRNN / pretrained / deep SED, no
Bonus 1 or Bonus 2).

## What this does
A multi-label Sound Event Detection pipeline: cut each recording into overlapping
1-second segments (hop 0.5s), standardize features, score every segment with a
multi-label classifier, smooth the per-second predictions with a median filter,
and merge consecutive active whole-second segments into onset/offset intervals.
Scored with the **official** `challenge_baseline/evaluate.py` (segment-based
Macro F1, 1-second resolution).

## Pipeline (run in order, from `task5_solution/src/`)
```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python3 00_audit_dataset.py            # confirm structure, classes, split, NaNs
python3 01_build_cache.py              # cache feature/label matrices to cache/
python3 02_run_baseline.py             # reproduce decision-tree baseline
python3 03_run_classical_models.py     # ridge / logistic / RF tuning + selection
python3 04_postprocess.py              # median-filter window study
python3 05_error_analysis.py           # 3 qualitative error-case figures
python3 06_generate_hidden_predictions.py   # final hidden-test CSV
python3 07_validate_submission.py      # validate the submission CSV
python3 08_build_report.py             # report figures + report.tex + report.pdf
python3 09_build_slides.py             # slides.pdf + slides.pptx + slides.md
python3 10_final_checklist.py          # FINAL_CHECKLIST.md
```

## Honest evaluation protocol vs. final submission
- **Reported numbers** use the baseline's seed-42 split of the provided
  validation set into a **development-validation** set (tuning / model selection)
  and a held-out **non-hidden test** set (final estimate only). All Task 5 numbers
  are recomputed on the Task 5 dataset — none are copied from Task 4.
- **Hidden-test submission** (`06_...`): the selected configuration is **retrained
  on train + the full provided validation labels**, because the hidden test split
  is separate and unlabeled. This is intentional and disclosed in the report.

## Environment note (important for reproducibility)
This machine is arm64 macOS with numpy linked against **Accelerate** BLAS. There,
sklearn's `RidgeClassifier` solver hangs, `SGDClassifier`/lbfgs logistic are very
slow, and any `n_jobs>1` (loky) deadlocks. We therefore: solve ridge in **closed
form** with numpy normal equations (mathematically identical to a one-vs-rest
RidgeClassifier with balanced weights), use `liblinear` (primal) for logistic,
and keep all estimators `n_jobs=1`. Features have **zero NaNs**, so no imputation
is used; standardization is fit on the training split only.

## Key constants (reused from the baseline)
15 classes, 960 features/segment, segment length 1.0s, hop 0.5s, seed 42.

## Outputs
- `results/` — all metrics (baseline, best classical, post-processed), tuning
  tables, per-class CSVs, audit, error-analysis notes.
- `figures/` — error-case figures. `report/report_assets/` — report figures.
- `submission/predictions_hidden_test.csv` — the file to submit (+ validation summary).
- `report/report.pdf` (+ `report.tex` for Overleaf), `slides/slides.pdf` (+ `.pptx`).
- `cache/` — cached matrices and fitted model artifacts (safe to delete; rebuilt by `01`).
