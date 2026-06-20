# Handoff to Teammate

## Summary
- Implemented the full MLPC Task 4 classification pipeline.
- Generated the Overleaf-ready report and all referenced figures.
- Main model selected by validation Macro F1: `linear_ridge` with threshold `0.50`.
- Validation Macro/Micro F1: `0.312` / `0.308`.
- Test Macro/Micro F1: `0.309` / `0.296`.

## How to Run
- From the project root, run: `python outputs/src/run_pipeline.py`
- Or open `outputs/notebooks/task4_classification_final.ipynb` and run the cells.
- The script reads only `data/MLPC2026_dataset_development/` and writes only under `outputs/`.

## Generated Files
- Report: `outputs/overleaf_report/mlpc_task4_report.tex`
- Figures: `outputs/overleaf_report/figures/`
- Notebook: `outputs/notebooks/task4_classification_final.ipynb`
- Results: `outputs/results/split_summary.csv`, `metrics_summary.csv`, `per_class_metrics.csv`, `hyperparameter_results.csv`, `case_studies.json`
- Extra diagnostics: `outputs/results/npz_structure_summary.csv`, `feature_summary.json`, `split_diagnostics.json`, `preprocessing_summary.json`, `final_checks.json`

## Label Aggregation and Split Strategy
- Used aligned `.npz` annotation tensors with shape `[T, C, A]`.
- Annotator vote is positive if overlap is greater than zero.
- Final segment label is positive if at least half of annotators vote positive.
- Split is at recording level and grouped by `collector_id`.
- Collector grouping was used: `True`.
- Recording leakage detected: `False`.
- Collector leakage detected: `False`.

## Things to Verify
- Compile the `.tex` file in Overleaf together with the `figures/` folder.
- Check that the report stays within 6 pages and 2000 words after Overleaf rendering.
- Confirm teammate name before submission.
- Read the case study paragraphs and decide if the examples are convincing.
- Confirm that no Moodle-specific formatting requirement is missing.

## Remaining Manual Checks Before Submission
- Upload only the report source and figures for this stage; no PDF, slides, or PowerPoint were produced.
- If a final PDF is later needed, compile it in Overleaf after teammate review.
- Keep the generated CSVs available in case the tutor asks how numbers were computed.
