# Run Instructions

1. Keep the raw dataset in `data/MLPC2026_dataset_development/`.
2. From the project root, run:

```bash
python outputs/src/run_pipeline.py
```

3. To reproduce interactively, open and run:

```text
outputs/notebooks/task4_classification_final.ipynb
```

4. Upload `outputs/overleaf_report/mlpc_task4_report.tex` and the complete `outputs/overleaf_report/figures/` folder to Overleaf.

The pipeline uses deterministic random seeds and fits imputation/scaling only on the training split.
