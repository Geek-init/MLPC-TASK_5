#!/usr/bin/env bash
# Reproduce the full MLPC 2026 Task 5 solution end to end.
# Run from the task5_solution/ directory:  bash run_all.sh
set -euo pipefail

# Single-threaded-ish BLAS avoids the Accelerate/loky issues on arm64 macOS.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

cd "$(dirname "$0")/src"

python3 00_audit_dataset.py
python3 01_build_cache.py
python3 02_run_baseline.py
python3 03_run_classical_models.py
python3 04_postprocess.py
python3 05_error_analysis.py
python3 06_generate_hidden_predictions.py
python3 07_validate_submission.py
python3 08_build_report.py
python3 09_build_slides.py
python3 10_final_checklist.py

echo "Done. See ../FINAL_CHECKLIST.md for the files to upload."
