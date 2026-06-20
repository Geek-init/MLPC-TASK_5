"""Phase 7: validate the hidden-test prediction CSV against the submission rules.

Checks: exact column order, no index column, .wav filenames, the 15 allowed
classes only, onset < offset, numeric non-negative timestamps, offsets within
the per-file duration inferred from the test segments, and full file coverage.
Also runs the official evaluate.py loader as an independent format check.
Writes submission/submission_validation_summary.txt.
"""
from __future__ import annotations

import csv
import time

import numpy as np
import pandas as pd

import common as C

EPS = 1e-6


def main() -> None:
    t0 = time.time()
    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        out_lines.append(s)

    sub = C.SUBMISSION_DIR / "predictions_hidden_test.csv"
    emit("=== Phase 7: submission validation ===")
    emit(f"File: {sub}")

    checks: dict[str, bool] = {}

    # --- Exact header / no index column (read raw) ------------------------- #
    with open(sub, newline="") as fh:
        header = next(csv.reader(fh))
    checks["columns_exact"] = header == ["filename", "annotation", "onset", "offset"]
    checks["no_index_column"] = header[0] == "filename"
    emit(f"Header: {header}")

    df = pd.read_csv(sub)
    emit(f"Rows: {len(df)}")

    # --- Per-file duration inferred from the hidden test segments ---------- #
    _, _, starts_te, index_te = C.load_split_cache("test", with_labels=False)
    durations: dict[str, float] = {}
    all_test_wavs = set()
    for wav, start, count in index_te:
        st = starts_te[start:start + count]
        ws = st[np.isclose(st % 1.0, 0.0)]
        durations[wav] = float(ws.max() + C.SEGMENT_LENGTH) if len(ws) else 0.0
        all_test_wavs.add(wav)

    # --- Field-level checks ------------------------------------------------ #
    checks["filenames_wav"] = bool(df["filename"].astype(str).str.endswith(".wav").all())
    checks["classes_valid"] = set(df["annotation"].unique()).issubset(set(C.CLASS_NAMES))
    onset = pd.to_numeric(df["onset"], errors="coerce")
    offset = pd.to_numeric(df["offset"], errors="coerce")
    checks["timestamps_numeric"] = not (onset.isna().any() or offset.isna().any())
    checks["onset_lt_offset"] = bool((offset > onset).all())
    checks["no_negative"] = bool((onset >= -EPS).all() and (offset >= -EPS).all())

    within = True
    bad_examples = []
    for fn, off in zip(df["filename"], offset):
        dur = durations.get(fn)
        if dur is None:
            within = False
            bad_examples.append((fn, "unknown file"))
        elif off > dur + EPS:
            within = False
            bad_examples.append((fn, f"offset {off} > dur {dur}"))
    checks["offsets_within_duration"] = within

    checks["filenames_in_testset"] = set(df["filename"].unique()).issubset(all_test_wavs)

    # --- Coverage ---------------------------------------------------------- #
    files_with_pred = set(df["filename"].unique())
    n_empty = len(all_test_wavs - files_with_pred)
    emit("")
    emit(f"Hidden test files total      : {len(all_test_wavs)}")
    emit(f"Files with >=1 prediction    : {len(files_with_pred)}")
    emit(f"Files with no prediction     : {n_empty}")
    emit(f"Predicted classes            : {sorted(df['annotation'].unique())}")
    emit("")

    # --- Independent format check via official evaluate.py loader ---------- #
    try:
        C.EVAL.load_annotation_csv(sub, ground_truth=False)
        checks["official_loader_accepts"] = True
    except Exception as exc:  # noqa: BLE001
        checks["official_loader_accepts"] = False
        emit(f"official loader error: {exc}")

    emit("Checks:")
    for k, v in checks.items():
        emit(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if bad_examples:
        emit("")
        emit("Out-of-duration examples (first 5):")
        for fn, why in bad_examples[:5]:
            emit(f"  {fn}: {why}")

    all_pass = all(checks.values())
    emit("")
    emit(f"RESULT: {'ALL CHECKS PASSED' if all_pass else 'VALIDATION FAILED'}")
    emit(f"Runtime {time.time()-t0:.1f}s")

    (C.SUBMISSION_DIR / "submission_validation_summary.txt").write_text("\n".join(out_lines))
    print(f"\nSaved: {C.SUBMISSION_DIR/'submission_validation_summary.txt'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
