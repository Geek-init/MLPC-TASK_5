"""Phase 1: build on-disk caches of the heavy feature/label matrices.

Loads every .npz once and writes concatenated arrays to cache/ so the later
stages never re-read thousands of npz files. Also records the baseline-faithful
validation split (dev-val vs non-hidden test) as filename lists.

Splits cached:
  train       -> X, Y, starts, index           (all ~226k segments)
  validation  -> X, Y, starts, index           (dev-val + non-hidden test)
  test        -> X,    starts, index           (hidden, no labels)
"""
from __future__ import annotations

import json
import time

import common as C


def main() -> None:
    t0 = time.time()

    train_files = C.list_npz(C.PATH_TRAIN)
    val_files = C.list_npz(C.PATH_VAL)
    test_files = C.list_npz(C.PATH_TEST)

    print(f"Building cache: train={len(train_files)}, "
          f"validation={len(val_files)}, test={len(test_files)}", flush=True)

    print("Caching train ...", flush=True)
    C.build_split_cache("train", train_files, with_labels=True)

    print("Caching validation ...", flush=True)
    C.build_split_cache("validation", val_files, with_labels=True)

    print("Caching test (hidden, no labels) ...", flush=True)
    C.build_split_cache("test", test_files, with_labels=False)

    # Record the seed-42 split as .wav filename lists -----------------------
    dev_val, nonhidden = C.validation_split()
    split = {
        "seed": C.SEED,
        "dev_val_wav": [C.to_wav(f) for f in dev_val],
        "nonhidden_test_wav": [C.to_wav(f) for f in nonhidden],
        "n_dev_val": len(dev_val),
        "n_nonhidden_test": len(nonhidden),
    }
    (C.CACHE_DIR / "validation_split.json").write_text(json.dumps(split, indent=2))
    print(f"Saved split: {len(dev_val)} dev-val / {len(nonhidden)} non-hidden test")

    print(f"Cache build done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
