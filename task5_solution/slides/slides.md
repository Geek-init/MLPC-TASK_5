# MLPC 2026 Task 5

Group: Quantized Transformers  
Aleksandr Masiev  •  Aleksandar Cvetkovic

_Non-bonus solution — precomputed audio features only_

---

## 1. Task & Final System Architecture

- Goal: detect 15 domestic sound events with onsets/offsets in recordings of any length
- Dataset: 3704 train / 999 val / 1007 hidden test; 960 precomputed features per 1s segment (hop 0.5s)
- Pipeline: 1s segments → standardize → multi-label classifier → per-second activity
- → median-filter smoothing → merge consecutive seconds into onset/offset intervals
- Metric: official segment-based Macro F1 (1s resolution), evaluated with evaluate.py
- Honest protocol: tune on dev-validation, report on a held-out non-hidden test split

---

## 2. Baseline Reproduction

- Per-class DecisionTree (max_depth=20, max_features='sqrt'), MultiOutputClassifier
- Trained on 50k subsampled raw segments — reproduced exactly (seed 42)
- Non-hidden test Macro F1 = 0.316
- Best on loud/sustained classes (running_water, vacuum_cleaner, phone_ringing)
- Weak on short/rare events (window_open_close, wardrobe_drawer, light_switch)
- Limits: no class co-occurrence, 1s resolution, under-fit on 960-dim imbalanced data

![](fig_label_distribution.png)

---

## 3. Classical Classifiers & Hyperparameters

- Start from Task 4 best: one-vs-rest linear ridge (closed-form, balanced weights)
- Tuned 2+ hyperparameters: ridge alpha {0.1..10} × sigmoid threshold {0..0.75}
- Also: logistic regression (C, class_weight) and RandomForest (depth)
- Best classical: logistic (C=1.0,class_weight=None), threshold 0.3
- Non-hidden test Macro F1 = 0.436 (baseline 0.316, +0.120)

![](fig_hyperparam.png)

---

## 4. Post-Processing (Median Filter)

- Per-class temporal median filter on per-second predictions (window 1 = none)
- Removes isolated false-positive blips; fills single-second gaps
- Window selected on dev-validation; evaluated once on non-hidden test
- Selected window = 3
- Non-hidden test: 0.436 → 0.446 (+0.010)

![](fig_postproc.png)

---

## 5. Qualitative Error Analysis

- Success: dominant sustained events detected with good temporal overlap
- False positives: extra short transients from acoustically similar onsets
- Misses: short/rare events lost at 1s resolution; boundary timing shifts
- Visuals use the precomputed mel feature representation (no raw waveform)

![](error_case_1.png)

---

## 6. Final System, Limits & Deployment

- Final system: logistic + median filter (w=3); retrained on train + full validation for the hidden test CSV
- Cheap & low-latency (one matrix multiply per second) — good for on-device smart-home use
- Limits: independent per-class linear models, 1s resolution, label noise, imbalance
- Future: temporal CNN/CRNN, per-class threshold calibration, hysteresis smoothing
- Deployment: precision matters (false alarms erode trust); validate across rooms/devices

![](fig_model_comparison.png)

---
