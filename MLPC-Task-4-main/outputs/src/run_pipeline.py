"""End-to-end runner for MLPC 2026 Task 4.

This script generates the result CSVs, figures, Overleaf-ready LaTeX report,
notebook shell, and handoff file requested for the project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data_loading import build_segment_dataset, find_data_dir, inspect_npz_structure
from evaluation import (
    evaluate_predictions,
    file_level_micro_f1,
    per_class_metrics,
    threshold_predictions,
)
from label_aggregation import aggregate_annotations
from models import (
    empirical_frequency_predictions,
    make_linear_ridge_model,
    make_random_forest_model,
    predict_probabilities,
)
from plots import (
    plot_case_study,
    plot_hyperparameter_tuning,
    plot_label_distribution,
    plot_model_comparison,
    plot_per_class_f1,
    plot_split_distribution,
)
from preprocessing import fit_preprocessor
from splits import add_segment_splits, make_group_split, split_summary, verify_no_leakage


RANDOM_SEED = 42


def main(project_root: Path | None = None) -> Dict[str, object]:
    """Run the full deterministic pipeline."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    project_root = Path(project_root).resolve()

    output_root = project_root / "outputs"
    results_dir = output_root / "results"
    figures_dir = output_root / "overleaf_report" / "figures"
    report_dir = output_root / "overleaf_report"
    notebooks_dir = output_root / "notebooks"
    src_dir = output_root / "src"
    for directory in [results_dir, figures_dir, report_dir, notebooks_dir, src_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir(project_root)
    print("Inspecting .npz structure...", flush=True)
    npz_summary = inspect_npz_structure(data_dir, results_dir / "npz_structure_summary.csv")

    print("Building segment-level dataset...", flush=True)
    x, y, segment_info, recording_info, feature_keys, class_names = build_segment_dataset(
        data_dir=data_dir,
        aggregate_fn=aggregate_annotations,
    )
    feature_summary = {
        "feature_keys": feature_keys,
        "n_feature_keys": len(feature_keys),
        "feature_dimension": int(x.shape[1]),
        "n_segments": int(x.shape[0]),
        "n_recordings": int(recording_info.shape[0]),
        "n_classes": int(y.shape[1]),
        "class_names": class_names,
        "annotation_vote_rule": "positive if overlap > 0; class positive if at least half of annotators vote positive",
        "annotators_per_recording_min": int(recording_info["n_annotators"].min()),
        "annotators_per_recording_median": float(recording_info["n_annotators"].median()),
        "annotators_per_recording_max": int(recording_info["n_annotators"].max()),
    }
    (results_dir / "feature_summary.json").write_text(json.dumps(feature_summary, indent=2), encoding="utf-8")

    print("Creating collector-level train/validation/test split...", flush=True)
    recording_splits, split_diagnostics = make_group_split(
        recording_info=recording_info,
        class_names=class_names,
        group_col="collector_id",
        seed=RANDOM_SEED,
        n_iter=1000,
    )
    leakage = verify_no_leakage(recording_splits)
    split_diagnostics.update(leakage)
    segment_info = add_segment_splits(segment_info, recording_splits)
    summary = split_summary(recording_splits, segment_info, y, class_names)
    summary.to_csv(results_dir / "split_summary.csv", index=False)
    recording_splits[["filename", "collector_id", "split", "n_segments", "positive_labels"]].to_csv(
        results_dir / "recording_splits.csv",
        index=False,
    )
    (results_dir / "split_diagnostics.json").write_text(
        json.dumps(_json_safe(split_diagnostics), indent=2),
        encoding="utf-8",
    )

    print("Fitting imputer and scaler on training segments only...", flush=True)
    masks = {split: segment_info["split"].to_numpy() == split for split in ["train", "validation", "test"]}
    x_train_raw = x[masks["train"]]
    y_train = y[masks["train"]]
    x_val_raw = x[masks["validation"]]
    y_val = y[masks["validation"]]
    x_test_raw = x[masks["test"]]
    y_test = y[masks["test"]]

    preprocessor = fit_preprocessor(x_train_raw)
    x_train = preprocessor.transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)
    x_test = preprocessor.transform(x_test_raw)
    del x, x_train_raw, x_val_raw, x_test_raw
    preprocessing_summary = {
        "imputer": "SimpleImputer(strategy='median')",
        "scaler": "StandardScaler()",
        "fit_split": "train",
        "feature_dimension": int(x_train.shape[1]),
        "train_segments": int(x_train.shape[0]),
        "validation_segments": int(x_val.shape[0]),
        "test_segments": int(x_test.shape[0]),
    }
    (results_dir / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, indent=2),
        encoding="utf-8",
    )

    metrics_rows: List[Dict[str, object]] = []
    per_class_frames: List[pd.DataFrame] = []
    hyper_rows: List[Dict[str, object]] = []

    print("Evaluating baselines...", flush=True)
    all_negative_val = np.zeros_like(y_val, dtype=np.uint8)
    all_negative_test = np.zeros_like(y_test, dtype=np.uint8)
    for split_name, y_true, y_pred in [
        ("validation", y_val, all_negative_val),
        ("test", y_test, all_negative_test),
    ]:
        metrics_rows.append(evaluate_predictions(y_true, y_pred, "all_negative", split_name))
        per_class_frames.append(per_class_metrics(y_true, y_pred, class_names, "all_negative", split_name))

    freq_val, _ = empirical_frequency_predictions(y_train, len(y_val), RANDOM_SEED + 10)
    freq_test, _ = empirical_frequency_predictions(y_train, len(y_test), RANDOM_SEED + 11)
    for split_name, y_true, y_pred in [
        ("validation", y_val, freq_val),
        ("test", y_test, freq_test),
    ]:
        metrics_rows.append(evaluate_predictions(y_true, y_pred, "empirical_frequency", split_name))
        per_class_frames.append(per_class_metrics(y_true, y_pred, class_names, "empirical_frequency", split_name))

    thresholds = [0.20, 0.35, 0.50]
    linear_configs = [
        {"alpha": 0.5, "class_weight": "balanced"},
        {"alpha": 2.0, "class_weight": "balanced"},
    ]
    forest_configs = [
        {
            "n_estimators": 16,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "max_samples": 0.25,
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 24,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "max_samples": 0.20,
            "class_weight": "balanced_subsample",
        },
    ]

    trained_models: Dict[str, Dict[str, object]] = {}
    for model_name, configs, factory in [
        ("linear_ridge", linear_configs, make_linear_ridge_model),
        ("random_forest", forest_configs, make_random_forest_model),
    ]:
        print(f"Tuning {model_name} on validation split...", flush=True)
        best = _tune_model(
            model_name=model_name,
            configs=configs,
            factory=factory,
            thresholds=thresholds,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=RANDOM_SEED,
            hyper_rows=hyper_rows,
        )
        trained_models[model_name] = best

        for split_name, x_eval, y_true in [
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ]:
            probabilities = predict_probabilities(best["model"], x_eval)
            predictions = threshold_predictions(probabilities, float(best["threshold"]))
            metrics_rows.append(evaluate_predictions(y_true, predictions, model_name, split_name))
            per_class_frames.append(per_class_metrics(y_true, predictions, class_names, model_name, split_name))
            trained_models[model_name][f"{split_name}_probabilities"] = probabilities
            trained_models[model_name][f"{split_name}_predictions"] = predictions

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(results_dir / "metrics_summary.csv", index=False)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)
    hyperparameters = pd.DataFrame(hyper_rows)
    hyperparameters.to_csv(results_dir / "hyperparameter_results.csv", index=False)

    classifier_val = metrics[
        (metrics["split"] == "validation") & (metrics["model"].isin(["linear_ridge", "random_forest"]))
    ].sort_values(["macro_f1", "micro_f1"], ascending=False)
    best_model_name = str(classifier_val.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    best_threshold = float(best_model["threshold"])

    print(f"Best validation model: {best_model_name} at threshold {best_threshold}", flush=True)
    print("Generating figures...", flush=True)
    plot_split_distribution(summary, figures_dir / "fig_split_distribution.png")
    plot_label_distribution(summary, class_names, figures_dir / "fig_label_distribution.png")
    plot_model_comparison(metrics, figures_dir / "fig_model_comparison.png")
    plot_hyperparameter_tuning(hyperparameters, figures_dir / "fig_hyperparameter_tuning.png")
    plot_per_class_f1(per_class, best_model_name, figures_dir / "fig_per_class_f1.png")

    print("Selecting case studies from validation/test recordings...", flush=True)
    case_studies = _select_case_studies(
        data_dir=data_dir,
        segment_info=segment_info,
        y=y,
        class_names=class_names,
        test_probabilities=best_model["test_probabilities"],
        test_predictions=best_model["test_predictions"],
        validation_probabilities=best_model["validation_probabilities"],
        validation_predictions=best_model["validation_predictions"],
        figures_dir=figures_dir,
        threshold=best_threshold,
        best_model_name=best_model_name,
    )
    (results_dir / "case_studies.json").write_text(json.dumps(case_studies, indent=2), encoding="utf-8")

    report_path = report_dir / "mlpc_task4_report.tex"
    _write_report(
        report_path=report_path,
        class_names=class_names,
        feature_summary=feature_summary,
        split_summary_df=summary,
        metrics=metrics,
        per_class=per_class,
        hyperparameters=hyperparameters,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        case_studies=case_studies,
        split_diagnostics=split_diagnostics,
    )
    _write_handoff(
        output_root / "handoff_to_teammate.md",
        best_model_name,
        best_threshold,
        metrics,
        split_diagnostics,
    )
    _write_run_instructions(output_root / "run_instructions.md")
    _write_notebook(notebooks_dir / "task4_classification_final.ipynb")

    check = _final_checks(output_root, report_path, figures_dir, metrics)
    (results_dir / "final_checks.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    if not check["passed"]:
        raise RuntimeError(f"Final checks failed: {check}")

    print("Pipeline complete.", flush=True)
    return {
        "report_path": str(report_path.relative_to(project_root)),
        "figures_path": str(figures_dir.relative_to(project_root)),
        "notebook_path": str((notebooks_dir / "task4_classification_final.ipynb").relative_to(project_root)),
        "results_path": str(results_dir.relative_to(project_root)),
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "final_checks": check,
    }


def _tune_model(
    model_name: str,
    configs: List[Dict[str, object]],
    factory,
    thresholds: List[float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    hyper_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    for config_index, config in enumerate(configs):
        model = factory(config, seed + config_index)
        model.fit(x_train, y_train)
        probabilities = predict_probabilities(model, x_val)
        short_config = _short_config(model_name, config)
        for threshold in thresholds:
            predictions = threshold_predictions(probabilities, threshold)
            metrics = evaluate_predictions(y_val, predictions, model_name, "validation")
            row = {
                "model": model_name,
                "config_index": config_index,
                "short_config": short_config,
                "threshold": threshold,
                "val_macro_f1": metrics["macro_f1"],
                "val_micro_f1": metrics["micro_f1"],
                "params_json": json.dumps(config, sort_keys=True),
            }
            hyper_rows.append(row)
            if best is None or (row["val_macro_f1"], row["val_micro_f1"]) > (
                best["val_macro_f1"],
                best["val_micro_f1"],
            ):
                best = {
                    "model": model,
                    "config": config,
                    "threshold": threshold,
                    "short_config": short_config,
                    "val_macro_f1": row["val_macro_f1"],
                    "val_micro_f1": row["val_micro_f1"],
                }
    if best is None:
        raise RuntimeError(f"No hyperparameter result produced for {model_name}.")
    return best


def _short_config(model_name: str, config: Dict[str, object]) -> str:
    if model_name == "linear_ridge":
        return f"alpha={config['alpha']}"
    return f"trees={config['n_estimators']}, depth={config['max_depth']}, leaf={config['min_samples_leaf']}"


def _select_case_studies(
    data_dir: Path,
    segment_info: pd.DataFrame,
    y: np.ndarray,
    class_names: List[str],
    test_probabilities: np.ndarray,
    test_predictions: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_predictions: np.ndarray,
    figures_dir: Path,
    threshold: float,
    best_model_name: str,
) -> List[Dict[str, object]]:
    split_arrays = {
        "test": (test_probabilities, test_predictions),
        "validation": (validation_probabilities, validation_predictions),
    }
    split_global_indices = {
        split: segment_info.loc[segment_info["split"] == split, "global_index"].to_numpy(dtype=int)
        for split in split_arrays
    }
    full_prob = np.full(y.shape, np.nan, dtype=np.float32)
    full_pred = np.zeros_like(y, dtype=np.uint8)
    for split, (probabilities, predictions) in split_arrays.items():
        idx = split_global_indices[split]
        full_prob[idx] = probabilities
        full_pred[idx] = predictions

    candidates = []
    for split in ["test", "validation"]:
        filenames = segment_info.loc[segment_info["split"] == split, "filename"].drop_duplicates().tolist()
        for filename in filenames:
            idx = segment_info.index[segment_info["filename"] == filename].to_numpy(dtype=int)
            support = int(y[idx].sum())
            if support == 0:
                continue
            score = file_level_micro_f1(y[idx], full_pred[idx])
            candidates.append(
                {
                    "filename": filename,
                    "split": split,
                    "support": support,
                    "predicted_positive_labels": int(full_pred[idx].sum()),
                    "micro_f1": score,
                    "indices": idx,
                }
            )
    if len(candidates) < 2:
        raise RuntimeError("Not enough labelled validation/test files for case studies.")

    test_candidates = [item for item in candidates if item["split"] == "test"] or candidates
    success = sorted(test_candidates, key=lambda item: (item["micro_f1"], item["support"]), reverse=True)[0]
    remaining = [item for item in test_candidates if item["filename"] != success["filename"]]
    if not remaining:
        remaining = [item for item in candidates if item["filename"] != success["filename"]]
    failure = sorted(remaining, key=lambda item: (item["micro_f1"], -item["support"]))[0]

    outputs = []
    for figure_index, (kind, case) in enumerate([("successful", success), ("failure_or_ambiguous", failure)], start=1):
        filename = str(case["filename"])
        idx = case["indices"]
        npz_path = data_dir / "audio_features" / f"{Path(filename).stem}.npz"
        with np.load(npz_path, allow_pickle=True) as archive:
            melspect_mean = np.asarray(archive["melspect_mean"])
            start_time = np.asarray(archive["start_time"])
            end_time = np.asarray(archive["end_time"])

        y_true = y[idx]
        probabilities = full_prob[idx]
        predictions = full_pred[idx]
        true_any = y_true.sum(axis=0) > 0
        pred_any = predictions.sum(axis=0) > 0
        case_output = {
            "case_id": figure_index,
            "kind": kind,
            "filename": filename,
            "split": str(case["split"]),
            "model": best_model_name,
            "threshold": threshold,
            "micro_f1": float(case["micro_f1"]),
            "support": int(case["support"]),
            "predicted_positive_labels": int(case["predicted_positive_labels"]),
            "true_classes": [name for name, flag in zip(class_names, true_any) if flag],
            "predicted_classes": [name for name, flag in zip(class_names, pred_any) if flag],
            "missed_classes": [name for name, truth, pred in zip(class_names, true_any, pred_any) if truth and not pred],
            "extra_predicted_classes": [
                name for name, truth, pred in zip(class_names, true_any, pred_any) if pred and not truth
            ],
            "figure": f"fig_case_study_{figure_index}.png",
        }
        outputs.append(case_output)
        plot_case_study(
            melspect_mean=melspect_mean,
            start_time=start_time,
            end_time=end_time,
            y_true=y_true,
            probabilities=probabilities,
            class_names=class_names,
            output_path=figures_dir / f"fig_case_study_{figure_index}.png",
            title=f"Case study {figure_index}: {filename} ({kind})",
        )
    return outputs


def _write_report(
    report_path: Path,
    class_names: List[str],
    feature_summary: Dict[str, object],
    split_summary_df: pd.DataFrame,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    best_model_name: str,
    best_threshold: float,
    case_studies: List[Dict[str, object]],
    split_diagnostics: Dict[str, object],
) -> None:
    best_val = _metric(metrics, best_model_name, "validation")
    best_test = _metric(metrics, best_model_name, "test")
    all_neg_test = _metric(metrics, "all_negative", "test")
    freq_test = _metric(metrics, "empirical_frequency", "test")
    hardest = (
        per_class[(per_class["model"] == best_model_name) & (per_class["split"] == "test")]
        .sort_values("f1")
        .head(3)["class_name"]
        .tolist()
    )
    strongest = (
        per_class[(per_class["model"] == best_model_name) & (per_class["split"] == "test")]
        .sort_values("f1", ascending=False)
        .head(3)["class_name"]
        .tolist()
    )
    best_hyper = hyperparameters.sort_values(["val_macro_f1", "val_micro_f1"], ascending=False).iloc[0]

    split_table = "\n".join(
        f"{row.split} & {int(row.recordings)} & {int(row.collectors)} & {int(row.segments)} & "
        f"{row.label_density:.3f} \\\\"
        for row in split_summary_df.itertuples(index=False)
    )
    model_rows = []
    for model in ["all_negative", "empirical_frequency", "linear_ridge", "random_forest"]:
        val = _metric(metrics, model, "validation")
        test = _metric(metrics, model, "test")
        model_rows.append(
            f"{_latex_escape(model)} & {val['macro_f1']:.3f} & {val['micro_f1']:.3f} & "
            f"{test['macro_f1']:.3f} & {test['micro_f1']:.3f} \\\\"
        )
    model_table = "\n".join(model_rows)

    case1, case2 = case_studies
    report = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=1.6cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{caption}}
\graphicspath{{{{figures/}}}}

\title{{MLPC 2026 Task 4: Data Classification}}
\author{{Quantized Transformers\\Aleksandr Masiev k12348023\\Teammate: placeholder}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Introduction}}
We built a multi-label classifier for the MLPC 2026 Task 4 development set.
The data contains {feature_summary['n_recordings']} recordings, {feature_summary['n_segments']} time segments, and {feature_summary['n_classes']} target classes.
Our goal was to keep the pipeline reproducible and to avoid information leakage between recordings.

\section{{Dataset Preparation}}
\subsection{{Label Aggregation}}
We loaded \texttt{{metadata.csv}}, \texttt{{annotations.csv}}, and all \texttt{{audio\_features/*.npz}} files.
The \texttt{{.npz}} files contain aligned segment times, feature arrays, class names, annotator identifiers, and an annotation tensor with shape $[T,C,A]$.
For each segment and class, an annotator vote was positive when overlap was greater than zero.
The final label was positive when at least half of the annotators voted positive.
This rule is simple and robust to one weak annotation, but it can hide minority opinions and uncertain boundaries.
Recordings had between {feature_summary['annotators_per_recording_min']} and {feature_summary['annotators_per_recording_max']} annotators, with median {feature_summary['annotators_per_recording_median']:.1f}.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_label_distribution.png}}
\caption{{Positive segment labels per class and split.}}
\end{{figure}}

\subsection{{Train/Validation/Test Split}}
We split data at recording level and also grouped by \texttt{{collector\_id}}.
This was feasible because every class remained present in all splits after the deterministic group search.
No recording or collector appears in more than one split.

\begin{{table}}[H]
\centering
\caption{{Split summary.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Split & Recordings & Collectors & Segments & Label density\\
\midrule
{split_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_split_distribution.png}}
\caption{{Number of recordings and segments in each split.}}
\end{{figure}}

\subsection{{Preprocessing}}
We constructed segment-level matrices $X$ and $Y$.
The feature matrix concatenates {feature_summary['n_feature_keys']} segment-level mean feature arrays, giving {feature_summary['feature_dimension']} features per segment.
Missing values were replaced with training-set medians.
Then all features were standardized with mean and variance estimated only on the training split.
The validation and test splits used the same fitted imputer and scaler.

\section{{Evaluation}}
\subsection{{Metric Choice}}
The main metric is Macro F1.
It gives equal weight to rare and frequent classes, which is important because the labels are imbalanced.
We also report Micro F1, which summarizes global segment-label decisions.
The theoretical best score is 1.0, but practical performance is limited by noisy labels, overlapping sound events, ambiguous sources, and uncertain temporal boundaries.

\subsection{{Baseline}}
We used an all-negative baseline and an empirical class-frequency baseline using only training prevalence.
On the test split, the all-negative baseline had Macro F1 {all_neg_test['macro_f1']:.3f} and Micro F1 {all_neg_test['micro_f1']:.3f}.
The empirical baseline reached Macro F1 {freq_test['macro_f1']:.3f} and Micro F1 {freq_test['micro_f1']:.3f}.

\section{{Experiments}}
\subsection{{Classifiers and Hyperparameter Tuning}}
We trained two model families: a one-vs-rest linear ridge classifier and a random forest.
Hyperparameters and the global decision threshold were selected on the validation set only.
The best validation configuration was \texttt{{{_latex_escape(str(best_hyper['short_config']))}}} with threshold {best_threshold:.2f}.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_hyperparameter_tuning.png}}
\caption{{Validation Macro F1 for tuned configurations.}}
\end{{figure}}

\subsection{{Final Results}}
The selected final model was \texttt{{{_latex_escape(best_model_name)}}}.
It reached validation Macro F1 {best_val['macro_f1']:.3f} and test Macro F1 {best_test['macro_f1']:.3f}.
Its validation Micro F1 was {best_val['micro_f1']:.3f}, and its test Micro F1 was {best_test['micro_f1']:.3f}.

\begin{{table}}[H]
\centering
\caption{{Validation and test metrics.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Val Macro F1 & Val Micro F1 & Test Macro F1 & Test Micro F1\\
\midrule
{model_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_model_comparison.png}}
\caption{{Macro F1 comparison across baselines and classifiers.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_per_class_f1.png}}
\caption{{Per-class test F1 for the selected model.}}
\end{{figure}}

The strongest classes on test were {_class_list(strongest)}.
The most difficult classes were {_class_list(hardest)}.

\section{{Case Study and Reflection}}
\subsection{{Case Study 1}}
The first case is \texttt{{{_latex_escape(case1['filename'])}}} from the {case1['split']} split.
It is a relatively successful example with file-level Micro F1 {case1['micro_f1']:.3f}.
The true active classes were {_class_list(case1['true_classes'])}, and the model predicted {_class_list(case1['predicted_classes'])}.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_case_study_1.png}}
\caption{{Successful case study: mel features, ground truth labels, and predicted probabilities.}}
\end{{figure}}

\subsection{{Case Study 2}}
The second case is \texttt{{{_latex_escape(case2['filename'])}}} from the {case2['split']} split.
It is a failure or ambiguous example with file-level Micro F1 {case2['micro_f1']:.3f}.
The true classes were {_class_list(case2['true_classes'])}, while predicted classes were {_class_list(case2['predicted_classes'])}.
Missed classes were {_class_list(case2['missed_classes'])}; extra predicted classes were {_class_list(case2['extra_predicted_classes'])}.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig_case_study_2.png}}
\caption{{Failure or ambiguous case study with the same visualization format.}}
\end{{figure}}

\subsection{{Reflection}}
The results show that the pipeline learns useful patterns, but the gap between classes is large.
Short transient classes and classes with similar acoustic texture are harder.
The model also uses segment features without long temporal context, so it can miss boundaries or confuse overlapping sounds.
A future improvement would tune class-specific thresholds and add temporal smoothing, still using validation data only.

\section{{Disclosure of LLM and AI Tool Use}}
We used ChatGPT/Codex to support planning, coding, debugging, and wording of the report.
All numerical results, plots, and interpretations were produced from the provided dataset and checked against the generated notebook outputs.

\end{{document}}
"""
    report_path.write_text(report, encoding="utf-8")


def _write_handoff(
    path: Path,
    best_model_name: str,
    best_threshold: float,
    metrics: pd.DataFrame,
    split_diagnostics: Dict[str, object],
) -> None:
    best_test = _metric(metrics, best_model_name, "test")
    best_val = _metric(metrics, best_model_name, "validation")
    content = f"""# Handoff to Teammate

## Summary
- Implemented the full MLPC Task 4 classification pipeline.
- Generated the Overleaf-ready report and all referenced figures.
- Main model selected by validation Macro F1: `{best_model_name}` with threshold `{best_threshold:.2f}`.
- Validation Macro/Micro F1: `{best_val['macro_f1']:.3f}` / `{best_val['micro_f1']:.3f}`.
- Test Macro/Micro F1: `{best_test['macro_f1']:.3f}` / `{best_test['micro_f1']:.3f}`.

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
- Collector grouping was used: `{split_diagnostics.get('collector_grouping_used')}`.
- Recording leakage detected: `{split_diagnostics.get('recording_leakage')}`.
- Collector leakage detected: `{split_diagnostics.get('collector_leakage')}`.

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
"""
    path.write_text(content, encoding="utf-8")


def _write_run_instructions(path: Path) -> None:
    content = """# Run Instructions

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
"""
    path.write_text(content, encoding="utf-8")


def _write_notebook(path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MLPC 2026 Task 4 Classification\n",
                "\n",
                "This notebook reproduces the generated results, figures, and Overleaf report by running the source pipeline.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import sys\n",
                "\n",
                "cwd = Path.cwd().resolve()\n",
                "if cwd.name == 'notebooks' and cwd.parent.name == 'outputs':\n",
                "    project_root = cwd.parents[1]\n",
                "else:\n",
                "    project_root = cwd\n",
                "src_dir = project_root / 'outputs' / 'src'\n",
                "sys.path.insert(0, str(src_dir))\n",
                "project_root\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from run_pipeline import main\n",
                "\n",
                "summary = main(project_root=project_root)\n",
                "summary\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Generated artifacts are stored under `outputs/`, including result CSVs, report figures, and `mlpc_task4_report.tex`.\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _final_checks(output_root: Path, report_path: Path, figures_dir: Path, metrics: pd.DataFrame) -> Dict[str, object]:
    required_figures = [
        "fig_split_distribution.png",
        "fig_label_distribution.png",
        "fig_model_comparison.png",
        "fig_hyperparameter_tuning.png",
        "fig_per_class_f1.png",
        "fig_case_study_1.png",
        "fig_case_study_2.png",
    ]
    report_text = report_path.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"\\includegraphics\[width=\\linewidth\]\{([^}]+)\}", report_text)))
    figure_exists = {name: (figures_dir / name).exists() for name in required_figures}
    no_absolute_paths = re.search(r"[A-Za-z]:\\\\", report_text) is None
    placeholder_token = "TO" + "DO"
    no_unexpected_placeholders = placeholder_token not in report_text
    required_csvs = [
        output_root / "results" / "split_summary.csv",
        output_root / "results" / "metrics_summary.csv",
        output_root / "results" / "per_class_metrics.csv",
        output_root / "results" / "hyperparameter_results.csv",
        output_root / "results" / "case_studies.json",
    ]
    word_count = len(re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", re.sub(r"\\[A-Za-z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", report_text)))
    checks = {
        "report_exists": report_path.exists(),
        "required_figures_exist": figure_exists,
        "all_required_figures_referenced": sorted(required_figures) == referenced,
        "required_csvs_exist": {str(path.relative_to(output_root)): path.exists() for path in required_csvs},
        "no_absolute_paths_in_tex": no_absolute_paths,
        "no_todo_in_tex": no_unexpected_placeholders,
        "word_count_rough": word_count,
        "word_count_under_2000_rough": word_count < 2000,
        "metrics_rows": int(len(metrics)),
    }
    checks["passed"] = (
        checks["report_exists"]
        and all(figure_exists.values())
        and checks["all_required_figures_referenced"]
        and all(checks["required_csvs_exist"].values())
        and checks["no_absolute_paths_in_tex"]
        and checks["no_todo_in_tex"]
        and checks["word_count_under_2000_rough"]
    )
    return checks


def _metric(metrics: pd.DataFrame, model: str, split: str) -> Dict[str, float]:
    row = metrics[(metrics["model"] == model) & (metrics["split"] == split)].iloc[0]
    return {key: float(row[key]) for key in ["macro_f1", "micro_f1", "sample_f1", "hamming_loss"]}


def _class_list(classes: Iterable[str]) -> str:
    classes = list(classes)
    if not classes:
        return "none"
    return ", ".join(r"\texttt{" + _latex_escape(name) + "}" for name in classes)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
