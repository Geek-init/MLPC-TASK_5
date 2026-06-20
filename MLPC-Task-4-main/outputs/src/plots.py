"""Matplotlib visualizations for the Task 4 report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, output_path: Path, dpi: int = 250) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    splits = summary["split"].tolist()
    axes[0].bar(splits, summary["recordings"], color="#4c78a8")
    axes[0].set_title("Recordings per split")
    axes[0].set_ylabel("Recordings")
    axes[1].bar(splits, summary["segments"], color="#59a14f")
    axes[1].set_title("Segments per split")
    axes[1].set_ylabel("Segments")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_label_distribution(summary: pd.DataFrame, class_names: Iterable[str], output_path: Path) -> None:
    class_names = list(class_names)
    class_cols = [f"label_{name}" for name in class_names]
    counts = summary.set_index("split")[class_cols].T
    counts.index = class_names
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(class_names))
    width = 0.25
    colors = {"train": "#4c78a8", "validation": "#f58518", "test": "#54a24b"}
    for i, split in enumerate(["train", "validation", "test"]):
        ax.bar(x + (i - 1) * width, counts[split], width, label=split, color=colors[split])
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Positive segment labels")
    ax.set_title("Label distribution by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    _save(fig, output_path)


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    subset = metrics[metrics["split"].isin(["validation", "test"])].copy()
    subset["label"] = subset["model"] + " (" + subset["split"] + ")"
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    colors = ["#4c78a8" if split == "validation" else "#59a14f" for split in subset["split"]]
    ax.bar(np.arange(len(subset)), subset["macro_f1"], color=colors)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(subset["label"], rotation=35, ha="right")
    ax.set_ylim(0, max(0.05, min(1.0, subset["macro_f1"].max() * 1.25)))
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_hyperparameter_tuning(hyperparameters: pd.DataFrame, output_path: Path) -> None:
    df = hyperparameters.copy().sort_values(["model", "val_macro_f1"], ascending=[True, False])
    df["config_label"] = df["model"] + "\n" + df["short_config"] + "\nthr=" + df["threshold"].astype(str)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    colors = df["model"].map({"linear_ridge": "#4c78a8", "random_forest": "#f58518"}).fillna("#777777")
    ax.bar(np.arange(len(df)), df["val_macro_f1"], color=colors)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["config_label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Validation-only hyperparameter tuning")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_per_class_f1(per_class: pd.DataFrame, best_model: str, output_path: Path) -> None:
    df = per_class[(per_class["model"] == best_model) & (per_class["split"] == "test")].copy()
    df = df.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    x = np.arange(len(df))
    ax.bar(x, df["f1"], color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test F1")
    ax.set_title(f"Per-class F1 for {best_model}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_case_study(
    melspect_mean: np.ndarray,
    start_time: np.ndarray,
    end_time: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    class_names = list(class_names)
    extent_time = [float(start_time[0]), float(end_time[-1])]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
    )
    axes[0].imshow(
        np.asarray(melspect_mean).T,
        aspect="auto",
        origin="lower",
        extent=[extent_time[0], extent_time[1], 0, melspect_mean.shape[1]],
        cmap="magma",
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(
        y_true.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[extent_time[0], extent_time[1], -0.5, len(class_names) - 0.5],
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    axes[1].set_ylabel("Ground truth")
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_yticklabels(class_names, fontsize=7)

    im = axes[2].imshow(
        probabilities.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[extent_time[0], extent_time[1], -0.5, len(class_names) - 0.5],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_ylabel("Prediction prob.")
    axes[2].set_yticks(np.arange(len(class_names)))
    axes[2].set_yticklabels(class_names, fontsize=7)
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.01)
    _save(fig, output_path)
