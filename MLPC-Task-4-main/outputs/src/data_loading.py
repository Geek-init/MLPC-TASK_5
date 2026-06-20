"""Data loading and feature construction for MLPC 2026 Task 4."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_SUFFIXES = ("_mean",)
NON_FEATURE_KEYS = {
    "start_time",
    "end_time",
    "annotations",
    "is_own_recording",
    "class_names",
    "annotator_ids",
    "target_classes",
    "non_target_classes",
    "recording_device",
    "recording_environments",
    "scene_description",
    "device_placement",
}


def find_data_dir(project_root: Path) -> Path:
    """Return the expected local dataset directory."""
    data_dir = project_root / "data" / "MLPC2026_dataset_development"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return data_dir


def load_metadata_and_annotations(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata.csv and annotations.csv from the dataset directory."""
    metadata_path = data_dir / "metadata.csv"
    annotations_path = data_dir / "annotations.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not annotations_path.exists():
        raise FileNotFoundError(annotations_path)
    return pd.read_csv(metadata_path), pd.read_csv(annotations_path)


def list_feature_files(data_dir: Path) -> List[Path]:
    """List all feature archives in deterministic order."""
    feature_dir = data_dir / "audio_features"
    files = sorted(feature_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {feature_dir}")
    return files


def inspect_npz_structure(data_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Inspect a representative .npz file and save key, shape, and dtype."""
    first_file = list_feature_files(data_dir)[0]
    rows = []
    with np.load(first_file, allow_pickle=True) as archive:
        for key in archive.files:
            value = archive[key]
            rows.append(
                {
                    "example_file": first_file.name,
                    "key": key,
                    "shape": "x".join(str(dim) for dim in value.shape),
                    "dtype": str(value.dtype),
                    "is_feature_candidate": _is_feature_candidate(key, value, archive),
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def _is_feature_candidate(key: str, value: np.ndarray, archive: np.lib.npyio.NpzFile) -> bool:
    if key in NON_FEATURE_KEYS:
        return False
    if not key.endswith(DEFAULT_FEATURE_SUFFIXES):
        return False
    if not np.issubdtype(value.dtype, np.number):
        return False
    if value.ndim != 2:
        return False
    return value.shape[0] == len(archive["start_time"])


def infer_feature_keys(
    data_dir: Path,
    suffixes: Sequence[str] = DEFAULT_FEATURE_SUFFIXES,
) -> List[str]:
    """Infer segment-level numeric feature arrays to concatenate."""
    first_file = list_feature_files(data_dir)[0]
    feature_keys = []
    with np.load(first_file, allow_pickle=True) as archive:
        n_segments = len(archive["start_time"])
        for key in archive.files:
            value = archive[key]
            if key in NON_FEATURE_KEYS:
                continue
            if not key.endswith(tuple(suffixes)):
                continue
            if not np.issubdtype(value.dtype, np.number):
                continue
            if value.ndim == 2 and value.shape[0] == n_segments:
                feature_keys.append(key)
    if not feature_keys:
        raise RuntimeError("No feature arrays were inferred from the .npz structure.")
    return feature_keys


def _string_list(values: Iterable[object]) -> List[str]:
    return [str(value) for value in values]


def build_segment_dataset(
    data_dir: Path,
    aggregate_fn: Callable[[np.ndarray], np.ndarray],
    feature_keys: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Build segment-level X/Y matrices and matching metadata tables.

    The aligned annotation tensor inside each .npz file is used as the source
    for labels. The CSV files are still loaded and checked so the pipeline
    documents the full provided dataset.
    """
    metadata, annotations = load_metadata_and_annotations(data_dir)
    metadata = metadata.copy()
    metadata["filename"] = metadata["filename"].astype(str)
    metadata_by_file = metadata.set_index("filename", drop=False)
    csv_filenames = set(metadata["filename"])
    annotation_filenames = set(annotations["filename"].astype(str))

    files = list_feature_files(data_dir)
    if feature_keys is None:
        feature_keys = infer_feature_keys(data_dir)
    feature_keys = list(feature_keys)

    class_names: List[str] | None = None
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    segment_rows = []
    recording_rows = []

    for file_index, path in enumerate(files):
        filename = f"{path.stem}.wav"
        if filename not in csv_filenames:
            raise ValueError(f"{filename} exists as .npz but is missing from metadata.csv.")
        if filename not in annotation_filenames:
            raise ValueError(f"{filename} exists as .npz but is missing from annotations.csv.")

        with np.load(path, allow_pickle=True) as archive:
            start_time = np.asarray(archive["start_time"], dtype=np.float32)
            end_time = np.asarray(archive["end_time"], dtype=np.float32)
            annotations_tensor = np.asarray(archive["annotations"])
            labels = aggregate_fn(annotations_tensor)
            current_classes = _string_list(archive["class_names"])
            if class_names is None:
                class_names = current_classes
            elif current_classes != class_names:
                raise ValueError(f"Class order differs in {path.name}.")

            feature_blocks = []
            for key in feature_keys:
                value = np.asarray(archive[key], dtype=np.float32)
                if value.shape[0] != len(start_time):
                    raise ValueError(
                        f"Feature {key} in {path.name} has {value.shape[0]} rows, "
                        f"expected {len(start_time)}."
                    )
                feature_blocks.append(value.reshape(len(start_time), -1))
            features = np.concatenate(feature_blocks, axis=1).astype(np.float32)

            x_parts.append(features)
            y_parts.append(labels.astype(np.uint8))

            meta_row = metadata_by_file.loc[filename].to_dict()
            recording_row = {
                "filename": filename,
                "file_index": file_index,
                "collector_id": str(meta_row.get("collector_id", "unknown")),
                "recording_device": str(meta_row.get("recording_device", "")),
                "recording_environment": str(meta_row.get("recording_environment", "")),
                "device_placement": str(meta_row.get("device_placement", "")),
                "n_segments": int(labels.shape[0]),
                "duration": float(end_time[-1]) if len(end_time) else 0.0,
                "n_annotators": int(annotations_tensor.shape[2]),
                "positive_labels": int(labels.sum()),
                "segments_with_any_label": int((labels.sum(axis=1) > 0).sum()),
            }
            for class_idx, class_name in enumerate(class_names):
                recording_row[f"label_{class_name}"] = int(labels[:, class_idx].sum())
            recording_rows.append(recording_row)

            for local_idx in range(labels.shape[0]):
                segment_rows.append(
                    {
                        "global_index": len(segment_rows),
                        "filename": filename,
                        "file_index": file_index,
                        "segment_index": local_idx,
                        "start_time": float(start_time[local_idx]),
                        "end_time": float(end_time[local_idx]),
                    }
                )

    if class_names is None:
        raise RuntimeError("No class names were loaded.")

    x_matrix = np.vstack(x_parts).astype(np.float32)
    y_matrix = np.vstack(y_parts).astype(np.uint8)
    segment_info = pd.DataFrame(segment_rows)
    recording_info = pd.DataFrame(recording_rows)
    return x_matrix, y_matrix, segment_info, recording_info, feature_keys, class_names
