#!/usr/bin/env python3
"""Aggregate THINGS-fMRI voxel-wise beta responses into image x ROI matrices.

The ds004192 ICA-beta derivative stores subject-specific voxel response
matrices. For a first external validation against EEG-derived cortical targets,
we average voxel betas inside metadata-defined ROIs, then average those ROI
responses across fMRI subjects for exact THINGS-EEG/THINGS-fMRI image matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_ROOT = WORKSPACE / "data" / "things_fmri" / "ds004192_ica_betas"
DEFAULT_OVERLAP = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "things_fmri_same_image_overlap.csv"
)
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
)


LIKELY_ROI_COLUMNS = [
    "roi",
    "ROI",
    "roi_name",
    "ROI_name",
    "region",
    "Region",
    "label",
    "Label",
    "atlas_label",
    "visual_area",
    "brain_area",
    "area",
]
BINARY_ROI_SENTINEL = "__binary_roi_columns__"
EXCLUDED_BINARY_COLUMNS = {
    "subject_id",
}


def read_flexible_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=",")
    if table.shape[1] == 1:
        table = pd.read_csv(path, sep="\t")
    return table


def find_h5_dataset(path: Path, n_rows: int, requested: str | None) -> tuple[str, bool]:
    with h5py.File(path, "r") as handle:
        if requested:
            if requested not in handle:
                raise KeyError(f"{requested} not found in {path}")
            dataset = handle[requested]
            if dataset.ndim != 2:
                raise ValueError(f"{requested} is not a 2D matrix: shape={dataset.shape}")
            if dataset.shape[0] == n_rows:
                return requested, False
            if dataset.shape[1] == n_rows:
                return requested, True
            raise ValueError(f"{requested} shape {dataset.shape} does not match n_rows={n_rows}")

        candidates: list[tuple[str, bool, tuple[int, int]]] = []

        def visitor(name: str, obj: h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
                return
            if obj.shape[0] == n_rows:
                candidates.append((name, False, tuple(obj.shape)))
            elif obj.shape[1] == n_rows:
                candidates.append((name, True, tuple(obj.shape)))

        handle.visititems(visitor)
    if not candidates:
        raise ValueError(f"No 2D numeric H5 dataset in {path} matches n_rows={n_rows}")
    candidates = sorted(candidates, key=lambda item: max(item[2]), reverse=True)
    return candidates[0][0], candidates[0][1]


def find_binary_roi_columns(voxel_meta: pd.DataFrame, max_rois: int) -> list[str]:
    roi_columns: list[str] = []
    for column in voxel_meta.columns:
        if column in EXCLUDED_BINARY_COLUMNS:
            continue
        series = pd.to_numeric(voxel_meta[column], errors="coerce")
        values = set(series.dropna().unique().tolist())
        if not values:
            continue
        if values.issubset({0, 1, 0.0, 1.0}) and float(series.sum()) > 0:
            roi_columns.append(column)
    if not roi_columns:
        raise ValueError("No binary ROI mask columns found in voxel metadata.")
    if len(roi_columns) > max_rois:
        raise ValueError(
            f"Found {len(roi_columns)} binary ROI columns; increase --max-rois "
            "or pass a narrower --roi-column."
        )
    return roi_columns


def choose_roi_source(voxel_meta: pd.DataFrame, requested: str, max_rois: int) -> tuple[str, list[str]]:
    if requested == "binary-columns":
        return BINARY_ROI_SENTINEL, find_binary_roi_columns(voxel_meta, max_rois)
    if requested != "auto":
        if requested not in voxel_meta.columns:
            raise KeyError(f"Requested ROI column {requested!r} not in voxel metadata")
        values = voxel_meta[requested].dropna().astype(str)
        roi_names = sorted(values.unique().tolist())
        return requested, roi_names
    for column in LIKELY_ROI_COLUMNS:
        if column in voxel_meta.columns:
            values = voxel_meta[column].dropna().astype(str)
            if 2 <= values.nunique() <= 300:
                return column, sorted(values.unique().tolist())
    candidates = []
    for column in voxel_meta.columns:
        values = voxel_meta[column].dropna()
        if values.empty:
            continue
        if values.dtype == object:
            n_unique = values.astype(str).nunique()
            if 2 <= n_unique <= 300:
                candidates.append((column, n_unique))
    if candidates:
        column = sorted(candidates, key=lambda item: item[1])[0][0]
        values = voxel_meta[column].dropna().astype(str)
        return column, sorted(values.unique().tolist())
    return BINARY_ROI_SENTINEL, find_binary_roi_columns(voxel_meta, max_rois)


def iter_chunks(indices: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def compute_roi_means(
    h5_path: Path,
    dataset_name: str,
    transpose: bool,
    row_indices: np.ndarray,
    voxel_meta: pd.DataFrame,
    roi_column: str,
    roi_source: str,
    roi_names: list[str],
    chunk_size: int,
) -> np.ndarray:
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    if roi_source == BINARY_ROI_SENTINEL:
        roi_masks = [
            np.flatnonzero(pd.to_numeric(voxel_meta[roi], errors="coerce").fillna(0).to_numpy() > 0.5)
            for roi in roi_names
        ]
    else:
        roi_masks = [
            np.flatnonzero(voxel_meta[roi_source].fillna("").astype(str).to_numpy() == roi)
            for roi in roi_names
        ]
    sorted_out = np.full((len(sorted_rows), len(roi_names)), np.nan, dtype=np.float32)
    with h5py.File(h5_path, "r") as handle:
        dataset = handle[dataset_name]
        for chunk in iter_chunks(np.arange(len(sorted_rows)), chunk_size):
            rows = sorted_rows[chunk]
            if transpose:
                block = np.asarray(dataset[:, rows], dtype=np.float32).T
            else:
                block = np.asarray(dataset[rows, :], dtype=np.float32)
            for roi_index, mask in enumerate(roi_masks):
                if len(mask) == 0:
                    continue
                sorted_out[chunk, roi_index] = np.nanmean(block[:, mask], axis=1)
    return sorted_out[inverse]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument("--overlap-csv", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--roi-column", default="auto")
    parser.add_argument("--h5-dataset", default=None)
    parser.add_argument("--max-rois", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    overlap = pd.read_csv(args.overlap_csv)
    overlap = overlap.sort_values(["split", "image_index", "image_file"]).reset_index(drop=True)
    image_files = overlap["image_file"].astype(str).to_numpy()

    subject_dirs = sorted((args.fmri_root / "derivatives" / "ICA-betas").glob("sub-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found in {args.fmri_root}")

    subject_mats = []
    report: dict[str, object] = {"subjects": []}
    roi_names: list[str] | None = None
    roi_source_used: str | None = None

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        meta_dir = subject_dir / "voxel-metadata"
        stim_path = meta_dir / f"{subject}_task-things_stimulus-metadata.tsv"
        voxel_path = meta_dir / f"{subject}_task-things_voxel-metadata.tsv"
        h5_path = meta_dir / f"{subject}_task-things_voxel-wise-responses.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing H5 response matrix: {h5_path}")
        stim = read_flexible_table(stim_path).reset_index().rename(columns={"index": "row_index"})
        stim["image_file"] = stim["stimulus"].astype(str).map(lambda value: Path(value).name)
        merged = pd.DataFrame({"image_file": image_files}).merge(
            stim[["image_file", "row_index"]], on="image_file", how="left"
        )
        if merged["row_index"].isna().any():
            missing = merged.loc[merged["row_index"].isna(), "image_file"].head(10).tolist()
            raise ValueError(f"{subject} missing overlap images, examples: {missing}")
        row_indices = merged["row_index"].to_numpy(dtype=np.int64)

        voxel_meta = read_flexible_table(voxel_path)
        roi_source, current_roi_names = choose_roi_source(voxel_meta, args.roi_column, args.max_rois)
        if roi_source_used is None:
            roi_source_used = roi_source
        elif roi_source != roi_source_used:
            raise ValueError(f"ROI source mismatch: {roi_source_used} vs {roi_source}")
        if roi_names is None:
            roi_names = current_roi_names
        elif roi_names != current_roi_names:
            raise ValueError("ROI names differ across subjects; use a shared metadata column.")

        dataset_name, transpose = find_h5_dataset(h5_path, len(stim), args.h5_dataset)
        mat = compute_roi_means(
            h5_path=h5_path,
            dataset_name=dataset_name,
            transpose=transpose,
            row_indices=row_indices,
            voxel_meta=voxel_meta,
            roi_source=roi_source,
            roi_names=roi_names,
            chunk_size=args.chunk_size,
        )
        subject_mats.append(mat)
        report["subjects"].append(
            {
                "subject": subject,
                "h5_path": str(h5_path),
                "dataset_name": dataset_name,
                "transpose": transpose,
                "n_stimulus_rows": int(len(stim)),
                "n_voxels": int(len(voxel_meta)),
                "roi_source": roi_source,
                "n_rois": int(len(roi_names)),
            }
        )

    subject_betas = np.stack(subject_mats, axis=0)
    mean_betas = np.nanmean(subject_betas, axis=0).astype(np.float32)
    roi_names_array = np.asarray(roi_names or [], dtype=object)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "things_fmri_roi_betas_subject_averaged.npz"
    np.savez_compressed(
        out_path,
        measured_roi_beta=mean_betas,
        subject_roi_beta=subject_betas.astype(np.float32),
        roi_names=roi_names_array,
        roi_source=np.asarray(roi_source_used),
        image_file=image_files,
        split=overlap["split"].astype(str).to_numpy(),
        image_index=overlap["image_index"].to_numpy(),
        concept=overlap["concept"].astype(str).to_numpy(),
        subjects=np.asarray([path.name for path in subject_dirs], dtype=object),
    )
    report["out_path"] = str(out_path)
    report["n_images"] = int(len(image_files))
    report["n_train"] = int((overlap["split"] == "train").sum())
    report["n_test"] = int((overlap["split"] == "test").sum())
    report["roi_names"] = list(roi_names or [])
    report_path = args.out_dir / "things_fmri_roi_betas_subject_averaged_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
