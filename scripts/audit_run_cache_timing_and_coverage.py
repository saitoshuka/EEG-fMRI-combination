#!/usr/bin/env python3
"""Audit derived EEG-fMRI run caches for timing and target coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOTS = [
    Path("data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache"),
    Path("data/pooled_raw_schaefer100_neurostorm_official_natview_labram_eeg/run_cache"),
]


def scalar(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return str(arr.item()) if arr.shape == () else str(arr)


def per_file_row(path: Path, dataset: str) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)
    subject = scalar(z, "subject", path.stem.split("_")[0])
    run = scalar(z, "run", path.stem)
    task = scalar(z, "task", "")
    sfreq = float(np.asarray(z["sfreq"]).item()) if "sfreq" in z.files else np.nan
    n_windows = int(z["Y"].shape[0]) if "Y" in z.files else int(z["sample_start"].shape[0])
    sample_time = z["sample_time"].astype(np.float64) if "sample_time" in z.files else None
    if sample_time is not None and sample_time.size > 1:
        diffs = np.diff(sample_time)
        step_median = float(np.nanmedian(diffs))
        step_min = float(np.nanmin(diffs))
        step_max = float(np.nanmax(diffs))
        time_min = float(np.nanmin(sample_time))
        time_max = float(np.nanmax(sample_time))
    else:
        step_median = step_min = step_max = time_min = time_max = np.nan
    if "Y_mask" in z.files:
        coverage = z["Y_mask"].astype(np.float32).sum(axis=1)
        roi_mean = float(np.nanmean(coverage))
        roi_min = float(np.nanmin(coverage))
        roi_ge95 = float(np.mean(coverage >= 95.0))
    elif "Y" in z.files:
        roi_mean = roi_min = float(z["Y"].shape[1])
        roi_ge95 = 1.0
    else:
        roi_mean = roi_min = roi_ge95 = np.nan
    return {
        "dataset": dataset,
        "file": str(path),
        "subject": subject,
        "run": run,
        "task": task,
        "sfreq": sfreq,
        "n_windows": n_windows,
        "sample_step_median": step_median,
        "sample_step_min": step_min,
        "sample_step_max": step_max,
        "sample_time_min": time_min,
        "sample_time_max": time_max,
        "roi_valid_mean": roi_mean,
        "roi_valid_min": roi_min,
        "roi_ge95_fraction": roi_ge95,
        "has_neurostorm": "Z_neurostorm" in z.files,
    }


def dataset_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return root.name


def collect_rows(roots: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            continue
        files = sorted(root.rglob("*.npz"))
        for path in files:
            rows.append(per_file_row(path, dataset_from_path(root, path)))
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds = [row for row in rows if row["dataset"] == dataset]

        def vals(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in ds if np.isfinite(float(row[key]))], dtype=np.float64)

        steps = vals("sample_step_median")
        roi_mean = vals("roi_valid_mean")
        roi_min = vals("roi_valid_min")
        ge95 = vals("roi_ge95_fraction")
        out.append(
            {
                "dataset": dataset,
                "n_runs": len(ds),
                "n_subjects": len({str(row["subject"]) for row in ds}),
                "n_windows": int(sum(int(row["n_windows"]) for row in ds)),
                "tasks": "|".join(sorted({str(row["task"]) for row in ds if str(row["task"])})),
                "sfreq_values": "|".join(sorted({f"{float(row['sfreq']):g}" for row in ds if np.isfinite(float(row["sfreq"]))})),
                "sample_step_median": float(np.nanmedian(steps)) if steps.size else np.nan,
                "sample_step_min": float(np.nanmin(steps)) if steps.size else np.nan,
                "sample_step_max": float(np.nanmax(steps)) if steps.size else np.nan,
                "roi_valid_mean": float(np.nanmean(roi_mean)) if roi_mean.size else np.nan,
                "roi_valid_min": float(np.nanmin(roi_min)) if roi_min.size else np.nan,
                "roi_ge95_fraction": float(np.nanmean(ge95)) if ge95.size else np.nan,
                "neurostorm_runs": int(sum(bool(row["has_neurostorm"]) for row in ds)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict[str, object]], per_file_csv: Path) -> None:
    lines = [
        "# Run Cache Timing And Coverage Audit",
        "",
        f"- Per-file CSV: `{per_file_csv}`",
        "",
        "| dataset | runs | subjects | windows | step sec | ROI mean | ROI min | ROI >=95 | tasks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['n_runs']} | {row['n_subjects']} | {row['n_windows']} | "
            f"{float(row['sample_step_median']):.3g} | {float(row['roi_valid_mean']):.1f} | "
            f"{float(row['roi_valid_min']):.1f} | {float(row['roi_ge95_fraction']):.2f} | {row['tasks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_source_metadata_audit"))
    args = parser.parse_args()

    roots = args.root or DEFAULT_ROOTS
    rows = collect_rows(roots)
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_cache_quality_per_file.csv", rows)
    write_csv(args.output_dir / "run_cache_quality_summary.csv", summary)
    write_report(args.output_dir / "run_cache_quality_report.md", summary, args.output_dir / "run_cache_quality_per_file.csv")
    (args.output_dir / "run_cache_quality_summary.json").write_text(
        json.dumps({"roots": [str(root) for root in roots], "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print({"runs": len(rows), "datasets": len(summary), "output_dir": str(args.output_dir)}, flush=True)


if __name__ == "__main__":
    main()
