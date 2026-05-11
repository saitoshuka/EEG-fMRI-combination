#!/usr/bin/env python3
"""Render ROI query-time dependency on fsaverage5 cortical surfaces.

This is a visualization helper: it paints ROI-query time-window scores back onto
the same Destrieux fsaverage5 parcels used to extract the TRIBE visual ROI
targets, then exports per-window surface panels and optional GIFs.

It is attribution painted on fixed atlas parcels, not measured time-resolved
fMRI activity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from nilearn import datasets, plotting

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from extract_visual_roi_targets_from_tribe import (  # noqa: E402
    build_group_masks,
    build_roi_masks,
    load_destrieux_maps,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_QUERY_TIME = DEFAULT_RESULTS / "atm_roi_query_time_dependency"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_surface_time_maps"
DEFAULT_NILEARN_DIR = WORKSPACE / "cache" / "nilearn"
WINDOW_ORDER = [
    "w000_100",
    "w100_200",
    "w200_300",
    "w300_400",
    "w400_500",
    "w500_600",
    "w600_700",
    "w700_800",
    "w800_900",
    "w900_1000",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def window_label(window: str) -> str:
    if not window.startswith("w"):
        return window
    start, end = window[1:].split("_")
    return f"{int(start)}-{int(end)} ms"


def load_masks(roi_kind: str, data_dir: Path) -> tuple[np.ndarray, list[str]]:
    label_map, labels = load_destrieux_maps(data_dir)
    if roi_kind == "parcel":
        masks, names, _ = build_roi_masks(label_map, labels)
    elif roi_kind == "group":
        masks, names, _ = build_group_masks(label_map, labels)
    else:
        raise ValueError(f"Unknown roi_kind: {roi_kind}")
    return masks, names


def values_for_window(
    rows: list[dict[str, str]],
    run_name: str,
    roi_kind: str,
    condition: str,
    window: str,
    value_key: str,
) -> dict[str, float]:
    out = {}
    for row in rows:
        if row["run"] != run_name or row["roi_kind"] != roi_kind:
            continue
        if row["condition"] != condition or row["window"] != window:
            continue
        try:
            value = float(row[value_key])
        except ValueError:
            continue
        if np.isfinite(value):
            out[row["roi_name"]] = value
    return out


def paint_surface(values: dict[str, float], masks: np.ndarray, names: list[str]) -> np.ndarray:
    surf = np.full(20484, np.nan, dtype="float32")
    for idx, name in enumerate(names):
        if name not in values:
            continue
        surf[masks[idx]] = float(values[name])
    return surf


def split_hemi(surf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return surf[:10242], surf[10242:]


def finite_max(frames: list[np.ndarray], floor: float = 1e-6) -> float:
    vals = np.concatenate([frame[np.isfinite(frame)] for frame in frames if np.isfinite(frame).any()])
    if vals.size == 0:
        return 1.0
    return max(float(np.nanmax(np.abs(vals))), floor)


def render_panel(
    surf: np.ndarray,
    fsaverage: dict[str, str],
    title: str,
    out_path: Path,
    vmax: float,
    cmap: str,
    symmetric: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, right = split_hemi(surf)
    fig = plt.figure(figsize=(10.6, 8.0), dpi=170)
    panels = [
        ("left", "lateral", left, fsaverage["infl_left"], fsaverage["sulc_left"], "LH lateral"),
        ("left", "medial", left, fsaverage["infl_left"], fsaverage["sulc_left"], "LH medial"),
        ("right", "lateral", right, fsaverage["infl_right"], fsaverage["sulc_right"], "RH lateral"),
        ("right", "medial", right, fsaverage["infl_right"], fsaverage["sulc_right"], "RH medial"),
    ]
    for i, (hemi, view, hemi_map, mesh, bg, panel_title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        plotting.plot_surf_stat_map(
            mesh,
            hemi_map,
            hemi=hemi,
            view=view,
            bg_map=bg,
            axes=ax,
            figure=fig,
            colorbar=False,
            cmap=cmap,
            threshold=None,
            vmax=vmax,
            symmetric_cbar=symmetric,
        )
        ax.set_title(panel_title, fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.02, hspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_paths: list[Path], out_path: Path, duration_ms: int) -> None:
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(out_path, frames, duration=duration_ms / 1000.0)


def render_condition(
    rows: list[dict[str, str]],
    run_name: str,
    roi_kind: str,
    condition: str,
    value_key: str,
    masks: np.ndarray,
    names: list[str],
    fsaverage: dict[str, str],
    out_dir: Path,
    cmap: str,
    symmetric: bool,
    duration_ms: int,
) -> dict[str, object]:
    surfaces = []
    valid_windows = []
    for window in WINDOW_ORDER:
        values = values_for_window(rows, run_name, roi_kind, condition, window, value_key)
        if not values:
            continue
        valid_windows.append(window)
        surfaces.append(paint_surface(values, masks, names))
    vmax = finite_max(surfaces)
    frame_paths = []
    for window, surf in zip(valid_windows, surfaces):
        frame_path = out_dir / condition / f"{window}_{value_key}.png"
        title = f"{run_name}\n{condition} {window_label(window)} | {value_key}"
        render_panel(surf, fsaverage, title, frame_path, vmax=vmax, cmap=cmap, symmetric=symmetric)
        frame_paths.append(frame_path)
    gif_path = out_dir / f"{condition}_{value_key}.gif"
    if frame_paths:
        make_gif(frame_paths, gif_path, duration_ms)
    return {
        "condition": condition,
        "value_key": value_key,
        "n_frames": len(frame_paths),
        "vmax": vmax,
        "frames": [str(path) for path in frame_paths],
        "gif": str(gif_path) if frame_paths else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-time-csv", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--roi-kind", choices=["parcel", "group"], default="parcel")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--nilearn-data-dir", type=Path, default=DEFAULT_NILEARN_DIR)
    parser.add_argument("--duration-ms", type=int, default=650)
    args = parser.parse_args()

    rows = read_rows(args.query_time_csv)
    masks, names = load_masks(args.roi_kind, args.nilearn_data_dir)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=str(args.nilearn_data_dir))
    out_dir = args.out_dir / args.tag / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        render_condition(
            rows,
            args.run_name,
            args.roi_kind,
            "keep",
            "corr_signal",
            masks,
            names,
            fsaverage,
            out_dir,
            cmap="coolwarm",
            symmetric=True,
            duration_ms=args.duration_ms,
        ),
        render_condition(
            rows,
            args.run_name,
            args.roi_kind,
            "drop",
            "drop_delta_from_full",
            masks,
            names,
            fsaverage,
            out_dir,
            cmap="magma",
            symmetric=False,
            duration_ms=args.duration_ms,
        ),
    ]
    summary = {
        "query_time_csv": str(args.query_time_csv),
        "run_name": args.run_name,
        "roi_kind": args.roi_kind,
        "atlas": "destrieux_surface_fsaverage5",
        "note": "Visualization only: query-time attribution painted onto fixed atlas parcels.",
        "outputs": outputs,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
