#!/usr/bin/env python3
"""Time-window ablation for ATM ROI-query models.

The saved THING-EEG tensors contain 250 samples after dropping the 200 ms
pre-stimulus baseline, so the model input corresponds to 0-1000 ms at 250 Hz.
This script masks/keeps/drops post-stimulus windows and measures which windows
support different TRIBE visual ROI groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    subject_to_id,
    visual_group_features,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_BRANCH = DEFAULT_RESULTS / "atm_roi_spatial_branch"
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_temporal_hierarchy"

WINDOWS_MS = [
    ("w000_100", 0, 100),
    ("w100_200", 100, 200),
    ("w200_300", 200, 300),
    ("w300_500", 300, 500),
    ("w500_800", 500, 800),
    ("w800_1000", 800, 1000),
]

FINE_WINDOWS_MS = [
    (f"w{start:03d}_{start + 100}", start, start + 100)
    for start in range(0, 1000, 100)
]

GROUP_ORDER = [
    "all_visual",
    "early_calcarine",
    "medial_occipital",
    "lateral_occipital",
    "ventral_occipitotemporal",
    "inferior_temporal",
    "semantic_object_temporal",
    "early_visual_combined",
    "high_level_combined",
]


def strip_hemi(name: str) -> str:
    return name[3:] if name.startswith(("lh_", "rh_")) else name


def load_roi_payload(path: Path, roi_kind: str) -> dict[str, object]:
    payload = np.load(path, allow_pickle=True)
    target_key = "group_targets" if roi_kind == "group" else "parcel_targets"
    name_key = "group_names" if roi_kind == "group" else "parcel_names"
    count_key = "group_vertex_counts" if roi_kind == "group" else "parcel_vertex_counts"
    visual_group_json = (
        json.loads(str(payload["visual_group_json"].item()))
        if "visual_group_json" in payload.files
        else None
    )
    return {
        "targets": payload[target_key].astype("float32"),
        "names": payload[name_key],
        "vertex_counts": payload[count_key],
        "image_index": payload["image_index"].astype(int),
        "visual_group_json": visual_group_json,
    }


def build_model(summary: dict, roi_payload: dict[str, object], device: torch.device) -> AtmSemanticSpatial:
    roi_names = roi_payload["names"]
    vertex_counts = roi_payload["vertex_counts"]
    visual_json = json.dumps(roi_payload["visual_group_json"]) if roi_payload["visual_group_json"] else None
    group_features = visual_group_features(roi_names, visual_json)  # type: ignore[arg-type]
    model = AtmSemanticSpatial(
        roi_names=roi_names,  # type: ignore[arg-type]
        vertex_counts=vertex_counts,  # type: ignore[arg-type]
        group_features=group_features,
        num_subjects=10,
        subject_mode=summary.get("subject_mode", "none"),
        atm_d_model=int(summary.get("atm_d_model", 256)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=summary.get("semantic_head", "shallow"),
        use_spatial=True,
    ).to(device)
    return model


def roi_group_indices(names: np.ndarray, visual_group_json: dict[str, list[str]] | None) -> dict[str, list[int]]:
    names_s = [str(x) for x in names]
    groups: dict[str, set[int]] = {"all_visual": set(range(len(names_s)))}
    base_groups = GROUP_ORDER[1:7]
    for group in base_groups:
        groups[group] = set()

    for idx, name in enumerate(names_s):
        stripped = strip_hemi(name)
        for group in base_groups:
            if stripped == group:
                groups[group].add(idx)
            elif visual_group_json and stripped in set(visual_group_json.get(group, [])):
                groups[group].add(idx)

    groups["early_visual_combined"] = (
        groups["early_calcarine"] | groups["medial_occipital"] | groups["lateral_occipital"]
    )
    groups["high_level_combined"] = groups["inferior_temporal"] | groups["semantic_object_temporal"]
    return {key: sorted(value) for key, value in groups.items() if value}


def apply_condition(eeg: torch.Tensor, condition: str, start_ms: int | None, end_ms: int | None) -> torch.Tensor:
    if condition == "full":
        return eeg
    assert start_ms is not None and end_ms is not None
    start = int(round(start_ms * 250 / 1000))
    end = int(round(end_ms * 250 / 1000))
    out = torch.zeros_like(eeg) if condition == "keep" else eeg.clone()
    if condition == "keep":
        out[..., start:end] = eeg[..., start:end]
    elif condition == "drop":
        out[..., start:end] = 0
    else:
        raise ValueError(condition)
    return out


def predict_roi(
    model: AtmSemanticSpatial,
    eeg_stack: torch.Tensor,
    subjects: list[str],
    device: torch.device,
    batch_size: int,
    condition: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> np.ndarray:
    model.eval()
    subject_preds = []
    with torch.no_grad():
        for subject_idx, subject in enumerate(subjects):
            eeg = apply_condition(eeg_stack[subject_idx], condition, start_ms, end_ms)
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            preds = []
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                out = model(x, sids)
                preds.append(out["roi_pred"].cpu())
            subject_preds.append(torch.cat(preds, dim=0))
    return torch.stack(subject_preds, dim=0).mean(dim=0).numpy()


def col_corr(pred: np.ndarray, target: np.ndarray, idx: list[int]) -> float:
    x = pred[:, idx].astype("float64")
    y = target[:, idx].astype("float64")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    corr = (x * y).sum(axis=0) / np.maximum(denom, 1e-12)
    return float(np.nanmean(corr))


def retrieval_metrics(pred: np.ndarray, target: np.ndarray, idx: list[int]) -> dict[str, float]:
    x = pred[:, idx].astype("float64")
    y = target[:, idx].astype("float64")
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-8)
    sims = x @ y.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "rank": float((1.0 - (ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "diag_minus_offdiag": float(diag.mean() - sims[~np.eye(len(pred), dtype=bool)].mean()),
    }


def metric_rows(
    pred: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[int]],
    run_name: str,
    roi_kind: str,
    condition: str,
    window: str,
    start_ms: int | None,
    end_ms: int | None,
) -> list[dict[str, float | int | str]]:
    shifted = np.roll(target, len(target) // 3, axis=0)
    rows = []
    for group, idx in groups.items():
        metrics = retrieval_metrics(pred, target, idx)
        shifted_metrics = retrieval_metrics(pred, shifted, idx)
        corr = col_corr(pred, target, idx)
        shifted_corr = col_corr(pred, shifted, idx)
        rows.append(
            {
                "run": run_name,
                "roi_kind": roi_kind,
                "condition": condition,
                "window": window,
                "start_ms": -1 if start_ms is None else start_ms,
                "end_ms": -1 if end_ms is None else end_ms,
                "roi_group": group,
                "n_roi": len(idx),
                "stim_corr": corr,
                "shifted_stim_corr": shifted_corr,
                "stim_corr_signal": corr - shifted_corr,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "rank": metrics["rank"],
                "diag_minus_offdiag": metrics["diag_minus_offdiag"],
                "shifted_rank": shifted_metrics["rank"],
                "rank_signal": metrics["rank"] - shifted_metrics["rank"],
            }
        )
    return rows


def write_heatmap(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    condition: str,
    value_key: str,
    title: str,
    windows_ms: list[tuple[str, int, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = [name for name, _, _ in windows_ms]
    groups = [g for g in GROUP_ORDER if any(r["roi_group"] == g for r in rows)]
    mat = np.full((len(groups), len(windows)), np.nan)
    for i, group in enumerate(groups):
        for j, window in enumerate(windows):
            vals = [
                float(r[value_key])
                for r in rows
                if r["condition"] == condition and r["roi_group"] == group and r["window"] == window
            ]
            if vals:
                mat[i, j] = vals[0]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=170)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([w.replace("w", "").replace("_", "-") for w in windows], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_best_windows(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    run_name: str,
) -> None:
    full_by_group = {
        str(r["roi_group"]): float(r["stim_corr_signal"])
        for r in rows
        if r["condition"] == "full" and r["window"] == "full"
    }
    out_rows = []
    for group in [g for g in GROUP_ORDER if g in full_by_group]:
        keep = [r for r in rows if r["condition"] == "keep" and r["roi_group"] == group]
        drop = [r for r in rows if r["condition"] == "drop" and r["roi_group"] == group]
        best_keep = max(keep, key=lambda r: float(r["stim_corr_signal"])) if keep else None
        worst_drop = min(drop, key=lambda r: float(r["stim_corr_signal"])) if drop else None
        if best_keep and worst_drop:
            out_rows.append(
                {
                    "run": run_name,
                    "roi_group": group,
                    "full_signal_corr": full_by_group[group],
                    "best_keep_window": best_keep["window"],
                    "best_keep_signal_corr": best_keep["stim_corr_signal"],
                    "most_important_drop_window": worst_drop["window"],
                    "drop_delta_signal_corr": full_by_group[group] - float(worst_drop["stim_corr_signal"]),
                }
            )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="n4096")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-mode", choices=["coarse", "fine100"], default="coarse")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_ms = FINE_WINDOWS_MS if args.window_mode == "fine100" else WINDOWS_MS

    all_rows: list[dict[str, float | int | str]] = []
    summaries = []
    for run_dir in args.runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        roi_kind = str(summary["roi_kind"])
        run_name = run_dir.name
        test_roi_path = Path(summary["test_roi"])
        if not test_roi_path.is_absolute():
            test_roi_path = WORKSPACE / test_roi_path
        train_roi_path = Path(summary["train_roi"])
        if not train_roi_path.is_absolute():
            train_roi_path = WORKSPACE.parent / train_roi_path

        test_payload = load_roi_payload(test_roi_path, roi_kind)
        train_payload = load_roi_payload(train_roi_path, roi_kind)
        groups = roi_group_indices(
            test_payload["names"],  # type: ignore[arg-type]
            test_payload["visual_group_json"],  # type: ignore[arg-type]
        )

        model = build_model(summary, train_payload, device)
        state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        subjects = list(summary["subjects"])
        eeg_stack = load_or_build_test_eeg_stack(
            args.data_root,
            subjects,
            test_payload["image_index"],  # type: ignore[arg-type]
            cache_dir=args.cache_dir,
            cache_tag=test_roi_path.stem,
        )
        assert eeg_stack is not None
        target = test_payload["targets"]  # type: ignore[assignment]

        conditions: list[tuple[str, str, int | None, int | None]] = [("full", "full", None, None)]
        for name, start_ms, end_ms in windows_ms:
            conditions.append(("keep", name, start_ms, end_ms))
            conditions.append(("drop", name, start_ms, end_ms))

        run_rows = []
        for condition, window, start_ms, end_ms in conditions:
            pred = predict_roi(
                model,
                eeg_stack,
                subjects,
                device,
                args.batch_size,
                condition,
                start_ms,
                end_ms,
            )
            run_rows.extend(
                metric_rows(
                    pred,
                    target,  # type: ignore[arg-type]
                    groups,
                    run_name,
                    roi_kind,
                    condition,
                    window,
                    start_ms,
                    end_ms,
                )
            )
        all_rows.extend(run_rows)

        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        write_heatmap(
            run_rows,
            run_out / "keep_window_stim_corr_signal.png",
            "keep",
            "stim_corr_signal",
            f"{run_name}: keep-window ROI signal corr",
            windows_ms,
        )
        write_heatmap(
            run_rows,
            run_out / "drop_window_stim_corr_signal.png",
            "drop",
            "stim_corr_signal",
            f"{run_name}: drop-window ROI signal corr",
            windows_ms,
        )
        write_best_windows(run_rows, run_out / "best_windows.csv", run_name)
        summaries.append({"run": run_name, "roi_kind": roi_kind, "groups": {k: len(v) for k, v in groups.items()}})

    csv_path = out_dir / "temporal_hierarchy_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        keys = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "data_root": str(args.data_root),
                "device": str(device),
                "time_axis": "250 samples = 0-1000 ms post-stimulus after dropping 50 baseline samples",
                "window_mode": args.window_mode,
                "windows_ms": windows_ms,
                "runs": summaries,
                "metrics_csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((out_dir / "summary.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
