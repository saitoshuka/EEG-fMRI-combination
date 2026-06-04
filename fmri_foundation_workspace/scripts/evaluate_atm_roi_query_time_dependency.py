#!/usr/bin/env python3
"""Query-level time-window dependency for ATM ROI-query models.

This is finer than the group-level temporal hierarchy script. It keeps the
ordered ROI supervision contract:

    query i -> ROI target i

and measures whether each learned ROI query depends on different EEG time
windows. For each query and each 100 ms window, the script writes keep/drop
Pearson signal against the paired pseudo-cortical target and a shifted-target
null. Parcel-level runs also get aggregated back into visual ROI groups for
readable figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from evaluate_atm_roi_temporal_hierarchy import (  # noqa: E402
    FINE_WINDOWS_MS,
    apply_condition,
    load_roi_payload,
    roi_group_indices,
)
from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    roi_query_features,
    subject_to_id,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_DATA_ROOT = ROOT / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_query_time_dependency"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, WORKSPACE):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def strip_hemi(name: str) -> str:
    return name[3:] if name.startswith(("lh_", "rh_")) else name


def primary_group_for_name(name: str, visual_group_json: dict[str, list[str]] | None) -> str:
    stripped = strip_hemi(name)
    if visual_group_json:
        for group, parcels in visual_group_json.items():
            if stripped in set(parcels):
                return group
    return stripped


def build_model(summary: dict, roi_payload: dict[str, object], device: torch.device) -> AtmSemanticSpatial:
    visual_json = json.dumps(roi_payload["visual_group_json"]) if roi_payload["visual_group_json"] else None
    metadata = summary.get("prototype_metadata_roi") or None
    metadata_path = resolve_path(metadata) if metadata else None
    group_features = roi_query_features(
        roi_payload["names"],  # type: ignore[arg-type]
        visual_json,
        feature_mode=summary.get("roi_feature_mode", "group"),
        prototype_metadata_roi=metadata_path,
    )
    model = AtmSemanticSpatial(
        roi_names=roi_payload["names"],  # type: ignore[arg-type]
        vertex_counts=roi_payload["vertex_counts"],  # type: ignore[arg-type]
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
        spatial_head=summary.get("spatial_head", "query"),
    ).to(device)
    return model


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
                preds.append(model(x, sids)["roi_pred"].cpu())
            subject_preds.append(torch.cat(preds, dim=0))
    return torch.stack(subject_preds, dim=0).mean(dim=0).numpy()


def pearson_1d(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype("float64")
    y = y.astype("float64")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < eps:
        return 0.0
    return float((x * y).sum() / denom)


def query_rows(
    pred: np.ndarray,
    target: np.ndarray,
    names: np.ndarray,
    visual_group_json: dict[str, list[str]] | None,
    run_name: str,
    roi_kind: str,
    condition: str,
    window: str,
    start_ms: int | None,
    end_ms: int | None,
    full_signal_by_query: dict[int, float],
) -> list[dict[str, float | int | str]]:
    shifted = np.roll(target, len(target) // 3, axis=0)
    rows: list[dict[str, float | int | str]] = []
    for idx, name in enumerate(names):
        corr = pearson_1d(pred[:, idx], target[:, idx])
        shifted_corr = pearson_1d(pred[:, idx], shifted[:, idx])
        signal = corr - shifted_corr
        full_signal = full_signal_by_query.get(idx, signal)
        rows.append(
            {
                "run": run_name,
                "roi_kind": roi_kind,
                "condition": condition,
                "window": window,
                "start_ms": -1 if start_ms is None else start_ms,
                "end_ms": -1 if end_ms is None else end_ms,
                "query_index": idx,
                "roi_name": str(name),
                "roi_group": primary_group_for_name(str(name), visual_group_json),
                "corr": corr,
                "shifted_corr": shifted_corr,
                "corr_signal": signal,
                "full_corr_signal": full_signal,
                "drop_delta_from_full": full_signal - signal if condition == "drop" else np.nan,
            }
        )
    return rows


def aggregate_rows(
    rows: list[dict[str, float | int | str]],
    groups: dict[str, list[int]],
) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    keys = sorted({(r["run"], r["roi_kind"], r["condition"], r["window"]) for r in rows})
    by_query = {(r["condition"], r["window"], int(r["query_index"])): r for r in rows}
    for run, roi_kind, condition, window in keys:
        group_source = [r for r in rows if (r["run"], r["roi_kind"], r["condition"], r["window"]) == (run, roi_kind, condition, window)]
        if not group_source:
            continue
        start_ms = group_source[0]["start_ms"]
        end_ms = group_source[0]["end_ms"]
        for group, idxs in groups.items():
            vals = [
                float(by_query[(condition, window, idx)]["corr_signal"])
                for idx in idxs
                if (condition, window, idx) in by_query
            ]
            drops = [
                float(by_query[(condition, window, idx)]["drop_delta_from_full"])
                for idx in idxs
                if (condition, window, idx) in by_query
                and not np.isnan(float(by_query[(condition, window, idx)]["drop_delta_from_full"]))
            ]
            if not vals:
                continue
            out.append(
                {
                    "run": run,
                    "roi_kind": roi_kind,
                    "condition": condition,
                    "window": window,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "roi_group": group,
                    "n_query": len(vals),
                    "mean_corr_signal": float(np.mean(vals)),
                    "sem_corr_signal": float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "mean_drop_delta_from_full": float(np.mean(drops)) if drops else np.nan,
                }
            )
    return out


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_rows(
    rows: Iterable[dict[str, float | int | str]],
    y_key: str,
    y_values: list[str],
    windows: list[str],
    value_key: str,
    condition: str,
) -> np.ndarray:
    mat = np.full((len(y_values), len(windows)), np.nan)
    row_list = list(rows)
    for i, y in enumerate(y_values):
        for j, window in enumerate(windows):
            vals = [
                float(r[value_key])
                for r in row_list
                if r[y_key] == y and r["window"] == window and r["condition"] == condition
            ]
            if vals:
                mat[i, j] = float(np.nanmean(vals))
    return mat


def write_heatmap(
    mat: np.ndarray,
    y_labels: list[str],
    windows: list[str],
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = max(3.8, 0.28 * len(y_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, height), dpi=170)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([w.replace("w", "").replace("_", "-") for w in windows], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7 if len(y_labels) > 20 else 9)
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_best_windows(
    query_metrics: list[dict[str, float | int | str]],
    aggregate_metrics: list[dict[str, float | int | str]],
    out_dir: Path,
) -> None:
    query_best = []
    for query_idx in sorted({int(r["query_index"]) for r in query_metrics}):
        rows = [r for r in query_metrics if int(r["query_index"]) == query_idx]
        keep = [r for r in rows if r["condition"] == "keep"]
        drop = [r for r in rows if r["condition"] == "drop"]
        full = next((r for r in rows if r["condition"] == "full"), None)
        if not keep or not drop or not full:
            continue
        best_keep = max(keep, key=lambda r: float(r["corr_signal"]))
        most_drop = max(drop, key=lambda r: float(r["drop_delta_from_full"]))
        query_best.append(
            {
                "query_index": query_idx,
                "roi_name": full["roi_name"],
                "roi_group": full["roi_group"],
                "full_corr_signal": full["corr_signal"],
                "best_keep_window": best_keep["window"],
                "best_keep_corr_signal": best_keep["corr_signal"],
                "most_important_drop_window": most_drop["window"],
                "drop_delta_from_full": most_drop["drop_delta_from_full"],
            }
        )
    write_csv(query_best, out_dir / "best_windows_by_query.csv")

    group_best = []
    for group in sorted({str(r["roi_group"]) for r in aggregate_metrics}):
        rows = [r for r in aggregate_metrics if r["roi_group"] == group]
        keep = [r for r in rows if r["condition"] == "keep"]
        drop = [r for r in rows if r["condition"] == "drop"]
        full = next((r for r in rows if r["condition"] == "full"), None)
        if not keep or not drop or not full:
            continue
        best_keep = max(keep, key=lambda r: float(r["mean_corr_signal"]))
        most_drop = max(drop, key=lambda r: float(r["mean_drop_delta_from_full"]))
        group_best.append(
            {
                "roi_group": group,
                "n_query": full["n_query"],
                "full_mean_corr_signal": full["mean_corr_signal"],
                "best_keep_window": best_keep["window"],
                "best_keep_mean_corr_signal": best_keep["mean_corr_signal"],
                "most_important_drop_window": most_drop["window"],
                "mean_drop_delta_from_full": most_drop["mean_drop_delta_from_full"],
            }
        )
    write_csv(group_best, out_dir / "best_windows_by_group.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="query_time_dependency")
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_root = args.out_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    windows_ms = FINE_WINDOWS_MS
    window_names = [name for name, _, _ in windows_ms]

    all_query_rows: list[dict[str, float | int | str]] = []
    all_group_rows: list[dict[str, float | int | str]] = []
    run_summaries = []
    for run_dir in args.runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        roi_kind = str(summary["roi_kind"])
        run_name = run_dir.name
        train_roi = resolve_path(summary["train_roi"])
        test_roi = resolve_path(summary["test_roi"])
        train_payload = load_roi_payload(train_roi, roi_kind)
        test_payload = load_roi_payload(test_roi, roi_kind)
        names = test_payload["names"]  # type: ignore[assignment]
        visual_group_json = test_payload["visual_group_json"]  # type: ignore[assignment]
        target = test_payload["targets"]  # type: ignore[assignment]
        groups = roi_group_indices(names, visual_group_json)  # type: ignore[arg-type]

        model = build_model(summary, train_payload, device)
        ckpt = run_dir / args.checkpoint_name
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        subjects = list(summary["subjects"])
        eeg_stack = load_or_build_test_eeg_stack(
            args.data_root,
            subjects,
            test_payload["image_index"],  # type: ignore[arg-type]
            cache_dir=args.cache_dir,
            cache_tag=test_roi.stem,
        )
        assert eeg_stack is not None

        conditions: list[tuple[str, str, int | None, int | None]] = [("full", "full", None, None)]
        for name, start_ms, end_ms in windows_ms:
            conditions.append(("keep", name, start_ms, end_ms))
            conditions.append(("drop", name, start_ms, end_ms))

        full_pred = predict_roi(model, eeg_stack, subjects, device, args.batch_size, "full")
        shifted = np.roll(target, len(target) // 3, axis=0)  # type: ignore[arg-type]
        full_signal = {
            idx: pearson_1d(full_pred[:, idx], target[:, idx]) - pearson_1d(full_pred[:, idx], shifted[:, idx])  # type: ignore[index]
            for idx in range(full_pred.shape[1])
        }
        run_rows = query_rows(
            full_pred,
            target,  # type: ignore[arg-type]
            names,  # type: ignore[arg-type]
            visual_group_json,  # type: ignore[arg-type]
            run_name,
            roi_kind,
            "full",
            "full",
            None,
            None,
            full_signal,
        )
        for condition, window, start_ms, end_ms in conditions[1:]:
            pred = predict_roi(model, eeg_stack, subjects, device, args.batch_size, condition, start_ms, end_ms)
            run_rows.extend(
                query_rows(
                    pred,
                    target,  # type: ignore[arg-type]
                    names,  # type: ignore[arg-type]
                    visual_group_json,  # type: ignore[arg-type]
                    run_name,
                    roi_kind,
                    condition,
                    window,
                    start_ms,
                    end_ms,
                    full_signal,
                )
            )
        group_rows = aggregate_rows(run_rows, groups)
        all_query_rows.extend(run_rows)
        all_group_rows.extend(group_rows)

        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        write_csv(run_rows, run_out / "query_time_metrics.csv")
        write_csv(group_rows, run_out / "group_aggregate_time_metrics.csv")
        write_best_windows(run_rows, group_rows, run_out)

        y_labels = [str(name) for name in names]  # type: ignore[union-attr]
        for condition, value_key, suffix, title_value in [
            ("keep", "corr_signal", "keep_query_corr_signal.png", "keep corr-signal"),
            ("drop", "corr_signal", "drop_query_corr_signal.png", "drop corr-signal"),
            ("drop", "drop_delta_from_full", "drop_delta_from_full.png", "drop delta from full"),
        ]:
            mat = matrix_from_rows(run_rows, "roi_name", y_labels, window_names, value_key, condition)
            write_heatmap(mat, y_labels, window_names, f"{run_name}: {title_value}", run_out / suffix)

        group_labels = [g for g in groups if any(r["roi_group"] == g for r in group_rows)]
        for condition, value_key, suffix, title_value in [
            ("keep", "mean_corr_signal", "keep_group_mean_corr_signal.png", "group mean keep corr-signal"),
            ("drop", "mean_corr_signal", "drop_group_mean_corr_signal.png", "group mean drop corr-signal"),
            ("drop", "mean_drop_delta_from_full", "drop_group_delta_from_full.png", "group mean drop delta"),
        ]:
            mat = matrix_from_rows(group_rows, "roi_group", group_labels, window_names, value_key, condition)
            write_heatmap(mat, group_labels, window_names, f"{run_name}: {title_value}", run_out / suffix)

        run_summaries.append(
            {
                "run": run_name,
                "roi_kind": roi_kind,
                "checkpoint": str(ckpt),
                "train_roi": str(train_roi),
                "test_roi": str(test_roi),
                "n_query": int(len(names)),  # type: ignore[arg-type]
                "groups": {k: len(v) for k, v in groups.items()},
            }
        )

    write_csv(all_query_rows, out_root / "all_query_time_metrics.csv")
    write_csv(all_group_rows, out_root / "all_group_aggregate_time_metrics.csv")
    summary = {
        "tag": args.tag,
        "checkpoint_name": args.checkpoint_name,
        "runs": run_summaries,
        "device": str(device),
        "time_axis": "250 samples = 0-1000 ms post-stimulus after dropping 50 baseline samples",
        "windows_ms": windows_ms,
        "metrics_csv": str(out_root / "all_query_time_metrics.csv"),
        "group_metrics_csv": str(out_root / "all_group_aggregate_time_metrics.csv"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
