#!/usr/bin/env python3
"""Export predicted-query x TRIBE-target ROI correlation/confusion maps.

The ROI branch is trained with ordered supervision:

    predicted query i -> target ROI i

This script checks whether that identity is actually preserved on the test set.
For each model, it predicts ROI activations from averaged THINGS-EEG test trials
and computes a correlation matrix:

    rows: predicted ROI-query outputs
    cols: TRIBE pseudo-ROI targets

A strong diagonal means each query is closest to its intended target rather than
collapsing into a global visual signal or permuting ROI identities.
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
ROOT = WORKSPACE.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from evaluate_atm_roi_query_time_dependency import (  # noqa: E402
    build_model,
    predict_roi,
    primary_group_for_name,
    resolve_path,
)
from evaluate_atm_roi_temporal_hierarchy import load_roi_payload  # noqa: E402
from train_atm_roi_spatial_branch import load_or_build_test_eeg_stack, visual_group_features  # noqa: E402,F401


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_DATA_ROOT = ROOT / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_query_target_confusion"


def zscore_cols(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype("float64")
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def corr_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_z = zscore_cols(pred)
    target_z = zscore_cols(target)
    return (pred_z.T @ target_z) / max(pred.shape[0] - 1, 1)


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_heatmap(
    mat: np.ndarray,
    y_labels: list[str],
    x_labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "coolwarm",
    vcenter: float = 0.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = max(4.0, 0.24 * len(y_labels) + 1.3)
    width = max(7.0, 0.24 * len(x_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax if vcenter == 0.0 else None, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=75, ha="right", fontsize=6 if len(x_labels) > 20 else 8)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6 if len(y_labels) > 20 else 8)
    ax.set_xlabel("TRIBE target ROI")
    ax.set_ylabel("EEG-predicted ROI query")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def strip_hemi(name: str) -> str:
    return name[3:] if name.startswith(("lh_", "rh_")) else name


def group_confusion(
    mat: np.ndarray,
    names: np.ndarray,
    visual_group_json: dict[str, list[str]] | None,
) -> tuple[np.ndarray, list[str]]:
    query_groups = [primary_group_for_name(str(name), visual_group_json) for name in names]
    groups = sorted(dict.fromkeys(query_groups))
    out = np.full((len(groups), len(groups)), np.nan, dtype="float64")
    for i, gi in enumerate(groups):
        row_idxs = [idx for idx, group in enumerate(query_groups) if group == gi]
        for j, gj in enumerate(groups):
            col_idxs = [idx for idx, group in enumerate(query_groups) if group == gj]
            if row_idxs and col_idxs:
                out[i, j] = float(np.nanmean(mat[np.ix_(row_idxs, col_idxs)]))
    return out, groups


def summarize_identity(
    mat: np.ndarray,
    names: np.ndarray,
    visual_group_json: dict[str, list[str]] | None,
    run_name: str,
    roi_kind: str,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    groups = [primary_group_for_name(str(name), visual_group_json) for name in names]
    rows: list[dict[str, float | int | str]] = []
    diag = np.diag(mat)
    ranks = []
    same_group_offdiag = []
    other_group_offdiag = []
    top1 = []
    for i, name in enumerate(names):
        order = np.argsort(-mat[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        best = int(order[0])
        same = [j for j, group in enumerate(groups) if group == groups[i] and j != i]
        other = [j for j, group in enumerate(groups) if group != groups[i]]
        same_val = float(np.mean(mat[i, same])) if same else np.nan
        other_val = float(np.mean(mat[i, other])) if other else np.nan
        if not np.isnan(same_val):
            same_group_offdiag.append(same_val)
        if not np.isnan(other_val):
            other_group_offdiag.append(other_val)
        top1.append(best == i)
        rows.append(
            {
                "run": run_name,
                "roi_kind": roi_kind,
                "query_index": i,
                "query_roi": str(name),
                "query_group": groups[i],
                "diag_target_corr": float(mat[i, i]),
                "diag_rank": rank,
                "diag_rank_percentile": float(1 - (rank - 1) / max(len(names) - 1, 1)),
                "best_target_index": best,
                "best_target_roi": str(names[best]),
                "best_target_group": groups[best],
                "best_target_corr": float(mat[i, best]),
                "diag_minus_best_other": float(mat[i, i] - mat[i, order[1]]) if len(order) > 1 and best == i else float(mat[i, i] - mat[i, best]),
                "same_group_offdiag_corr": same_val,
                "other_group_offdiag_corr": other_val,
            }
        )
    summary = {
        "run": run_name,
        "roi_kind": roi_kind,
        "n_roi": int(len(names)),
        "diag_mean_corr": float(np.mean(diag)),
        "diag_median_corr": float(np.median(diag)),
        "offdiag_mean_corr": float(np.mean(mat[~np.eye(len(names), dtype=bool)])),
        "diag_minus_offdiag_mean": float(np.mean(diag) - np.mean(mat[~np.eye(len(names), dtype=bool)])),
        "diag_top1_fraction": float(np.mean(top1)),
        "diag_top5_fraction": float(np.mean(np.asarray(ranks) <= 5)),
        "diag_rank_percentile_mean": float(np.mean(1 - (np.asarray(ranks) - 1) / max(len(names) - 1, 1))),
        "same_group_offdiag_mean": float(np.mean(same_group_offdiag)) if same_group_offdiag else float("nan"),
        "other_group_offdiag_mean": float(np.mean(other_group_offdiag)) if other_group_offdiag else float("nan"),
        "same_minus_other_group_offdiag": float(np.mean(same_group_offdiag) - np.mean(other_group_offdiag)) if same_group_offdiag and other_group_offdiag else float("nan"),
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="query_target_confusion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_root = args.out_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    all_matrix_rows: list[dict[str, float | int | str]] = []
    all_query_rows: list[dict[str, float | int | str]] = []
    all_group_rows: list[dict[str, float | int | str]] = []
    summaries = []

    for run_dir in args.runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        roi_kind = str(summary["roi_kind"])
        subjects = list(summary["subjects"])
        train_roi = resolve_path(summary["train_roi"])
        test_roi = resolve_path(summary["test_roi"])
        train_payload = load_roi_payload(train_roi, roi_kind)
        test_payload = load_roi_payload(test_roi, roi_kind)
        names = test_payload["names"]  # type: ignore[assignment]
        target = np.asarray(test_payload["targets"], dtype="float32")  # type: ignore[index]
        visual_group_json = test_payload["visual_group_json"]  # type: ignore[assignment]
        model = build_model(summary, train_payload, device)
        ckpt = run_dir / args.checkpoint_name
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        eeg_stack = load_or_build_test_eeg_stack(
            args.data_root,
            subjects,
            test_payload["image_index"],  # type: ignore[arg-type]
            cache_dir=args.cache_dir,
            cache_tag=test_roi.stem,
        )
        assert eeg_stack is not None
        pred = predict_roi(model, eeg_stack, subjects, device, args.batch_size, "full")
        mat = corr_matrix(pred, target)

        run_out = out_root / f"{run_dir.name}_{args.checkpoint_name.replace('.pt', '')}"
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "query_target_correlation.npz",
            corr=mat,
            pred=pred,
            target=target,
            roi_names=names,
        )
        labels = [strip_hemi(str(name)) for name in names]  # type: ignore[union-attr]
        write_heatmap(
            mat,
            labels,
            labels,
            f"{run_dir.name}: predicted query x target ROI correlation",
            run_out / "query_target_corr_heatmap.png",
        )

        group_mat, group_labels = group_confusion(mat, names, visual_group_json)  # type: ignore[arg-type]
        write_heatmap(
            group_mat,
            group_labels,
            group_labels,
            f"{run_dir.name}: ROI-group confusion",
            run_out / "group_confusion_heatmap.png",
        )

        run_summary, query_rows = summarize_identity(mat, names, visual_group_json, run_dir.name, roi_kind)  # type: ignore[arg-type]
        run_summary["checkpoint_name"] = args.checkpoint_name
        summaries.append(run_summary)
        write_csv(query_rows, run_out / "per_query_identity.csv")
        all_query_rows.extend(query_rows)

        matrix_rows = []
        groups = [primary_group_for_name(str(name), visual_group_json) for name in names]  # type: ignore[arg-type]
        for i, query_name in enumerate(names):
            for j, target_name in enumerate(names):
                row = {
                    "run": run_dir.name,
                    "roi_kind": roi_kind,
                    "checkpoint_name": args.checkpoint_name,
                    "query_index": i,
                    "query_roi": str(query_name),
                    "query_group": groups[i],
                    "target_index": j,
                    "target_roi": str(target_name),
                    "target_group": groups[j],
                    "corr": float(mat[i, j]),
                    "is_diagonal": int(i == j),
                    "is_same_group": int(groups[i] == groups[j]),
                }
                matrix_rows.append(row)
        write_csv(matrix_rows, run_out / "query_target_corr_matrix_long.csv")
        all_matrix_rows.extend(matrix_rows)

        group_rows = []
        for i, query_group in enumerate(group_labels):
            for j, target_group in enumerate(group_labels):
                group_rows.append(
                    {
                        "run": run_dir.name,
                        "roi_kind": roi_kind,
                        "checkpoint_name": args.checkpoint_name,
                        "query_group": query_group,
                        "target_group": target_group,
                        "mean_corr": float(group_mat[i, j]),
                        "is_diagonal": int(i == j),
                    }
                )
        write_csv(group_rows, run_out / "group_confusion_matrix_long.csv")
        all_group_rows.extend(group_rows)

        print(json.dumps(run_summary, indent=2), flush=True)

    write_csv(all_matrix_rows, out_root / "all_query_target_corr_matrix_long.csv")
    write_csv(all_query_rows, out_root / "all_per_query_identity.csv")
    write_csv(all_group_rows, out_root / "all_group_confusion_matrix_long.csv")
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint_name": args.checkpoint_name, "runs": summaries}, f, indent=2)
    print(f"Wrote {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
