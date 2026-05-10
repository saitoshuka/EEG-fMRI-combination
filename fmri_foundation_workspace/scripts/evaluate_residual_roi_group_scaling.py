#!/usr/bin/env python3
"""Per-ROI-group scaling analysis for residual ROI-query checkpoints."""

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

from evaluate_atm_roi_temporal_hierarchy import load_roi_payload, roi_group_indices  # noqa: E402
from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    subject_to_id,
    visual_group_features,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "roi_semantic_residual"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [ROOT / p, WORKSPACE / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / p


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray, idx: list[int]) -> dict[str, float]:
    x = l2_normalize(pred[:, idx].astype("float64"))
    y = l2_normalize(target[:, idx].astype("float64"))
    sims = x @ y.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    shifted = (np.arange(len(pred)) + max(1, len(pred) // 3)) % len(pred)
    shifted_diag = sims[np.arange(len(pred)), shifted]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(axis=1) + 1
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "rank": float((1.0 - (ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "shifted_rank": float((1.0 - (shifted_ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "diag_minus_offdiag": float(diag.mean() - sims[~np.eye(len(pred), dtype=bool)].mean()),
    }


def col_corr(pred: np.ndarray, target: np.ndarray, idx: list[int], eps: float = 1e-12) -> float:
    x = pred[:, idx].astype("float64")
    y = target[:, idx].astype("float64")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    corr = (x * y).sum(axis=0) / np.maximum(denom, eps)
    return float(np.nanmean(corr))


def predict_roi(run_dir: Path, data_root: Path, cache_dir: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, dict, dict]:
    summary = json.loads((run_dir / "summary.json").read_text())
    roi_kind = str(summary["roi_kind"])
    train_payload = load_roi_payload(resolve_path(summary["train_roi"]), roi_kind)
    test_payload = load_roi_payload(resolve_path(summary["test_roi"]), roi_kind)
    visual_json = json.dumps(train_payload["visual_group_json"]) if train_payload["visual_group_json"] else None
    group_features = visual_group_features(train_payload["names"], visual_json)  # type: ignore[arg-type]
    model = AtmSemanticSpatial(
        roi_names=train_payload["names"],  # type: ignore[arg-type]
        vertex_counts=train_payload["vertex_counts"],  # type: ignore[arg-type]
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
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    subjects = list(summary["subjects"])
    eeg_stack = load_or_build_test_eeg_stack(
        data_root,
        subjects,
        test_payload["image_index"],  # type: ignore[arg-type]
        cache_dir=cache_dir,
        cache_tag=resolve_path(summary["test_roi"]).stem,
    )
    assert eeg_stack is not None
    subject_preds = []
    with torch.no_grad():
        for subject_idx, subject in enumerate(subjects):
            eeg = eeg_stack[subject_idx]
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            preds = []
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                preds.append(model(x, sids)["roi_pred"].cpu())
            subject_preds.append(torch.cat(preds, dim=0))
    pred = torch.stack(subject_preds, dim=0).mean(dim=0).numpy()
    return pred, summary, test_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="per_group_scaling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in args.runs:
        pred, summary, test_payload = predict_roi(run_dir, args.data_root, args.cache_dir, device, args.batch_size)
        roi_kind = str(summary["roi_kind"])
        groups = roi_group_indices(
            test_payload["names"],  # type: ignore[arg-type]
            test_payload["visual_group_json"],  # type: ignore[arg-type]
        )
        target = test_payload["targets"]  # type: ignore[assignment]
        for group, idx in groups.items():
            metrics = retrieval_metrics(pred, target, idx)  # type: ignore[arg-type]
            rows.append(
                {
                    "run": run_dir.name,
                    "train_images": int(summary["train_images"]),
                    "roi_kind": roi_kind,
                    "roi_group": group,
                    "n_roi": len(idx),
                    **metrics,
                    "rank_signal": metrics["rank"] - metrics["shifted_rank"],
                    "col_corr": col_corr(pred, target, idx),  # type: ignore[arg-type]
                }
            )

    csv_path = out_dir / "per_group_residual_roi_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []
    keys = sorted({(r["roi_kind"], r["roi_group"]) for r in rows})
    for roi_kind, group in keys:
        budget_rows = [r for r in rows if r["roi_kind"] == roi_kind and r["roi_group"] == group]
        by_budget: dict[int, dict[str, float | int | str]] = {}
        for row in budget_rows:
            budget = int(row["train_images"])
            current = by_budget.get(budget)
            if current is None or float(row["rank"]) > float(current["rank"]):
                by_budget[budget] = row
        if len(by_budget) >= 2:
            budgets = sorted(by_budget)
            first = by_budget[budgets[0]]
            out_row: dict[str, float | int | str] = {
                "roi_kind": roi_kind,
                "roi_group": group,
                "n_roi": first["n_roi"],
            }
            for budget in budgets:
                row = by_budget[budget]
                out_row[f"best_run_{budget}"] = row["run"]
                out_row[f"rank_{budget}"] = row["rank"]
                out_row[f"shifted_rank_{budget}"] = row["shifted_rank"]
                out_row[f"rank_signal_{budget}"] = row["rank_signal"]
                out_row[f"col_corr_{budget}"] = row["col_corr"]
            for prev, cur in zip(budgets[:-1], budgets[1:]):
                out_row[f"rank_delta_{prev}_to_{cur}"] = float(by_budget[cur]["rank"]) - float(by_budget[prev]["rank"])
                out_row[f"rank_signal_delta_{prev}_to_{cur}"] = (
                    float(by_budget[cur]["rank_signal"]) - float(by_budget[prev]["rank_signal"])
                )
                out_row[f"col_corr_delta_{prev}_to_{cur}"] = float(by_budget[cur]["col_corr"]) - float(by_budget[prev]["col_corr"])
            if 8192 in by_budget and 16540 in by_budget:
                out_row["rank_delta_8192_to_16k"] = float(by_budget[16540]["rank"]) - float(by_budget[8192]["rank"])
                out_row["rank_signal_delta_8192_to_16k"] = (
                    float(by_budget[16540]["rank_signal"]) - float(by_budget[8192]["rank_signal"])
                )
            comparison_rows.append(out_row)
    comparison_path = out_dir / "per_group_residual_roi_scaling.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        keys_out = sorted({key for row in comparison_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys_out)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "tag": args.tag,
        "runs": [str(r) for r in args.runs],
        "metrics_csv": str(csv_path),
        "comparison_csv": str(comparison_path),
        "device": str(device),
        "rows": comparison_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
