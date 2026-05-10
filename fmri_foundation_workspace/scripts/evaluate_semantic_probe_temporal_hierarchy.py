#!/usr/bin/env python3
"""Temporal hierarchy baseline for semantic-only ATM embeddings.

This script asks whether the ROI timing pattern already exists in the
semantic-only EEG-CLIP representation. It loads a semantic-only ATM checkpoint,
averages its train EEG embeddings per image, fits a ridge probe from semantic
embedding to TRIBE visual ROI targets, and then runs the same test-time EEG
time-window ablation used for the ROI-query branch.

This is intentionally a probe baseline, not a new end-to-end model:

    EEG -> semantic-only ATM embedding -> linear ROI probe -> ROI target
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

from evaluate_atm_roi_temporal_hierarchy import (  # noqa: E402
    FINE_WINDOWS_MS,
    GROUP_ORDER,
    WINDOWS_MS,
    apply_condition,
    load_roi_payload,
    metric_rows,
    roi_group_indices,
    write_best_windows,
    write_heatmap,
)
from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    load_or_build_train_eeg_subset,
    subject_to_id,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_semantic_probe_temporal_hierarchy"
DEFAULT_TEST_ROI = DEFAULT_RESULTS / "visual_roi_targets" / "visual_roi_targets_n200.npz"


class RidgeProbe:
    def __init__(self, weight: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
        self.weight = weight
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        xz = (x - self.x_mean) / self.x_std
        x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1)
        yz = x_aug @ self.weight
        return yz * self.y_std + self.y_mean


def build_semantic_model(summary: dict, device: torch.device) -> AtmSemanticSpatial:
    model = AtmSemanticSpatial(
        roi_names=None,
        vertex_counts=None,
        group_features=None,
        num_subjects=10,
        subject_mode=summary.get("subject_mode", "none"),
        atm_d_model=int(summary.get("atm_d_model", 256)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=summary.get("semantic_head", "shallow"),
        use_spatial=False,
    ).to(device)
    return model


def predict_train_semantic_by_image(
    model: AtmSemanticSpatial,
    eeg_subset,
    n_images: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    sums = torch.zeros(n_images, 1024, dtype=torch.float32)
    counts = torch.zeros(n_images, 1, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(eeg_subset.eeg), batch_size):
            end = min(start + batch_size, len(eeg_subset.eeg))
            x = eeg_subset.eeg[start:end].to(device)
            sids = eeg_subset.subject_ids[start:end].to(device)
            image_ids = eeg_subset.image_local[start:end].long()
            sem = model(x, sids)["semantic"].cpu()
            sums.index_add_(0, image_ids, sem)
            counts.index_add_(0, image_ids, torch.ones((len(image_ids), 1), dtype=torch.float32))
    return (sums / counts.clamp_min(1.0)).numpy()


def predict_test_semantic(
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
                preds.append(model(x, sids)["semantic"].cpu())
            subject_preds.append(torch.cat(preds, dim=0))
    return torch.stack(subject_preds, dim=0).mean(dim=0).numpy()


def fit_ridge_probe(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeProbe:
    x = x.astype("float64")
    y = y.astype("float64")
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + 1e-6
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=xtx.dtype) * alpha
    reg[-1, -1] = 0.0
    weight = np.linalg.solve(xtx + reg, x_aug.T @ yz)
    return RidgeProbe(
        weight=weight.astype("float32"),
        x_mean=x_mean.astype("float32"),
        x_std=x_std.astype("float32"),
        y_mean=y_mean.astype("float32"),
        y_std=y_std.astype("float32"),
    )


def format_window(window: str) -> str:
    return window.replace("w", "").replace("_", "-") if window != "full" else window


def write_probe_summary(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
) -> None:
    full = {
        (str(r["roi_kind"]), str(r["roi_group"])): float(r["stim_corr_signal"])
        for r in rows
        if r["condition"] == "full" and r["window"] == "full"
    }
    out_rows = []
    for roi_kind in sorted({str(r["roi_kind"]) for r in rows}):
        for group in GROUP_ORDER:
            key = (roi_kind, group)
            if key not in full:
                continue
            keep = [r for r in rows if str(r["roi_kind"]) == roi_kind and r["roi_group"] == group and r["condition"] == "keep"]
            drop = [r for r in rows if str(r["roi_kind"]) == roi_kind and r["roi_group"] == group and r["condition"] == "drop"]
            if not keep or not drop:
                continue
            best_keep = max(keep, key=lambda r: float(r["stim_corr_signal"]))
            worst_drop = min(drop, key=lambda r: float(r["stim_corr_signal"]))
            out_rows.append(
                {
                    "roi_kind": roi_kind,
                    "roi_group": group,
                    "full_signal_corr": full[key],
                    "best_keep_window": best_keep["window"],
                    "best_keep_window_ms": format_window(str(best_keep["window"])),
                    "best_keep_signal_corr": best_keep["stim_corr_signal"],
                    "most_important_drop_window": worst_drop["window"],
                    "most_important_drop_window_ms": format_window(str(worst_drop["window"])),
                    "drop_delta_signal_corr": full[key] - float(worst_drop["stim_corr_signal"]),
                }
            )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--train-roi", type=Path, required=True)
    parser.add_argument("--test-roi", type=Path, default=DEFAULT_TEST_ROI)
    parser.add_argument("--roi-kinds", nargs="+", choices=["group", "parcel"], default=["group", "parcel"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="semantic_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--window-mode", choices=["coarse", "fine100"], default="fine100")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_ms = FINE_WINDOWS_MS if args.window_mode == "fine100" else WINDOWS_MS

    summary = json.loads((args.semantic_run / "summary.json").read_text())
    subjects = list(summary["subjects"])
    train_roi_ref = np.load(args.train_roi, allow_pickle=True)
    train_image_index = train_roi_ref["image_index"].astype(int)

    eeg_subset = load_or_build_train_eeg_subset(
        args.data_root,
        subjects,
        train_image_index,
        cache_dir=args.cache_dir,
        cache_tag=args.train_roi.stem,
    )
    model = build_semantic_model(summary, device)
    state = torch.load(args.semantic_run / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)

    emb_cache = out_dir / f"train_semantic_embeddings_{args.semantic_run.name}_{args.train_roi.stem}.npz"
    if emb_cache.exists():
        train_sem = np.load(emb_cache)["semantic"].astype("float32")
    else:
        train_sem = predict_train_semantic_by_image(
            model,
            eeg_subset,
            n_images=len(train_image_index),
            device=device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(emb_cache, semantic=train_sem, image_index=train_image_index)

    test_ref = np.load(args.test_roi, allow_pickle=True)
    test_image_index = test_ref["image_index"].astype(int)
    eeg_stack = load_or_build_test_eeg_stack(
        args.data_root,
        subjects,
        test_image_index,
        cache_dir=args.cache_dir,
        cache_tag=args.test_roi.stem,
    )
    assert eeg_stack is not None

    conditions: list[tuple[str, str, int | None, int | None]] = [("full", "full", None, None)]
    for name, start_ms, end_ms in windows_ms:
        conditions.append(("keep", name, start_ms, end_ms))
        conditions.append(("drop", name, start_ms, end_ms))

    semantic_by_condition: dict[tuple[str, str], np.ndarray] = {}
    for condition, window, start_ms, end_ms in conditions:
        semantic_by_condition[(condition, window)] = predict_test_semantic(
            model,
            eeg_stack,
            subjects,
            device,
            args.batch_size,
            condition,
            start_ms,
            end_ms,
        )

    all_rows: list[dict[str, float | int | str]] = []
    probe_info = []
    for roi_kind in args.roi_kinds:
        train_payload = load_roi_payload(args.train_roi, roi_kind)
        test_payload = load_roi_payload(args.test_roi, roi_kind)
        target_train = train_payload["targets"]  # type: ignore[assignment]
        target_test = test_payload["targets"]  # type: ignore[assignment]
        groups = roi_group_indices(
            test_payload["names"],  # type: ignore[arg-type]
            test_payload["visual_group_json"],  # type: ignore[arg-type]
        )
        probe = fit_ridge_probe(train_sem, target_train, alpha=args.ridge_alpha)  # type: ignore[arg-type]
        run_rows = []
        for condition, window, start_ms, end_ms in conditions:
            pred = probe.predict(semantic_by_condition[(condition, window)])
            run_rows.extend(
                metric_rows(
                    pred,
                    target_test,  # type: ignore[arg-type]
                    groups,
                    args.semantic_run.name,
                    roi_kind,
                    condition,
                    window,
                    start_ms,
                    end_ms,
                )
            )
        all_rows.extend(run_rows)
        roi_out = out_dir / f"{roi_kind}_semantic_probe"
        roi_out.mkdir(parents=True, exist_ok=True)
        write_heatmap(
            run_rows,
            roi_out / "keep_window_stim_corr_signal.png",
            "keep",
            "stim_corr_signal",
            f"semantic-only probe {roi_kind}: keep-window ROI signal corr",
            windows_ms,
        )
        write_heatmap(
            run_rows,
            roi_out / "drop_window_stim_corr_signal.png",
            "drop",
            "stim_corr_signal",
            f"semantic-only probe {roi_kind}: drop-window ROI signal corr",
            windows_ms,
        )
        write_best_windows(run_rows, roi_out / "best_windows.csv", f"semantic_probe_{roi_kind}")
        probe_info.append({"roi_kind": roi_kind, "n_roi": int(target_train.shape[1]), "groups": {k: len(v) for k, v in groups.items()}})

    csv_path = out_dir / "semantic_probe_temporal_hierarchy_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    write_probe_summary(all_rows, out_dir / "semantic_probe_best_windows.csv")

    out_summary = {
        "semantic_run": str(args.semantic_run),
        "train_roi": str(args.train_roi),
        "test_roi": str(args.test_roi),
        "ridge_alpha": args.ridge_alpha,
        "train_images": int(len(train_image_index)),
        "subjects": subjects,
        "device": str(device),
        "window_mode": args.window_mode,
        "time_axis": "250 samples = 0-1000 ms post-stimulus after dropping 50 baseline samples",
        "emb_cache": str(emb_cache),
        "metrics_csv": str(csv_path),
        "probes": probe_info,
    }
    (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(json.dumps(out_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
