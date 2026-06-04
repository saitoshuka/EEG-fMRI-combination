#!/usr/bin/env python3
"""Evaluate completed ATM real-fMRI visual64 runs with correlation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_things_fmri_external_roi_breakdown import fisher_mean
from evaluate_raw_eeg_to_realfmri_overlap_holdout import retrieval_metrics, row_corr, vector_corr
from train_atm_roi_spatial_branch import (
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    subject_to_id,
    visual_group_features,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_shared_roi_targets"
)
DEFAULT_RUN_ROOT = WORKSPACE / "results" / "eeg_image_bridge" / "atm_real_fmri_shared_roi207"
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_visual64_eval"
)


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def corr_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    roi_corr = np.asarray([vector_corr(pred[:, i], target[:, i]) for i in range(pred.shape[1])])
    rank = retrieval_metrics(pred, target)
    shifted = retrieval_metrics(np.roll(pred, 1, axis=0), target)
    return {
        "rank_percentile": rank["rank_percentile"],
        "shifted_rank_percentile": shifted["rank_percentile"],
        "rank_delta": rank["rank_percentile"] - shifted["rank_percentile"],
        "top1": rank["top1"],
        "top5": rank["top5"],
        "diag_minus_offdiag": rank["diag_minus_offdiag"],
        "image_pattern_corr_mean": float(np.nanmean(row_corr(pred, target))),
        "image_pattern_corr_median": float(np.nanmedian(row_corr(pred, target))),
        "roi_corr_fisher_mean": fisher_mean(roi_corr),
        "roi_corr_median": float(np.nanmedian(roi_corr)),
    }


@torch.no_grad()
def evaluate_run(
    run_dir: Path,
    test_roi_npz: Path,
    data_root: Path,
    cache_dir: Path,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, float | str | int], np.ndarray, np.ndarray]:
    summary = json.loads((run_dir / "summary.json").read_text())
    target_payload = np.load(test_roi_npz, allow_pickle=True)
    roi_target = torch.from_numpy(target_payload["parcel_targets"].astype("float32"))
    roi_names = target_payload["parcel_names"]
    vertex_counts = target_payload["parcel_vertex_counts"]
    image_index = target_payload["image_index"].astype(int)
    group_features = visual_group_features(roi_names, None)

    subjects = summary["subjects"]
    test_eeg_stack = load_or_build_test_eeg_stack(
        data_root,
        subjects,
        image_index,
        cache_dir=cache_dir,
        cache_tag=test_roi_npz.stem,
    )
    model = AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        subject_mode=summary["subject_mode"],
        atm_d_model=int(summary["atm_d_model"]),
        atm_heads=int(summary["atm_heads"]),
        atm_layers=int(summary["atm_layers"]),
        atm_dropout=float(summary["atm_dropout"]),
        atm_d_ff=int(summary["atm_d_ff"]),
        semantic_head=summary["semantic_head"],
        use_spatial=True,
        spatial_head=summary.get("spatial_head", "query"),
    ).to(device)
    ckpt_path = run_dir / checkpoint_name
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    preds = []
    for subject_idx, subject in enumerate(subjects):
        eeg = test_eeg_stack[subject_idx]
        sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
        out = model(eeg.to(device), sid.to(device))
        preds.append(out["roi_pred"].detach().cpu())
    pred = torch.stack(preds, dim=0).mean(dim=0).numpy()
    target = roi_target.numpy()
    metrics = corr_metrics(pred, target)
    metrics.update(
        {
            "run": run_dir.name,
            "checkpoint": checkpoint_name,
            "n_test": int(len(image_index)),
            "n_roi": int(target.shape[1]),
            "spatial_head": summary.get("spatial_head", ""),
        }
    )
    return metrics, pred.astype(np.float32), target.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    test_roi_npz = args.target_dir / "real_fmri_visual_roi64_ztrain_test_n77.npz"
    run_names = [
        "atm_query_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005",
        "atm_pooled_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_name in run_names:
        run_dir = args.run_root / run_name
        for checkpoint in ["model_best_roi_rank.pt", "model_final.pt"]:
            metrics, pred, target = evaluate_run(
                run_dir,
                test_roi_npz,
                args.data_root,
                args.cache_dir,
                device,
                checkpoint,
            )
            rows.append(metrics)
            np.savez_compressed(
                args.out_dir / f"{run_name}_{checkpoint.replace('.pt', '')}_predictions.npz",
                pred=pred,
                target=target,
            )
    summary = {"rows": rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
