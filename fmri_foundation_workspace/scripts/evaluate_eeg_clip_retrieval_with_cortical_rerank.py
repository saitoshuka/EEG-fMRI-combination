#!/usr/bin/env python3
"""Evaluate cortical reranking for EEG image retrieval on THINGS-fMRI overlap.

The experiment asks whether the real-fMRI cortical branch helps image retrieval,
not just fMRI-pattern retrieval.  For each image-heldout split:

1. Train a semantic ridge readout from image-averaged EEG to CLIP image features.
2. Load the trainable factorized-query EEG->real-fMRI visual prediction.
3. Tune a scalar fusion weight on validation retrieval only.
4. Evaluate semantic-only, cortical-only, and fused retrieval on heldout images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_raw_eeg_realfmri_temporal_channel_ablation import (
    DEFAULT_DATA_ROOT,
    channel_group_masks,
    load_channel_names,
)
from evaluate_image_features_to_realfmri_overlap_holdout import load_feature_rows
from evaluate_raw_eeg_to_realfmri_overlap_holdout import (
    DEFAULT_EEG_MEMMAP_DIR,
    build_image_level_eeg,
    norm_rows,
    retrieval_metrics,
    ridge_predict,
    zscore_train,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EXT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
DEFAULT_PREDICTOR_DIR = DEFAULT_EXT_DIR / "image_feature_predictors"
DEFAULT_OUT_DIR = DEFAULT_EXT_DIR / "eeg_clip_retrieval_cortical_rerank"
DEFAULT_REPORT = WORKSPACE / "notes" / "eeg_image_bridge" / "eeg_clip_retrieval_cortical_rerank_20260604.md"


def split_npz_path(ext_dir: Path, seed: int) -> Path:
    if seed == 33:
        return ext_dir / "raw_eeg_to_realfmri_overlap_holdout" / "raw_eeg_to_realfmri_overlap_holdout_predictions.npz"
    return (
        ext_dir
        / f"raw_eeg_to_realfmri_overlap_holdout_seed{seed}"
        / "raw_eeg_to_realfmri_overlap_holdout_predictions.npz"
    )


def factorized_npz_path(ext_dir: Path, seed: int) -> Path:
    return (
        ext_dir
        / f"raw_eeg_factorized_query_model_seed{seed}"
        / "factorized_query_posterior_P_PO_O_predictions.npz"
    )


def select_channel_mask(ch_names: list[str], channel_set: str) -> np.ndarray:
    if channel_set == "full":
        return np.ones(len(ch_names), dtype=bool)
    groups = channel_group_masks(ch_names)
    if channel_set not in groups:
        raise ValueError(f"Unknown channel_set={channel_set}")
    return groups[channel_set]


def build_eeg_features(
    image_index: np.ndarray,
    subjects: list[str],
    memmap_dir: Path,
    data_root: Path,
    channel_set: str,
    time_pool: int,
) -> tuple[np.ndarray, list[str]]:
    tensor = build_image_level_eeg(image_index, subjects, memmap_dir, time_pool).reshape(len(image_index), 63, -1)
    ch_names = load_channel_names(data_root, subjects[0])
    mask = select_channel_mask(ch_names, channel_set)
    selected = tensor[:, mask, :]
    selected_names = [ch for ch, keep in zip(ch_names, mask) if keep]
    return selected.reshape(len(image_index), -1).astype(np.float32), selected_names


def sim_metrics(sim: np.ndarray) -> dict[str, float]:
    n = sim.shape[0]
    true = np.diag(sim)
    ranks = (sim > true[:, None]).sum(axis=1) + 1
    offdiag = ~np.eye(n, dtype=bool)
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "top10": float((ranks <= min(10, n)).mean()),
        "rank": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
        "diag_off": float(true.mean() - sim[offdiag].mean()),
        "mean_rank": float(ranks.mean()),
    }


def retrieval_from_features(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return sim_metrics(norm_rows(query.astype(np.float32)) @ norm_rows(target.astype(np.float32)).T)


def tune_semantic_ridge(
    x_fit_raw: np.ndarray,
    x_val_raw: np.ndarray,
    x_hold_raw: np.ndarray,
    clip_fit_raw: np.ndarray,
    clip_val_raw: np.ndarray,
    clip_hold_raw: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    x_fit, x_val = zscore_train(x_fit_raw, x_val_raw)
    _, x_hold = zscore_train(x_fit_raw, x_hold_raw)
    y_fit, y_val = zscore_train(clip_fit_raw, clip_val_raw)
    _, y_hold = zscore_train(clip_fit_raw, clip_hold_raw)
    best_alpha = alphas[0]
    best_metrics: dict[str, float] | None = None
    best_val_pred = None
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha)
        metrics = retrieval_from_features(pred_val, y_val)
        if best_metrics is None or (metrics["rank"], metrics["diag_off"]) > (
            best_metrics["rank"],
            best_metrics["diag_off"],
        ):
            best_alpha = alpha
            best_metrics = metrics
            best_val_pred = pred_val
    pred_hold = ridge_predict(x_fit, y_fit, x_hold, best_alpha)
    return best_val_pred.astype(np.float32), pred_hold, y_val, y_hold, best_alpha, best_metrics or {}


def zscore_sim(sim: np.ndarray) -> np.ndarray:
    return (sim - sim.mean(axis=1, keepdims=True)) / (sim.std(axis=1, keepdims=True) + 1e-6)


def tune_fusion(
    semantic_val_sim: np.ndarray,
    cortical_val_sim: np.ndarray,
    weights: list[float],
) -> tuple[float, dict[str, float]]:
    best_weight = weights[0]
    best_metrics: dict[str, float] | None = None
    sem = zscore_sim(semantic_val_sim)
    cort = zscore_sim(cortical_val_sim)
    for weight in weights:
        sim = (1.0 - weight) * sem + weight * cort
        metrics = sim_metrics(sim)
        if best_metrics is None or (metrics["rank"], metrics["top5"], metrics["diag_off"]) > (
            best_metrics["rank"],
            best_metrics["top5"],
            best_metrics["diag_off"],
        ):
            best_weight = weight
            best_metrics = metrics
    return best_weight, best_metrics or {}


def evaluate_seed(
    *,
    seed: int,
    ext_dir: Path,
    predictor_path: Path,
    subjects: list[str],
    memmap_dir: Path,
    data_root: Path,
    channel_set: str,
    time_pool: int,
    alphas: list[float],
    fusion_weights: list[float],
) -> dict[str, object]:
    split_data = np.load(split_npz_path(ext_dir, seed), allow_pickle=True)
    fit_idx = split_data["fit_image_index"].astype(int)
    val_idx = split_data["val_image_index"].astype(int)
    hold_idx = split_data["holdout_image_index"].astype(int)
    x_fit, ch_names = build_eeg_features(fit_idx, subjects, memmap_dir, data_root, channel_set, time_pool)
    x_val, _ = build_eeg_features(val_idx, subjects, memmap_dir, data_root, channel_set, time_pool)
    x_hold, _ = build_eeg_features(hold_idx, subjects, memmap_dir, data_root, channel_set, time_pool)
    clip_fit = load_feature_rows(predictor_path, fit_idx)
    clip_val = load_feature_rows(predictor_path, val_idx)
    clip_hold = load_feature_rows(predictor_path, hold_idx)
    sem_val_pred, sem_hold_pred, clip_val_target, clip_hold_target, best_alpha, sem_val_metrics = tune_semantic_ridge(
        x_fit,
        x_val,
        x_hold,
        clip_fit,
        clip_val,
        clip_hold,
        alphas,
    )
    sem_val_sim = norm_rows(sem_val_pred) @ norm_rows(clip_val_target).T
    sem_hold_sim = norm_rows(sem_hold_pred) @ norm_rows(clip_hold_target).T

    factorized = np.load(factorized_npz_path(ext_dir, seed), allow_pickle=True)
    cortical_val_pred = np.asarray(factorized["val_pred"], dtype=np.float32)
    cortical_val_target = np.asarray(factorized["val_target"], dtype=np.float32)
    cortical_hold_pred = np.asarray(factorized["holdout_pred"], dtype=np.float32)
    cortical_hold_target = np.asarray(factorized["holdout_target"], dtype=np.float32)
    if not np.array_equal(factorized["val_image_index"].astype(int), val_idx):
        raise ValueError(f"Seed {seed}: factorized val indices do not match split")
    if not np.array_equal(factorized["holdout_image_index"].astype(int), hold_idx):
        raise ValueError(f"Seed {seed}: factorized holdout indices do not match split")
    cortical_val_sim = norm_rows(cortical_val_pred) @ norm_rows(cortical_val_target).T
    cortical_hold_sim = norm_rows(cortical_hold_pred) @ norm_rows(cortical_hold_target).T

    best_weight, fusion_val_metrics = tune_fusion(sem_val_sim, cortical_val_sim, fusion_weights)
    fusion_hold_sim = (1.0 - best_weight) * zscore_sim(sem_hold_sim) + best_weight * zscore_sim(cortical_hold_sim)
    semantic_hold = sim_metrics(sem_hold_sim)
    cortical_hold = sim_metrics(cortical_hold_sim)
    fusion_hold = sim_metrics(fusion_hold_sim)
    oracle_rows = []
    for weight in fusion_weights:
        sim = (1.0 - weight) * zscore_sim(sem_hold_sim) + weight * zscore_sim(cortical_hold_sim)
        metrics = sim_metrics(sim)
        oracle_rows.append({"weight": weight, **metrics})
    oracle = max(oracle_rows, key=lambda row: (row["rank"], row["top5"], row["diag_off"]))
    return {
        "seed": seed,
        "n_fit": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_holdout": int(len(hold_idx)),
        "channel_set": channel_set,
        "n_channels": int(len(ch_names)),
        "semantic_alpha": float(best_alpha),
        "fusion_weight": float(best_weight),
        "semantic_val": sem_val_metrics,
        "fusion_val": fusion_val_metrics,
        "semantic_hold": semantic_hold,
        "cortical_hold": cortical_hold,
        "fusion_hold": fusion_hold,
        "oracle_fusion_hold": oracle,
        "fusion_minus_semantic_rank": fusion_hold["rank"] - semantic_hold["rank"],
        "fusion_minus_semantic_top5": fusion_hold["top5"] - semantic_hold["top5"],
        "cortical_minus_semantic_rank": cortical_hold["rank"] - semantic_hold["rank"],
    }


def flat_row(result: dict[str, object]) -> dict[str, object]:
    row = {
        "seed": result["seed"],
        "semantic_alpha": result["semantic_alpha"],
        "fusion_weight": result["fusion_weight"],
        "semantic_rank": result["semantic_hold"]["rank"],
        "semantic_top1": result["semantic_hold"]["top1"],
        "semantic_top5": result["semantic_hold"]["top5"],
        "semantic_top10": result["semantic_hold"]["top10"],
        "cortical_rank": result["cortical_hold"]["rank"],
        "cortical_top1": result["cortical_hold"]["top1"],
        "cortical_top5": result["cortical_hold"]["top5"],
        "fusion_rank": result["fusion_hold"]["rank"],
        "fusion_top1": result["fusion_hold"]["top1"],
        "fusion_top5": result["fusion_hold"]["top5"],
        "fusion_top10": result["fusion_hold"]["top10"],
        "fusion_minus_semantic_rank": result["fusion_minus_semantic_rank"],
        "fusion_minus_semantic_top5": result["fusion_minus_semantic_top5"],
        "oracle_weight": result["oracle_fusion_hold"]["weight"],
        "oracle_rank": result["oracle_fusion_hold"]["rank"],
    }
    return row


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXT_DIR)
    parser.add_argument("--predictor-dir", type=Path, default=DEFAULT_PREDICTOR_DIR)
    parser.add_argument("--predictor", default="clip_vith14_train_test.npz")
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 11, 77, 101])
    parser.add_argument("--channel-set", default="posterior_P_PO_O")
    parser.add_argument("--time-pool", type=int, default=5)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--fusion-weights", default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0")
    args = parser.parse_args()

    alphas = [float(value) for value in args.alphas.split(",") if value]
    fusion_weights = [float(value) for value in args.fusion_weights.split(",") if value]
    predictor_path = args.predictor_dir / args.predictor
    results = [
        evaluate_seed(
            seed=seed,
            ext_dir=args.external_dir,
            predictor_path=predictor_path,
            subjects=args.subjects,
            memmap_dir=args.eeg_memmap_dir,
            data_root=args.data_root,
            channel_set=args.channel_set,
            time_pool=args.time_pool,
            alphas=alphas,
            fusion_weights=fusion_weights,
        )
        for seed in args.seeds
    ]
    rows = [flat_row(result) for result in results]
    df = pd.DataFrame(rows)
    numeric = df.drop(columns=["seed"]).astype(float)
    out_df = pd.concat(
        [df, pd.DataFrame([{"seed": "mean", **numeric.mean().to_dict()}, {"seed": "std", **numeric.std(ddof=1).to_dict()}])],
        ignore_index=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    json_path = args.out_dir / "summary.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"results": results, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    columns = [
        "seed",
        "semantic_rank",
        "semantic_top5",
        "cortical_rank",
        "fusion_weight",
        "fusion_rank",
        "fusion_top5",
        "fusion_minus_semantic_rank",
        "oracle_weight",
        "oracle_rank",
    ]
    mean = out_df[out_df["seed"] == "mean"].iloc[0]
    report = f"""# EEG CLIP Retrieval with Cortical Reranking

## Protocol

- Same image-heldout THINGS-fMRI overlap splits as the real-fMRI validation.
- Semantic baseline: ridge from image-averaged posterior EEG to CLIP ViT-H/14 image features.
- Cortical branch: trainable factorized-query EEG -> real-fMRI visual ROI prediction.
- Fusion: row-zscored semantic similarity plus cortical similarity; fusion weight selected on validation split only.

## Results

{markdown_table(out_df, columns)}

## Interpretation

- Semantic EEG->CLIP retrieval is already strong on the overlap splits.
- The cortical branch alone retrieves images above chance because it predicts real visual-fMRI patterns.
- Validation-selected fusion currently gives mean rank {mean['fusion_rank']:.4f} vs semantic-only {mean['semantic_rank']:.4f}. This is {'an improvement' if mean['fusion_minus_semantic_rank'] > 0 else 'not an improvement'} under the current fusion protocol.
- The oracle column shows whether a useful fusion weight exists on heldout; it should not be used for claims, only for diagnosing whether validation selection is the bottleneck.

Artifacts:

- `{csv_path}`
- `{json_path}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
