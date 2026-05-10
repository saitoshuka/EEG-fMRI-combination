#!/usr/bin/env python3
"""Evaluate whether ATM ROI predictions can improve image retrieval by reranking.

This script loads existing ATM semantic/spatial checkpoints, recomputes test
EEG predictions, and scores candidate images with:

    final = zscore(semantic EEG-CLIP similarity) + alpha * zscore(spatial ROI similarity)

Spatial reranking is evaluated both as a full-test exploratory sweep and with a
simple val/heldout split for alpha/topK selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_late_fusion"
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")


def row_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + eps)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def rank_metrics(scores: np.ndarray, row_idx: np.ndarray | None = None) -> dict[str, float]:
    if row_idx is None:
        row_idx = np.arange(scores.shape[0])
    sub = scores[row_idx]
    diag = sub[np.arange(len(row_idx)), row_idx]
    ranks = (sub > diag[:, None]).sum(axis=1) + 1
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "rank": float((1.0 - (ranks - 1) / max(scores.shape[1] - 1, 1)).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def shifted_metrics(scores: np.ndarray, row_idx: np.ndarray | None = None) -> dict[str, float]:
    if row_idx is None:
        row_idx = np.arange(scores.shape[0])
    shifted = (row_idx + max(1, scores.shape[0] // 3)) % scores.shape[0]
    sub = scores[row_idx]
    diag = sub[np.arange(len(row_idx)), shifted]
    ranks = (sub > diag[:, None]).sum(axis=1) + 1
    return {
        "shifted_top1": float((ranks == 1).mean()),
        "shifted_top5": float((ranks <= 5).mean()),
        "shifted_rank": float((1.0 - (ranks - 1) / max(scores.shape[1] - 1, 1)).mean()),
    }


def topk_rerank_scores(semantic: np.ndarray, spatial: np.ndarray, alpha: float, topk: int) -> np.ndarray:
    semantic_z = row_zscore(semantic)
    spatial_z = row_zscore(spatial)
    if topk <= 0 or topk >= semantic.shape[1]:
        return semantic_z + alpha * spatial_z
    final = semantic_z.copy()
    idx = np.argpartition(-semantic_z, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(semantic.shape[0])[:, None]
    final[rows, idx] = semantic_z[rows, idx] + alpha * spatial_z[rows, idx]
    return final


def rescue_damage(
    base_scores: np.ndarray,
    fused_scores: np.ndarray,
    row_idx: np.ndarray | None = None,
) -> dict[str, int]:
    if row_idx is None:
        row_idx = np.arange(base_scores.shape[0])
    labels = row_idx
    base_top = base_scores[row_idx].argmax(axis=1)
    fused_top = fused_scores[row_idx].argmax(axis=1)
    base_ok = base_top == labels
    fused_ok = fused_top == labels
    return {
        "base_correct": int(base_ok.sum()),
        "fused_correct": int(fused_ok.sum()),
        "rescue": int((~base_ok & fused_ok).sum()),
        "damage": int((base_ok & ~fused_ok).sum()),
        "same_correct": int((base_ok & fused_ok).sum()),
    }


def load_roi_payload(path: Path, roi_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str | None]:
    payload = np.load(path, allow_pickle=True)
    target_key = "group_targets" if roi_kind == "group" else "parcel_targets"
    name_key = "group_names" if roi_kind == "group" else "parcel_names"
    count_key = "group_vertex_counts" if roi_kind == "group" else "parcel_vertex_counts"
    visual_group_json = str(payload["visual_group_json"].item()) if "visual_group_json" in payload.files else None
    return (
        payload[target_key].astype("float32"),
        payload[name_key],
        payload[count_key],
        payload["image_index"].astype(int),
        visual_group_json,
    )


def build_model_from_summary(
    summary: dict,
    roi_names: np.ndarray | None,
    vertex_counts: np.ndarray | None,
    group_features: torch.Tensor | None,
    use_spatial: bool,
) -> AtmSemanticSpatial:
    return AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        subject_mode=summary.get("subject_mode", "none"),
        atm_d_model=int(summary.get("atm_d_model", 256)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=summary.get("semantic_head", "shallow"),
        use_spatial=use_spatial,
    )


def predict_checkpoint(
    run_dir: Path,
    data_root: Path,
    image_root: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | dict | list[str] | str]:
    summary = json.loads((run_dir / "summary.json").read_text())
    roi_kind = summary["roi_kind"]
    use_spatial = summary["mode"] == "spatial"
    test_roi_path = Path(summary["test_roi"])
    if not test_roi_path.is_absolute():
        test_roi_path = WORKSPACE / test_roi_path
    train_roi_path = Path(summary["train_roi"])
    if not train_roi_path.is_absolute():
        train_roi_path = WORKSPACE.parent / train_roi_path

    roi_test, _, _, test_image_index, _ = load_roi_payload(test_roi_path, roi_kind)
    _, roi_names, vertex_counts, _, visual_group_json = load_roi_payload(train_roi_path, roi_kind)
    group_features = visual_group_features(roi_names, visual_group_json)
    subjects = list(summary["subjects"])

    model = build_model_from_summary(
        summary,
        roi_names if use_spatial else None,
        vertex_counts if use_spatial else None,
        group_features if use_spatial else None,
        use_spatial=use_spatial,
    ).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    clip_test = torch.load(image_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_test = F.normalize(clip_test, dim=-1).numpy()
    eeg_stack = load_or_build_test_eeg_stack(
        data_root,
        subjects,
        test_image_index,
        cache_dir=cache_dir,
        cache_tag=test_roi_path.stem,
    )

    sem_parts = []
    roi_parts = []
    with torch.no_grad():
        for subject_idx, subject in enumerate(subjects):
            eeg = eeg_stack[subject_idx]
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            subject_sem = []
            subject_roi = []
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                out = model(x, sids)
                subject_sem.append(out["semantic"].cpu())
                if "roi_pred" in out:
                    subject_roi.append(out["roi_pred"].cpu())
            sem_parts.append(torch.cat(subject_sem, dim=0))
            if subject_roi:
                roi_parts.append(torch.cat(subject_roi, dim=0))

    sem_pred = torch.stack(sem_parts, dim=0).mean(dim=0).numpy()
    result: dict[str, np.ndarray | dict | list[str] | str] = {
        "run_dir": str(run_dir),
        "summary": summary,
        "subjects": subjects,
        "semantic_pred": sem_pred,
        "clip_test": clip_test,
        "roi_test": roi_test,
    }
    if roi_parts:
        result["roi_pred"] = torch.stack(roi_parts, dim=0).mean(dim=0).numpy()
    return result


def write_matrix_png(path: Path, matrix: np.ndarray, title: str, max_rows: int = 80) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = matrix[:max_rows, :max_rows]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("candidate image")
    ax.set_ylabel("EEG trial")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_rescue_png(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r["source"]) for r in rows]
    rescue = [int(r["rescue"]) for r in rows]
    damage = [int(r["damage"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(x - 0.18, rescue, width=0.36, label="rescue")
    ax.bar(x + 0.18, damage, width=0.36, label="damage")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("test images")
    ax.set_title("Late-fusion top1 rescue vs damage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate_fusion(
    semantic_scores: np.ndarray,
    spatial_scores: np.ndarray,
    source: str,
    alphas: list[float],
    topks: list[int],
    val_idx: np.ndarray,
    heldout_idx: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str], np.ndarray]:
    rows: list[dict[str, float | int | str]] = []
    base_scores = row_zscore(semantic_scores)
    base_all = rank_metrics(base_scores)
    for alpha in alphas:
        for topk in topks:
            fused = topk_rerank_scores(semantic_scores, spatial_scores, alpha, topk)
            row: dict[str, float | int | str] = {
                "source": source,
                "alpha": float(alpha),
                "topk": int(topk),
                "split": "full",
                "base_top1": base_all["top1"],
                "base_top5": base_all["top5"],
                "base_rank": base_all["rank"],
                **{f"fused_{k}": v for k, v in rank_metrics(fused).items()},
                **shifted_metrics(fused),
                **rescue_damage(base_scores, fused),
            }
            rows.append(row)

    val_rows = []
    for alpha in alphas:
        for topk in topks:
            fused = topk_rerank_scores(semantic_scores, spatial_scores, alpha, topk)
            m = rank_metrics(fused, val_idx)
            val_rows.append((m["top1"], m["top5"], m["rank"], alpha, topk))
    val_rows.sort(reverse=True)
    _, _, _, best_alpha, best_topk = val_rows[0]
    heldout_fused = topk_rerank_scores(semantic_scores, spatial_scores, best_alpha, best_topk)
    heldout_base = rank_metrics(base_scores, heldout_idx)
    heldout = {
        "source": source,
        "alpha": float(best_alpha),
        "topk": int(best_topk),
        "split": "heldout",
        "base_top1": heldout_base["top1"],
        "base_top5": heldout_base["top5"],
        "base_rank": heldout_base["rank"],
        **{f"fused_{k}": v for k, v in rank_metrics(heldout_fused, heldout_idx).items()},
        **shifted_metrics(heldout_fused, heldout_idx),
        **rescue_damage(base_scores, heldout_fused, heldout_idx),
    }
    return rows, heldout, heldout_fused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--spatial-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="n4096")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--topks", default="0,10,20,50,100")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]

    semantic = predict_checkpoint(
        args.semantic_run,
        args.data_root,
        args.image_root,
        args.cache_dir,
        device,
        args.batch_size,
    )
    sem_score_strong = l2_normalize(semantic["semantic_pred"]) @ l2_normalize(semantic["clip_test"]).T  # type: ignore[index]
    n = sem_score_strong.shape[0]
    val_idx = np.arange(0, n, 2)
    heldout_idx = np.arange(1, n, 2)

    all_rows: list[dict[str, float | int | str]] = []
    heldout_rows: list[dict[str, float | int | str]] = []
    rescue_rows: list[dict[str, float | int | str]] = []
    matrices: dict[str, np.ndarray] = {"semantic_only_score": sem_score_strong}

    for spatial_run in args.spatial_runs:
        spatial = predict_checkpoint(
            spatial_run,
            args.data_root,
            args.image_root,
            args.cache_dir,
            device,
            args.batch_size,
        )
        if "roi_pred" not in spatial:
            continue
        roi_kind = str(spatial["summary"]["roi_kind"])  # type: ignore[index]
        spatial_score = l2_normalize(spatial["roi_pred"]) @ l2_normalize(spatial["roi_test"]).T  # type: ignore[index]
        sem_score_same = l2_normalize(spatial["semantic_pred"]) @ l2_normalize(spatial["clip_test"]).T  # type: ignore[index]
        matrices[f"{roi_kind}_spatial_score"] = spatial_score

        for source, sem_score in [
            (f"{roi_kind}_own_semantic_plus_roi", sem_score_same),
            (f"semantic_only_plus_{roi_kind}_roi", sem_score_strong),
        ]:
            rows, heldout, heldout_fused = evaluate_fusion(
                sem_score,
                spatial_score,
                source,
                alphas,
                topks,
                val_idx,
                heldout_idx,
            )
            all_rows.extend(rows)
            heldout_rows.append(heldout)
            rescue_rows.append(heldout)
            matrices[f"{source}_heldout_fused"] = heldout_fused

    csv_path = out_dir / "late_fusion_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        keys = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    heldout_path = out_dir / "late_fusion_heldout.csv"
    with heldout_path.open("w", newline="", encoding="utf-8") as f:
        keys = sorted({key for row in heldout_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(heldout_rows)

    np.savez_compressed(out_dir / "late_fusion_scores.npz", **matrices)

    for name, matrix in matrices.items():
        if name.endswith("score"):
            write_matrix_png(out_dir / f"{name}.png", row_zscore(matrix), name)
    if rescue_rows:
        write_rescue_png(out_dir / "heldout_rescue_damage.png", rescue_rows)

    summary = {
        "semantic_run": str(args.semantic_run),
        "spatial_runs": [str(p) for p in args.spatial_runs],
        "data_root": str(args.data_root),
        "image_root": str(args.image_root),
        "out_dir": str(out_dir),
        "n_test": int(n),
        "val_idx": "even rows",
        "heldout_idx": "odd rows",
        "alphas": alphas,
        "topks": topks,
        "semantic_only_full": rank_metrics(row_zscore(sem_score_strong)),
        "heldout_rows": heldout_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
