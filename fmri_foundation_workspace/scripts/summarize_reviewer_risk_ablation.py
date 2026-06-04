#!/usr/bin/env python3
"""Summarize V-JEPA target-space control and cortical-structure evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from evaluate_target_space_ablation import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_VJEPA_DIR,
    choose_ridge_alpha,
    load_clip,
    load_mean_atm_embeddings,
    load_npz_features,
    load_subjects,
    metric_block,
    norm_rows,
    ridge_fit_predict,
)


WORKSPACE = Path(__file__).resolve().parents[1]
RESULT_ROOT = WORKSPACE / "results" / "eeg_image_bridge"
NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "reviewer_risk_vjepa_vs_cortical_structure_20260604.md"


def per_image_rank(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = norm_rows(query)
    target = norm_rows(target)
    sims = query @ target.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    rank_pct = 1.0 - (ranks - 1) / max(len(query) - 1, 1)
    top1 = (ranks <= 1).astype("float64")
    return rank_pct.astype("float64"), top1


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 33, n_boot: int = 5000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, len(diff)))
    signflip = (diff[None, :] * signs).mean(axis=1)
    obs = float(diff.mean())
    return {
        "diff": obs,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "signflip_p": float((np.sum(np.abs(signflip) >= abs(obs)) + 1) / (n_boot + 1)),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_space_bootstrap(size: int) -> dict[str, object]:
    rows = json.loads((RESULT_ROOT / "target_space_ablation" / "target_space_ablation_metrics.json").read_text())["rows"]
    selected = [
        row
        for row in rows
        if row["train_images"] == size and row["model"] == "frozen_plus_atm_to_vjepa_to_clip"
    ][0]
    subjects = load_subjects(DEFAULT_ASSET_ROOT)
    x_train, x_test = load_mean_atm_embeddings(DEFAULT_ASSET_ROOT, subjects)
    clip_train, clip_test = load_clip(DEFAULT_ASSET_ROOT)
    vjepa_train, vjepa_test = load_npz_features(
        DEFAULT_VJEPA_DIR / "vjepa2_features_train_merged_n16540.npz",
        DEFAULT_VJEPA_DIR / "vjepa2_features_test_merged_n200.npz",
    )
    x_size = x_train[:size]
    alpha = float(selected["alpha"])
    map_alpha = float(selected["map_alpha"])
    blend = float(selected["blend"])
    pred_v = ridge_fit_predict(x_size, vjepa_train[:size], x_test, alpha)
    pred_clip = ridge_fit_predict(vjepa_train[:size], clip_train[:size], pred_v, map_alpha)
    fused = norm_rows((1.0 - blend) * x_test + blend * pred_clip)
    frozen_rank, frozen_top1 = per_image_rank(x_test, clip_test)
    fused_rank, fused_top1 = per_image_rank(fused, clip_test)
    return {
        "size": size,
        "alpha": alpha,
        "map_alpha": map_alpha,
        "blend": blend,
        "frozen": metric_block(x_test, clip_test),
        "fused": metric_block(fused, clip_test),
        "rank_diff": bootstrap_diff(fused_rank, frozen_rank),
        "top1_diff": bootstrap_diff(fused_top1, frozen_top1),
    }


def select_run(summary: dict[str, object], run_contains: str) -> dict[str, object]:
    for row in summary["runs"]:
        if run_contains in row["run"]:
            return row
    raise KeyError(run_contains)


def cortical_binding_summary() -> dict[str, dict[str, object]]:
    raw = json.loads(
        (RESULT_ROOT / "atm_roi_query_target_confusion" / "raw_query_target_n16540_seed33_final_cpu_v2" / "summary.json").read_text()
    )
    residual = json.loads(
        (
            RESULT_ROOT
            / "atm_roi_query_target_confusion"
            / "residual_query_target_n16540_seed33_lam005_best_roi_cpu_v2"
            / "summary.json"
        ).read_text()
    )
    return {
        "raw_group": select_run(raw, "spatial_group_raw"),
        "raw_parcel_clean": select_run(raw, "spatial_parcel_raw_train"),
        "raw_parcel_strong": select_run(raw, "raw_strongroi"),
        "residual_group": select_run(residual, "spatial_group_clip_residual"),
        "residual_parcel": select_run(residual, "spatial_parcel_clip_residual"),
    }


def group_time_rows(path: Path) -> list[dict[str, str]]:
    wanted = {"all_visual", "early_calcarine", "early_visual_combined", "high_level_combined"}
    return [row for row in read_csv(path) if row["roi_group"] in wanted]


def channel_top(path: Path, top_k: int = 8) -> dict[str, object]:
    rows = read_csv(path)
    by_channel: dict[str, list[float]] = {}
    for row in rows:
        by_channel.setdefault(row["channel"], []).append(float(row["attention_mean"]))
    means = {ch: float(np.mean(vals)) for ch, vals in by_channel.items()}
    ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
    posterior = [
        value
        for ch, value in means.items()
        if ch.startswith("O") or ch.startswith("PO") or ch in {"P7", "P5", "P3", "Pz", "P4", "P6", "P8"}
    ]
    return {
        "top": ordered[:top_k],
        "posterior_mean": float(np.mean(posterior)),
        "all_mean": float(np.mean(list(means.values()))),
        "posterior_over_all": float(np.mean(posterior) / max(np.mean(list(means.values())), 1e-8)),
    }


def main() -> int:
    target_boot = [target_space_bootstrap(size) for size in (8192, 16540)]
    binding = cortical_binding_summary()
    raw_time = group_time_rows(
        RESULT_ROOT
        / "atm_roi_query_time_dependency"
        / "raw_query_time_n16540_seed33_cpu"
        / "atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010"
        / "best_windows_by_group.csv"
    )
    residual_time = group_time_rows(
        RESULT_ROOT
        / "atm_roi_query_time_dependency"
        / "residual_query_time_n16540_seed33_cpu"
        / "atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005"
        / "best_windows_by_group.csv"
    )
    raw_channel = channel_top(
        RESULT_ROOT
        / "atm_roi_query_attention_maps"
        / "raw_attention_n16540_seed33_final_cpu"
        / "atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010"
        / "query_channel_attention_mean.csv"
    )
    residual_channel = channel_top(
        RESULT_ROOT
        / "atm_roi_query_attention_maps"
        / "residual_attention_n16540_seed33_lam005_final_cpu"
        / "atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005"
        / "query_channel_attention_mean.csv"
    )

    out_json = RESULT_ROOT / "target_space_ablation" / "reviewer_risk_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "target_space_bootstrap": target_boot,
                "cortical_binding": binding,
                "raw_time": raw_time,
                "residual_time": residual_time,
                "raw_channel": raw_channel,
                "residual_channel": residual_channel,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Reviewer-Risk Summary: V-JEPA Target Control vs Cortical Structure",
        "",
        "## Target-Space Control",
        "",
        "| train images | model | CLIP top1 | CLIP top5 | rank | top1 gain vs frozen | rank gain vs frozen |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in target_boot:
        frozen = item["frozen"]
        fused = item["fused"]
        lines.append(
            f"| {item['size']} | frozen ATM->CLIP | {frozen['top1']:.4f} | {frozen['top5']:.4f} | {frozen['rank_percentile']:.4f} | 0 | 0 |"
        )
        lines.append(
            f"| {item['size']} | frozen + ATM->V-JEPA->CLIP | {fused['top1']:.4f} | {fused['top5']:.4f} | {fused['rank_percentile']:.4f} | "
            f"{item['top1_diff']['diff']:.4f} CI {item['top1_diff']['ci95']} | "
            f"{item['rank_diff']['diff']:.4f} CI {item['rank_diff']['ci95']} |"
        )
    lines.extend(
        [
            "",
            "Readout: V-JEPA target-space supervision is a useful control, but by itself it is not a drop-in replacement for the CLIP retrieval/generation interface. The only CLIP-space gain appears after blending with the already strong frozen ATM->CLIP embedding, and the gain is small.",
            "",
            "## Query-Target Binding",
            "",
            "| condition | n ROI | diag-offdiag | diag rank pct | within-between group | target-geometry corr | shuffled diag-offdiag mean/std |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["raw_group", "raw_parcel_clean", "raw_parcel_strong", "residual_group", "residual_parcel"]:
        row = binding[name]
        lines.append(
            f"| {name} | {int(row['n_roi'])} | {row['diag_minus_offdiag_mean']:.4f} | "
            f"{row['diag_rank_percentile_mean']:.4f} | {row['same_minus_other_group_offdiag']:.4f} | "
            f"{row['query_target_vs_target_self_matrix_corr_all']:.4f} | "
            f"{row['shuffled_diag_minus_offdiag_mean']:.4f}/{row['shuffled_diag_minus_offdiag_std']:.4f} |"
        )
    lines.extend(["", "## Time-Window Structure", "", "| target | group | full corr | best keep | most important drop | drop delta |", "|---|---|---:|---|---|---:|"])
    for target, rows in [("raw parcel strong", raw_time), ("residual parcel", residual_time)]:
        for row in rows:
            lines.append(
                f"| {target} | {row['roi_group']} | {float(row['full_mean_corr_signal']):.4f} | "
                f"{row['best_keep_window']} | {row['most_important_drop_window']} | {float(row['mean_drop_delta_from_full']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Channel Structure",
            "",
            f"- Raw strong parcel top channels: {raw_channel['top']}; posterior/all mean ratio = {raw_channel['posterior_over_all']:.2f}.",
            f"- Residual parcel top channels: {residual_channel['top']}; posterior/all mean ratio = {residual_channel['posterior_over_all']:.2f}.",
            "",
            "## Verdict",
            "",
            "- The best reviewer-facing story is not pure retrieval gain. V-JEPA-only control does not replace CLIP; it gives weak auxiliary evidence.",
            "- Raw cortical targets preserve the clearest interpretable cortical structure: strong query-target binding, posterior EEG channels, and staged time-window effects.",
            "- Residual targets keep query identity/group geometry above shuffled null, but their visible cortical maps are weaker. Use residual for rigor, raw for visualization and intuition.",
        ]
    )
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"note": str(NOTE), "json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
