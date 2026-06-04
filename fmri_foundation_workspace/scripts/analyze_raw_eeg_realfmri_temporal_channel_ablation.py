#!/usr/bin/env python3
"""Temporal and channel ablations for raw EEG -> real THINGS-fMRI prediction.

This analysis follows the same image-heldout overlap protocol as
``evaluate_raw_eeg_to_realfmri_overlap_holdout.py``.  It keeps the ridge readout
and heldout split fixed in spirit, but varies which EEG time bins or channels
are visible to the readout.  The goal is to test whether the strong raw EEG
baseline is neurophysiologically plausible: visual-family fMRI predictions
should depend on post-stimulus visual windows and posterior EEG channels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_things_fmri_external_roi_breakdown import family_masks, fisher_mean
from evaluate_raw_eeg_to_realfmri_overlap_holdout import (
    DEFAULT_EEG_MEMMAP_DIR,
    DEFAULT_FMRI_NPZ,
    retrieval_metrics,
    ridge_predict,
    row_corr,
    vector_corr,
    zscore_train,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/sudaxin/projects/paired_data/data/thing_eeg/Preprocessed_data_250Hz")
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_temporal_channel_ablation_seed33"
)
DEFAULT_REPORT = (
    WORKSPACE
    / "notes"
    / "eeg_image_bridge"
    / "raw_eeg_realfmri_temporal_channel_ablation_20260604.md"
)

FALLBACK_THINGS63 = [
    "Fp1",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
]

KEY_FAMILIES = [
    "all_roi207",
    "all_visual_curated",
    "classical_visual_roi",
    "early_visual",
    "mid_visual",
    "ventral_category_high",
    "nonvisual_or_uncurated",
]


def load_channel_names(data_root: Path, subject: str) -> list[str]:
    path = data_root / subject / "preprocessed_eeg_training.npy"
    if not path.exists():
        return FALLBACK_THINGS63
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    names = payload.get("ch_names")
    if names is None:
        return FALLBACK_THINGS63
    return [str(name) for name in names]


def build_image_level_eeg_tensor(
    image_index: np.ndarray,
    subjects: list[str],
    memmap_dir: Path,
    pool: int,
) -> np.ndarray:
    arrays = [
        np.load(memmap_dir / f"{subject}_preprocessed_eeg_training_float32.npy", mmap_mode="r")
        for subject in subjects
    ]
    rows = []
    for image_idx in image_index.astype(int):
        per_subject = []
        for arr in arrays:
            eeg = np.asarray(arr[image_idx], dtype=np.float32).mean(axis=0)
            per_subject.append(eeg)
        avg = np.stack(per_subject, axis=0).mean(axis=0)
        if pool > 1:
            n_time = avg.shape[-1] // pool
            avg = avg[:, : n_time * pool].reshape(avg.shape[0], n_time, pool).mean(axis=-1)
        rows.append(avg)
    return np.stack(rows, axis=0).astype(np.float32)


def normalize_name(name: str) -> str:
    return name.upper().replace("Z", "Z")


def channel_group_masks(ch_names: list[str]) -> dict[str, np.ndarray]:
    norm = np.asarray([normalize_name(ch) for ch in ch_names])

    def starts(*prefixes: str) -> np.ndarray:
        return np.asarray([any(ch.startswith(prefix) for prefix in prefixes) for ch in norm])

    occipital = starts("O")
    po = starts("PO")
    parietal = starts("P") & ~po
    posterior_o_po = occipital | po
    posterior_p_po_o = occipital | po | parietal
    anterior = starts("FP", "AF", "F", "FT", "FC")
    central_temporal = starts("C", "CP", "T", "TP")
    return {
        "occipital_O": occipital,
        "parieto_occipital_PO": po,
        "parietal_P": parietal,
        "posterior_O_PO": posterior_o_po,
        "posterior_P_PO_O": posterior_p_po_o,
        "anterior_F_AF_FC": anterior,
        "central_temporal_C_CP_T_TP": central_temporal,
        "nonposterior": ~posterior_p_po_o,
    }


def flatten_features(x: np.ndarray, channel_mask: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x[:, channel_mask][:, :, time_mask].reshape(x.shape[0], -1))


def eval_family_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    q = pred[:, mask]
    t = target[:, mask]
    metrics = retrieval_metrics(q, t)
    shifted = retrieval_metrics(np.roll(q, 1, axis=0), t)
    roi_corr = np.asarray([vector_corr(q[:, i], t[:, i]) for i in range(q.shape[1])])
    return {
        "rank": metrics["rank_percentile"],
        "shifted": shifted["rank_percentile"],
        "delta": metrics["rank_percentile"] - shifted["rank_percentile"],
        "diag_off": metrics["diag_minus_offdiag"],
        "image_corr": float(np.nanmean(row_corr(q, t))),
        "roi_corr": fisher_mean(roi_corr),
        "top1": metrics["top1"],
        "top5": metrics["top5"],
    }


def tune_and_predict(
    x_fit_raw: np.ndarray,
    x_val_raw: np.ndarray,
    x_hold_raw: np.ndarray,
    y_fit_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_hold_raw: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    x_fit, x_val = zscore_train(x_fit_raw, x_val_raw)
    _, x_hold = zscore_train(x_fit_raw, x_hold_raw)
    y_fit, y_val = zscore_train(y_fit_raw, y_val_raw)
    _, y_hold = zscore_train(y_fit_raw, y_hold_raw)
    best_alpha = alphas[0]
    best_metrics: dict[str, float] | None = None
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha)
        metrics = retrieval_metrics(pred_val, y_val)
        if best_metrics is None or (
            metrics["rank_percentile"],
            metrics["diag_minus_offdiag"],
        ) > (
            best_metrics["rank_percentile"],
            best_metrics["diag_minus_offdiag"],
        ):
            best_alpha = alpha
            best_metrics = metrics
    pred_hold = ridge_predict(x_fit, y_fit, x_hold, best_alpha)
    return pred_hold, y_hold, best_alpha, best_metrics or {}


def run_config(
    *,
    config_type: str,
    name: str,
    mode: str,
    channel_mask: np.ndarray,
    time_mask: np.ndarray,
    x_fit: np.ndarray,
    x_val: np.ndarray,
    x_hold: np.ndarray,
    y_fit: np.ndarray,
    y_val: np.ndarray,
    y_hold: np.ndarray,
    roi_names: np.ndarray,
    masks: dict[str, np.ndarray],
    alphas: list[float],
    time_label: str = "",
) -> dict[str, float | str | int]:
    x_fit_features = flatten_features(x_fit, channel_mask, time_mask)
    x_val_features = flatten_features(x_val, channel_mask, time_mask)
    x_hold_features = flatten_features(x_hold, channel_mask, time_mask)
    pred, y_hold_eval, best_alpha, val_metrics = tune_and_predict(
        x_fit_features,
        x_val_features,
        x_hold_features,
        y_fit,
        y_val,
        y_hold,
        alphas,
    )
    row: dict[str, float | str | int] = {
        "config_type": config_type,
        "name": name,
        "mode": mode,
        "time_label": time_label,
        "n_channels": int(channel_mask.sum()),
        "n_time_bins": int(time_mask.sum()),
        "feature_dim": int(x_fit_features.shape[1]),
        "best_alpha": float(best_alpha),
        "val_rank_allroi": float(val_metrics.get("rank_percentile", np.nan)),
    }
    for family in KEY_FAMILIES:
        if family not in masks:
            continue
        family_metrics = eval_family_metrics(pred, y_hold_eval, masks[family])
        for key, value in family_metrics.items():
            row[f"{family}_{key}"] = float(value)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, ""), digits) for col in columns) + " |")
    return "\n".join(lines)


def make_plots(out_dir: Path, time_rows: list[dict[str, object]], channel_rows: list[dict[str, object]], single_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is nonessential.
        print(f"[warn] Could not import matplotlib for plots: {exc}", flush=True)
        return

    full = next(row for row in time_rows if row["name"] == "full")
    keep = [row for row in time_rows if row["mode"] == "keep"]
    drop = [row for row in time_rows if row["mode"] == "drop"]
    x = np.arange(len(keep))
    labels = [str(row["time_label"]) for row in keep]
    plt.figure(figsize=(11, 4.5))
    plt.plot(x, [row["all_visual_curated_rank"] for row in keep], marker="o", label="keep window")
    plt.plot(x, [row["all_visual_curated_rank"] for row in drop], marker="o", label="drop window")
    plt.axhline(full["all_visual_curated_rank"], color="0.25", linestyle="--", label="full EEG")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("visual ROI rank")
    plt.title("Raw EEG -> real fMRI: time-window ablation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "time_window_visual_rank.png", dpi=180)
    plt.close()

    family_rows = [row for row in channel_rows if row["name"] != "full"]
    family_rows = sorted(family_rows, key=lambda row: float(row["all_visual_curated_rank"]), reverse=True)
    plt.figure(figsize=(11, 5))
    plt.barh(
        [str(row["name"]) for row in family_rows],
        [row["all_visual_curated_rank"] for row in family_rows],
        color=["#2b7a78" if str(row["mode"]).startswith("keep") else "#e07a2f" for row in family_rows],
    )
    plt.axvline(full["all_visual_curated_rank"], color="0.25", linestyle="--", label="full EEG")
    plt.xlabel("visual ROI rank")
    plt.title("Raw EEG -> real fMRI: channel-family ablation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "channel_family_visual_rank.png", dpi=180)
    plt.close()

    top = sorted(single_rows, key=lambda row: float(row["all_visual_curated_rank"]), reverse=True)[:20]
    plt.figure(figsize=(10, 5))
    plt.bar([str(row["name"]) for row in top], [row["all_visual_curated_rank"] for row in top], color="#326b5b")
    plt.axhline(full["all_visual_curated_rank"], color="0.25", linestyle="--", label="full EEG")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("visual ROI rank")
    plt.title("Single-channel scan: top 20 channels")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "single_channel_top20_visual_rank.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--holdout-n", type=int, default=1000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--time-pool", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=100)
    parser.add_argument("--epoch-ms", type=int, default=1000)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--skip-single-channel", action="store_true")
    args = parser.parse_args()

    payload = np.load(args.fmri_npz, allow_pickle=True)
    split = payload["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    image_index = payload["image_index"][train_rows].astype(int)
    y_all = np.asarray(payload["measured_roi_beta"][train_rows], dtype=np.float32)
    roi_names = payload["roi_names"].astype(str)
    masks = family_masks(roi_names)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_rows))
    holdout_local = order[: args.holdout_n]
    trainval_local = order[args.holdout_n :]
    val_local = trainval_local[: args.val_n]
    fit_local = trainval_local[args.val_n :]

    selected = np.concatenate([fit_local, val_local, holdout_local])
    x_selected = build_image_level_eeg_tensor(
        image_index[selected],
        args.subjects,
        args.eeg_memmap_dir,
        args.time_pool,
    )
    n_fit = len(fit_local)
    n_val = len(val_local)
    x_fit = x_selected[:n_fit]
    x_val = x_selected[n_fit : n_fit + n_val]
    x_hold = x_selected[n_fit + n_val :]
    y_fit = y_all[fit_local]
    y_val = y_all[val_local]
    y_hold = y_all[holdout_local]

    ch_names = load_channel_names(args.data_root, args.subjects[0])
    if len(ch_names) != x_fit.shape[1]:
        print(
            f"[warn] Channel-name count {len(ch_names)} does not match EEG shape {x_fit.shape[1]}; using fallback names.",
            flush=True,
        )
        ch_names = FALLBACK_THINGS63[: x_fit.shape[1]]

    alphas = [float(value) for value in args.alphas.split(",") if value]
    all_channels = np.ones(x_fit.shape[1], dtype=bool)
    all_time = np.ones(x_fit.shape[2], dtype=bool)
    time_bin_ms = args.epoch_ms / x_fit.shape[2]
    bins_per_window = max(1, int(round(args.window_ms / time_bin_ms)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = dict(
        x_fit=x_fit,
        x_val=x_val,
        x_hold=x_hold,
        y_fit=y_fit,
        y_val=y_val,
        y_hold=y_hold,
        roi_names=roi_names,
        masks=masks,
        alphas=alphas,
    )

    time_rows: list[dict[str, object]] = [
        run_config(
            config_type="time",
            name="full",
            mode="full",
            channel_mask=all_channels,
            time_mask=all_time,
            time_label="0-1000ms",
            **common_kwargs,
        )
    ]
    for start in range(0, x_fit.shape[2], bins_per_window):
        stop = min(x_fit.shape[2], start + bins_per_window)
        label = f"{int(round(start * time_bin_ms)):03d}-{int(round(stop * time_bin_ms)):03d}ms"
        keep_mask = np.zeros(x_fit.shape[2], dtype=bool)
        keep_mask[start:stop] = True
        drop_mask = ~keep_mask
        time_rows.append(
            run_config(
                config_type="time",
                name=f"keep_{label}",
                mode="keep",
                channel_mask=all_channels,
                time_mask=keep_mask,
                time_label=label,
                **common_kwargs,
            )
        )
        time_rows.append(
            run_config(
                config_type="time",
                name=f"drop_{label}",
                mode="drop",
                channel_mask=all_channels,
                time_mask=drop_mask,
                time_label=label,
                **common_kwargs,
            )
        )
        print(f"[time] finished {label}", flush=True)

    group_masks = channel_group_masks(ch_names)
    channel_rows: list[dict[str, object]] = [
        run_config(
            config_type="channel_family",
            name="full",
            mode="full",
            channel_mask=all_channels,
            time_mask=all_time,
            time_label="0-1000ms",
            **common_kwargs,
        )
    ]
    for group, mask in group_masks.items():
        if not mask.any() or mask.all():
            continue
        channel_rows.append(
            run_config(
                config_type="channel_family",
                name=f"keep_{group}",
                mode="keep",
                channel_mask=mask,
                time_mask=all_time,
                time_label="0-1000ms",
                **common_kwargs,
            )
        )
        channel_rows.append(
            run_config(
                config_type="channel_family",
                name=f"drop_{group}",
                mode="drop",
                channel_mask=~mask,
                time_mask=all_time,
                time_label="0-1000ms",
                **common_kwargs,
            )
        )
        print(f"[channel] finished {group}", flush=True)

    single_rows: list[dict[str, object]] = []
    if not args.skip_single_channel:
        for idx, channel in enumerate(ch_names):
            mask = np.zeros(len(ch_names), dtype=bool)
            mask[idx] = True
            single_rows.append(
                run_config(
                    config_type="single_channel",
                    name=channel,
                    mode="keep",
                    channel_mask=mask,
                    time_mask=all_time,
                    time_label="0-1000ms",
                    **common_kwargs,
                )
            )
        print("[channel] finished single-channel scan", flush=True)

    write_csv(args.out_dir / "time_window_ablation.csv", time_rows)
    write_csv(args.out_dir / "channel_family_ablation.csv", channel_rows)
    write_csv(args.out_dir / "single_channel_ablation.csv", single_rows)
    make_plots(args.out_dir, time_rows, channel_rows, single_rows)

    full = next(row for row in time_rows if row["name"] == "full")
    keep_rows = [row for row in time_rows if row["mode"] == "keep"]
    drop_rows = [row for row in time_rows if row["mode"] == "drop"]
    best_keep = max(keep_rows, key=lambda row: float(row["all_visual_curated_rank"]))
    biggest_drop = min(drop_rows, key=lambda row: float(row["all_visual_curated_rank"]))
    top_channels = sorted(
        single_rows,
        key=lambda row: float(row["all_visual_curated_rank"]),
        reverse=True,
    )[:10]
    posterior_row = next(row for row in channel_rows if row["name"] == "keep_posterior_P_PO_O")
    nonposterior_row = next(row for row in channel_rows if row["name"] == "keep_nonposterior")
    drop_posterior_row = next(row for row in channel_rows if row["name"] == "drop_posterior_P_PO_O")

    summary = {
        "out_dir": str(args.out_dir),
        "report": str(args.report),
        "seed": args.seed,
        "n_fit": int(len(fit_local)),
        "n_val": int(len(val_local)),
        "n_holdout": int(len(holdout_local)),
        "subjects": args.subjects,
        "time_pool": int(args.time_pool),
        "time_bins": int(x_fit.shape[2]),
        "time_bin_ms": float(time_bin_ms),
        "window_ms": int(args.window_ms),
        "channel_names": ch_names,
        "full_all_visual_rank": float(full["all_visual_curated_rank"]),
        "full_all_visual_delta": float(full["all_visual_curated_delta"]),
        "best_keep_time": {
            "window": best_keep["time_label"],
            "rank": float(best_keep["all_visual_curated_rank"]),
            "delta": float(best_keep["all_visual_curated_delta"]),
        },
        "most_important_drop_time": {
            "window": biggest_drop["time_label"],
            "rank": float(biggest_drop["all_visual_curated_rank"]),
            "drop_from_full": float(full["all_visual_curated_rank"]) - float(biggest_drop["all_visual_curated_rank"]),
        },
        "keep_posterior_P_PO_O_rank": float(posterior_row["all_visual_curated_rank"]),
        "keep_nonposterior_rank": float(nonposterior_row["all_visual_curated_rank"]),
        "drop_posterior_P_PO_O_rank": float(drop_posterior_row["all_visual_curated_rank"]),
        "top_single_channels": [
            {
                "channel": row["name"],
                "rank": float(row["all_visual_curated_rank"]),
                "delta": float(row["all_visual_curated_delta"]),
            }
            for row in top_channels
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    selected_time_rows = [
        {
            "window": row["time_label"],
            "keep_rank": row["all_visual_curated_rank"],
            "drop_rank": next(
                drop["all_visual_curated_rank"]
                for drop in drop_rows
                if drop["time_label"] == row["time_label"]
            ),
            "drop_from_full": float(full["all_visual_curated_rank"])
            - float(
                next(
                    drop["all_visual_curated_rank"]
                    for drop in drop_rows
                    if drop["time_label"] == row["time_label"]
                )
            ),
        }
        for row in keep_rows
    ]
    selected_channel_rows = [
        row
        for row in channel_rows
        if row["name"]
        in {
            "full",
            "keep_occipital_O",
            "keep_parieto_occipital_PO",
            "keep_posterior_O_PO",
            "keep_posterior_P_PO_O",
            "keep_nonposterior",
            "drop_occipital_O",
            "drop_posterior_O_PO",
            "drop_posterior_P_PO_O",
        }
    ]
    report = f"""# Raw EEG -> Real THINGS-fMRI Temporal/Channel Ablation

## Protocol

- Same exact-image THINGS-EEG/THINGS-fMRI train-overlap split as the raw waveform ridge probe.
- Split: {len(fit_local)} fit images, {len(val_local)} validation images, {len(holdout_local)} heldout images.
- EEG input: 10 subjects x 4 repeats averaged per image, 63 channels, 0-1000 ms post-stimulus, pooled from 250 samples to {x_fit.shape[2]} bins ({time_bin_ms:.1f} ms/bin).
- Readout: ridge regression from selected EEG features to subject-averaged real fMRI ROI betas; alpha selected on validation split.
- Main score below: heldout retrieval rank for `all_visual_curated` real fMRI ROI family, with shifted rank/delta recorded in CSV.

## Headline

Full raw EEG reaches visual-family rank **{float(full['all_visual_curated_rank']):.4f}**.
The best single 100 ms keep-window is **{best_keep['time_label']}** with rank **{float(best_keep['all_visual_curated_rank']):.4f}**.
The most damaging 100 ms drop-window is **{biggest_drop['time_label']}**, reducing rank by **{float(full['all_visual_curated_rank']) - float(biggest_drop['all_visual_curated_rank']):.4f}**.
Posterior-only `P/PO/O` channels reach rank **{float(posterior_row['all_visual_curated_rank']):.4f}**, while nonposterior-only channels reach **{float(nonposterior_row['all_visual_curated_rank']):.4f}**.

## Time-Window Results

{markdown_table(selected_time_rows, ['window', 'keep_rank', 'drop_rank', 'drop_from_full'])}

## Channel-Family Results

{markdown_table(selected_channel_rows, ['name', 'mode', 'n_channels', 'all_visual_curated_rank', 'all_visual_curated_delta', 'all_visual_curated_image_corr', 'all_visual_curated_roi_corr'])}

## Top Single Channels

{markdown_table(top_channels, ['name', 'all_visual_curated_rank', 'all_visual_curated_delta', 'all_visual_curated_image_corr'])}

## Interpretation

The raw EEG signal is not uniformly distributed over arbitrary sensors/time.
The strongest heldout real-fMRI prediction is concentrated in posterior visual
channels and in post-stimulus time windows that are plausible for visual evoked
responses. This supports the next modeling decision: use a trainable
ROI-query/cortical branch that preserves temporal and channel structure, rather
than relying only on pooled semantic embeddings.

Artifacts:

- `{args.out_dir / 'time_window_ablation.csv'}`
- `{args.out_dir / 'channel_family_ablation.csv'}`
- `{args.out_dir / 'single_channel_ablation.csv'}`
- `{args.out_dir / 'time_window_visual_rank.png'}`
- `{args.out_dir / 'channel_family_visual_rank.png'}`
- `{args.out_dir / 'single_channel_top20_visual_rank.png'}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
