#!/usr/bin/env python3
"""CCA/PLS retrieval diagnostic for LaBraM EEG features vs NeuroSTORM latents.

This is deliberately not a deep supervised model.  It asks a simpler question:
do frozen LaBraM EEG representations identify the corresponding fMRI teacher
latent timepoint better than shifted or mean controls?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import choose_labram_channels, feature_from_tokens, labram_input_chans, load_labram_model  # noqa: E402
from pooled_deep import column_corr, make_subject_folds, resolve_device, row_corr  # noqa: E402
from pooled_raw import session_shift  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_official_natview_labram_eeg/run_cache"
DEFAULT_FEATURES = REPO_ROOT / "data/labram_neurostorm_retrieval_natview/features.npz"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_neurostorm_retrieval_natview"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"


@dataclass
class MetricRow:
    fold: int
    method: str
    target_mode: str
    n_train: int
    n_test: int
    test_subjects: str
    x_dim: int
    y_dim: int
    n_components: int
    latent_corr_mean: float
    latent_corr_median: float
    row_corr_mean: float
    row_corr_median: float
    r2: float
    retrieval_top1: float
    retrieval_top5: float
    retrieval_mrr: float
    retrieval_median_rank: float
    retrieval_rank_percentile_mean: float
    diag_minus_offdiag: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def extract_features(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    args.feature_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_labram_model(args.checkpoint, args.device)
    files = sorted(args.raw_cache_dir.glob("*/*.npz"))
    if args.max_runs > 0:
        files = files[: args.max_runs]

    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    subjects: list[str] = []
    runs: list[str] = []
    sample_ids: list[int] = []
    time_frac: list[float] = []
    manifest = []

    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    if patch_samples != 200:
        raise ValueError("LaBraM patch size must be 200 samples")
    n_patches = window_samples // patch_samples

    for i, path in enumerate(files, start=1):
        try:
            z = np.load(path, allow_pickle=True)
            channels = [str(x) for x in z["channels"]]
            indices, chosen = choose_labram_channels(channels, args.min_channels, args.max_channels)
            starts = z["sample_start"].astype(np.int64)
            raw = z["raw"].astype(np.float32)[indices]
            teacher = z["Z_neurostorm"].astype(np.float32).reshape(z["Z_neurostorm"].shape[0], -1)
            n = min(starts.size, teacher.shape[0])
            starts = starts[:n]
            teacher = teacher[:n]
            good = (starts >= 0) & (starts + window_samples <= raw.shape[1])
            starts = starts[good]
            teacher = teacher[good]
            if starts.size < args.min_windows:
                raise ValueError(f"too few windows: {starts.size}")
            input_chans = labram_input_chans(chosen).to(args.device)
            feats = []
            with torch.inference_mode():
                for start in range(0, starts.size, args.batch_size):
                    batch_starts = starts[start : start + args.batch_size]
                    windows = np.stack([raw[:, s : s + window_samples] for s in batch_starts], axis=0)
                    if args.window_zscore:
                        mean = windows.mean(axis=-1, keepdims=True)
                        std = windows.std(axis=-1, keepdims=True)
                        windows = (windows - mean) / np.maximum(std, 1e-6)
                    windows = windows.reshape(windows.shape[0], windows.shape[1], n_patches, patch_samples)
                    x = torch.from_numpy(windows.astype(np.float32)).to(args.device)
                    with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
                        tokens = model.forward_features(x, input_chans=input_chans, return_all_tokens=True)
                        feat = feature_from_tokens(tokens.float(), args.feature_mode)
                    feats.append(feat.cpu().numpy().astype(np.float32))
            feat_all = np.concatenate(feats, axis=0)
            xs.append(feat_all)
            zs.append(teacher.astype(np.float32))
            subject = scalar(z["subject"])
            run = scalar(z["run"])
            subjects.extend([subject] * starts.size)
            runs.extend([run] * starts.size)
            sample_ids.extend(range(starts.size))
            if "time_frac" in z.files:
                tf = np.asarray(z["time_frac"], dtype=np.float32)[:n][good]
            else:
                tf = np.linspace(0, 1, starts.size, dtype=np.float32)
            time_frac.extend([float(v) for v in tf])
            manifest.append(
                {
                    "path": str(path),
                    "subject": subject,
                    "run": run,
                    "windows": int(starts.size),
                    "channels": int(len(chosen)),
                    "feature_dim": int(feat_all.shape[1]),
                }
            )
            print(f"[{i:03d}/{len(files):03d}] features {subject} windows={starts.size} channels={len(chosen)}", flush=True)
        except Exception as exc:
            manifest.append({"path": str(path), "status": "error", "error": str(exc)})
            print(f"[{i:03d}/{len(files):03d}] error {path.name}: {exc}", flush=True)

    if not xs:
        raise RuntimeError("No features extracted")
    np.savez_compressed(
        args.feature_path,
        X=np.concatenate(xs, axis=0).astype(np.float32),
        Z=np.concatenate(zs, axis=0).astype(np.float32),
        subject=np.asarray(subjects, dtype="U32"),
        run=np.asarray(runs, dtype="U128"),
        sample_id=np.asarray(sample_ids, dtype=np.int32),
        time_frac=np.asarray(time_frac, dtype=np.float32),
        feature_mode=np.asarray(args.feature_mode, dtype="U32"),
        source_cache=np.asarray(str(args.raw_cache_dir), dtype="U512"),
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "feature_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({key for row in manifest for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"feature_path": str(args.feature_path), "windows": int(sum(x.shape[0] for x in xs))}, indent=2))


def time_basis(time_frac: np.ndarray, harmonics: int) -> np.ndarray:
    t = time_frac.reshape(-1, 1).astype(np.float32)
    parts = [np.ones_like(t), t, t * t]
    for k in range(1, harmonics + 1):
        parts.append(np.sin(2 * np.pi * k * t))
        parts.append(np.cos(2 * np.pi * k * t))
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_pca_standardized(y: np.ndarray, train_idx: np.ndarray, dim: int, seed: int):
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y[train_idx])
    n_comp = min(dim, y.shape[1], max(1, train_idx.size - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(y_train)
    y_pc = pca.transform(scaler.transform(y)).astype(np.float32)
    return y_pc, float(np.sum(pca.explained_variance_ratio_))


def reduce_x(x: np.ndarray, train_idx: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, int, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    if dim <= 0 or dim >= x.shape[1]:
        x_all = scaler.transform(x).astype(np.float32)
        return x_all, x_all.shape[1], 1.0
    n_comp = min(dim, x.shape[1], max(1, train_idx.size - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(x_train)
    return pca.transform(scaler.transform(x)).astype(np.float32), n_comp, float(np.sum(pca.explained_variance_ratio_))


def shift_targets_by_run(y: np.ndarray, run: np.ndarray, seed: int) -> np.ndarray:
    return session_shift(y, run.astype(str), seed)


def retrieval_metrics(query: np.ndarray, target: np.ndarray, run: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    ranks = []
    top1 = []
    top5 = []
    rr = []
    rank_pct = []
    diag_minus = []
    q = query.astype(np.float64)
    t = target.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    run_test = run[test_idx].astype(str)
    for r in sorted(set(run_test)):
        local = np.where(run_test == r)[0]
        if local.size < 5:
            continue
        sim = q[local] @ t[local].T
        diag = np.diag(sim)
        for i in range(local.size):
            greater = int(np.sum(sim[i] > sim[i, i]))
            equal_others = int(np.sum(np.isclose(sim[i], sim[i, i], rtol=1e-7, atol=1e-9))) - 1
            rank = float(1 + greater + 0.5 * max(0, equal_others))
            ranks.append(rank)
            top1.append(rank <= 1)
            top5.append(rank <= 5)
            rr.append(1.0 / rank)
            rank_pct.append(1.0 - (rank - 1) / max(1, local.size - 1))
        off = sim[~np.eye(local.size, dtype=bool)]
        diag_minus.append(float(diag.mean() - off.mean()))
    if not ranks:
        return {
            "top1": math.nan,
            "top5": math.nan,
            "mrr": math.nan,
            "median_rank": math.nan,
            "rank_percentile_mean": math.nan,
            "diag_minus_offdiag": math.nan,
        }
    return {
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "mrr": float(np.mean(rr)),
        "median_rank": float(np.median(ranks)),
        "rank_percentile_mean": float(np.mean(rank_pct)),
        "diag_minus_offdiag": float(np.mean(diag_minus)),
    }


def prediction_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    corr = column_corr(true, pred)
    rows = row_corr(true, pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "latent_corr_mean": float(np.nanmean(corr)),
        "latent_corr_median": float(np.nanmedian(corr)),
        "row_corr_mean": float(np.nanmean(rows)),
        "row_corr_median": float(np.nanmedian(rows)),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
    }


def add_row(
    rows: list[MetricRow],
    fold: int,
    method: str,
    target_mode: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_subjects: np.ndarray,
    x_dim: int,
    y_dim: int,
    n_components: int,
    pred: np.ndarray,
    true: np.ndarray,
    query_for_retrieval: np.ndarray,
    target_for_retrieval: np.ndarray,
    run: np.ndarray,
) -> None:
    pm = prediction_metrics(pred, true)
    rm = retrieval_metrics(query_for_retrieval, target_for_retrieval, run, test_idx)
    rows.append(
        MetricRow(
            fold=fold,
            method=method,
            target_mode=target_mode,
            n_train=int(train_idx.size),
            n_test=int(test_idx.size),
            test_subjects=" ".join(test_subjects.astype(str)),
            x_dim=int(x_dim),
            y_dim=int(y_dim),
            n_components=int(n_components),
            latent_corr_mean=pm["latent_corr_mean"],
            latent_corr_median=pm["latent_corr_median"],
            row_corr_mean=pm["row_corr_mean"],
            row_corr_median=pm["row_corr_median"],
            r2=pm["r2"],
            retrieval_top1=rm["top1"],
            retrieval_top5=rm["top5"],
            retrieval_mrr=rm["mrr"],
            retrieval_median_rank=rm["median_rank"],
            retrieval_rank_percentile_mean=rm["rank_percentile_mean"],
            diag_minus_offdiag=rm["diag_minus_offdiag"],
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.feature_path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    y_flat = z["Z"].astype(np.float32)
    subject = z["subject"].astype(str)
    run = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int32)
    time_frac = z["time_frac"].astype(np.float32)
    if args.split_mode == "subject":
        folds = []
        subject_folds = make_subject_folds(subject, args.folds, args.seed)
        if args.max_folds > 0:
            subject_folds = subject_folds[: args.max_folds]
        for fold_id, test_subjects in enumerate(subject_folds, start=1):
            test_mask = np.isin(subject, test_subjects)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            folds.append((fold_id, train_idx, test_idx, test_subjects))
    elif args.split_mode == "within_run_random":
        folds = []
        n_splits = args.max_folds if args.max_folds > 0 else args.folds
        for fold_id in range(1, n_splits + 1):
            rng = np.random.default_rng(args.seed + fold_id * 1009)
            train_parts = []
            test_parts = []
            for r in sorted(set(run.astype(str))):
                idx = np.where(run.astype(str) == r)[0]
                if idx.size < 10:
                    continue
                shuffled = idx.copy()
                rng.shuffle(shuffled)
                n_test = max(5, int(round(idx.size * args.within_run_test_frac)))
                test_parts.append(np.sort(shuffled[:n_test]))
                train_parts.append(np.sort(shuffled[n_test:]))
            train_idx = np.concatenate(train_parts)
            test_idx = np.concatenate(test_parts)
            folds.append((fold_id, train_idx, test_idx, np.asarray([f"within_run_random_{fold_id}"])))
    elif args.split_mode == "within_run_block":
        folds = []
        n_splits = args.max_folds if args.max_folds > 0 else args.folds
        for fold_id in range(1, n_splits + 1):
            rng = np.random.default_rng(args.seed + fold_id * 2011)
            train_parts = []
            test_parts = []
            for r in sorted(set(run.astype(str))):
                idx = np.where(run.astype(str) == r)[0]
                if idx.size < max(10, args.block_windows * 2):
                    continue
                idx = idx[np.argsort(sample_id[idx])]
                blocks = [idx[start : start + args.block_windows] for start in range(0, idx.size, args.block_windows)]
                blocks = [block for block in blocks if block.size >= max(5, args.block_windows // 3)]
                if len(blocks) < 2:
                    continue
                n_test_blocks = max(1, int(round(len(blocks) * args.within_run_test_frac)))
                test_block_ids = set(rng.choice(np.arange(len(blocks)), size=min(n_test_blocks, len(blocks) - 1), replace=False).tolist())
                test_local = []
                train_local = []
                for block_id, block in enumerate(blocks):
                    if block_id in test_block_ids:
                        test_local.append(block)
                    else:
                        train_local.append(block)
                test_idx_run = np.concatenate(test_local)
                train_candidates = np.concatenate(train_local)
                if args.block_gap_windows > 0:
                    keep = np.ones(train_candidates.shape[0], dtype=bool)
                    cand_samples = sample_id[train_candidates]
                    for block in test_local:
                        lo = int(sample_id[block].min())
                        hi = int(sample_id[block].max())
                        keep &= ~((cand_samples >= lo - args.block_gap_windows) & (cand_samples <= hi + args.block_gap_windows))
                    train_candidates = train_candidates[keep]
                if train_candidates.size < 10 or test_idx_run.size < 5:
                    continue
                train_parts.append(np.sort(train_candidates))
                test_parts.append(np.sort(test_idx_run))
            train_idx = np.concatenate(train_parts)
            test_idx = np.concatenate(test_parts)
            folds.append((fold_id, train_idx, test_idx, np.asarray([f"within_run_block_{fold_id}"])))
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")

    rows: list[MetricRow] = []
    for fold_id, train_idx, test_idx, test_subjects in folds:
        y_pc, y_var = fit_pca_standardized(y_flat, train_idx, args.target_pca_dim, args.seed + fold_id)
        x_red, x_dim, x_var = reduce_x(x, train_idx, args.x_pca_dim, args.seed + 100 + fold_id)
        print(
            f"fold={fold_id} train={train_idx.size} test={test_idx.size} "
            f"x_dim={x_dim} x_var={x_var:.3f} y_dim={y_pc.shape[1]} y_var={y_var:.3f}",
            flush=True,
        )

        # Mean control in target-PCA coordinates.
        mean_pred = np.zeros_like(y_pc[test_idx])
        add_row(
            rows,
            fold_id,
            "train_mean",
            "real",
            train_idx,
            test_idx,
            test_subjects,
            x_dim,
            y_pc.shape[1],
            0,
            mean_pred,
            y_pc[test_idx],
            mean_pred,
            y_pc[test_idx],
            run,
        )

        # Time baseline.
        tb = time_basis(time_frac, args.time_harmonics)
        tb_scaler = StandardScaler()
        tb_train = tb_scaler.fit_transform(tb[train_idx])
        tb_all = tb_scaler.transform(tb).astype(np.float32)
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 10))
        ridge.fit(tb_train, y_pc[train_idx])
        pred_time = ridge.predict(tb_all[test_idx]).astype(np.float32)
        add_row(
            rows,
            fold_id,
            "time_ridge",
            "real",
            train_idx,
            test_idx,
            test_subjects,
            tb_all.shape[1],
            y_pc.shape[1],
            0,
            pred_time,
            y_pc[test_idx],
            pred_time,
            y_pc[test_idx],
            run,
        )

        for target_mode in ("real", "shifted_null"):
            y_train_target = y_pc.copy()
            if target_mode == "shifted_null":
                y_train_target = shift_targets_by_run(y_pc, run, args.seed + fold_id * 1000)

            # Ridge/PLS predict target-PC coordinates from LaBraM features.
            ridge = RidgeCV(alphas=np.logspace(-2, 4, 10))
            ridge.fit(x_red[train_idx], y_train_target[train_idx])
            pred = ridge.predict(x_red[test_idx]).astype(np.float32)
            add_row(
                rows,
                fold_id,
                "ridge",
                target_mode,
                train_idx,
                test_idx,
                test_subjects,
                x_dim,
                y_pc.shape[1],
                0,
                pred,
                y_pc[test_idx],
                pred,
                y_pc[test_idx],
                run,
            )

            n_pls = min(args.pls_components, x_dim, y_pc.shape[1], max(1, train_idx.size - 1))
            pls = PLSRegression(n_components=n_pls, scale=True, max_iter=1000)
            pls.fit(x_red[train_idx], y_train_target[train_idx])
            pred = pls.predict(x_red[test_idx]).astype(np.float32)
            add_row(
                rows,
                fold_id,
                "pls",
                target_mode,
                train_idx,
                test_idx,
                test_subjects,
                x_dim,
                y_pc.shape[1],
                n_pls,
                pred,
                y_pc[test_idx],
                pred,
                y_pc[test_idx],
                run,
            )

            n_cca = min(args.cca_components, x_dim, y_pc.shape[1], max(1, train_idx.size - 1))
            cca = CCA(n_components=n_cca, scale=True, max_iter=args.cca_max_iter, tol=args.cca_tol)
            cca.fit(x_red[train_idx], y_train_target[train_idx])
            x_c_test, y_c_test = cca.transform(x_red[test_idx], y_pc[test_idx])
            # For CCA, prediction metrics are canonical-space agreement rather than latent-PC MSE.
            add_row(
                rows,
                fold_id,
                "cca_shared",
                target_mode,
                train_idx,
                test_idx,
                test_subjects,
                x_dim,
                y_pc.shape[1],
                n_cca,
                x_c_test.astype(np.float32),
                y_c_test.astype(np.float32),
                x_c_test.astype(np.float32),
                y_c_test.astype(np.float32),
                run,
            )

    write_outputs(args, rows)


def write_outputs(args: argparse.Namespace, rows: list[MetricRow]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MetricRow.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "models": {}}
    for row in rows:
        key = f"fold{row.fold}/{row.method}/{row.target_mode}"
        summary["models"][key] = asdict(row)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LaBraM NeuroSTORM Retrieval Diagnostic",
        "",
        "Frozen LaBraM features are aligned to official-style NeuroSTORM latent PCs using linear Ridge, PLS, and CCA. Retrieval is computed within each held-out run: the correct EEG/fMRI timepoint should rank above other timepoints from the same run.",
        "",
        f"- Feature cache: `{args.feature_path}`",
        f"- Target PCA dim: {args.target_pca_dim}",
        f"- X PCA dim: {args.x_pca_dim}",
        "",
        "| fold | method | target | latent r | row r | R2 | top1 | top5 | MRR | median rank | rank pct | diag-off |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.fold} | {row.method} | {row.target_mode} | {row.latent_corr_mean:.4f} | "
            f"{row.row_corr_mean:.4f} | {row.r2:.4f} | {row.retrieval_top1:.4f} | {row.retrieval_top5:.4f} | "
            f"{row.retrieval_mrr:.4f} | {row.retrieval_median_rank:.1f} | {row.retrieval_rank_percentile_mean:.4f} | "
            f"{row.diag_minus_offdiag:.4f} |"
        )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [f"{r.method}\n{r.target_mode}" for r in rows]
    top5 = [r.retrieval_top5 for r in rows]
    rank_pct = [r.retrieval_rank_percentile_mean for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, top5, width=0.36, label="top5")
    ax.bar(x + 0.18, rank_pct, width=0.36, label="rank percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Held-out within-run retrieval")
    fig.tight_layout()
    fig.savefig(args.results_dir / "retrieval_bar.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-mode", default="cls_mean_std", choices=["mean", "cls_mean", "mean_std", "cls_mean_std"])
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--patch-sec", type=float, default=1.0)
    p.add_argument("--resample-hz", type=float, default=200.0)
    p.add_argument("--window-zscore", action="store_true")
    p.add_argument("--min-channels", type=int, default=16)
    p.add_argument("--max-channels", type=int, default=64)
    p.add_argument("--min-windows", type=int, default=20)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--split-mode", choices=["subject", "within_run_random", "within_run_block"], default="subject")
    p.add_argument("--within-run-test-frac", type=float, default=0.33)
    p.add_argument("--block-windows", type=int, default=24)
    p.add_argument("--block-gap-windows", type=int, default=4)
    p.add_argument("--x-pca-dim", type=int, default=128)
    p.add_argument("--target-pca-dim", type=int, default=64)
    p.add_argument("--pls-components", type=int, default=16)
    p.add_argument("--cca-components", type=int, default=16)
    p.add_argument("--cca-max-iter", type=int, default=1000)
    p.add_argument("--cca-tol", type=float, default=1e-5)
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--seed", type=int, default=31)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract or args.force_extract or not args.feature_path.exists():
        extract_features(args)
    evaluate(args)


if __name__ == "__main__":
    main()
