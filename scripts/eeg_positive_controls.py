#!/usr/bin/env python3
"""EEG feature positive controls for paired EEG-fMRI experiments.

These diagnostics intentionally do not use fMRI targets.  They ask whether the
current EEG representation contains any usable structure, and whether that
structure is likely to be physiologically/stimulus relevant or mostly
subject/session/time drift.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]

DEFAULT_RESULTS = REPO_ROOT / "results/eeg_positive_controls_v1"
DEFAULT_FEATURES = [
    ("affective", REPO_ROOT / "data/labram_retrieval_schaefer100_affective/features.npz"),
    ("sleep", REPO_ROOT / "data/labram_retrieval_schaefer100_sleep/features.npz"),
    ("natview_neurostorm", REPO_ROOT / "data/labram_neurostorm_retrieval_natview_labram_eeg/features.npz"),
]


@dataclass
class ControlRow:
    dataset: str
    control: str
    split: str
    fold: int
    n_train: int
    n_test: int
    n_classes: int
    score_name: str
    score: float
    chance: float
    aux_name: str
    aux_score: float
    interpretation: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_feature_spec(items: list[str]) -> list[tuple[str, Path]]:
    if not items:
        return DEFAULT_FEATURES
    specs = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).parent.name
        specs.append((name, Path(path)))
    return specs


def maybe_reduce_x(x: np.ndarray, train_idx: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    meta = {"x_dim_in": float(x.shape[1]), "x_dim_out": float(x.shape[1]), "x_var": 1.0}
    if dim <= 0 or dim >= x.shape[1]:
        return scaler.transform(x).astype(np.float32), meta
    n_comp = min(dim, x.shape[1], max(1, train_idx.size - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(x_train)
    meta = {"x_dim_in": float(x.shape[1]), "x_dim_out": float(n_comp), "x_var": float(pca.explained_variance_ratio_.sum())}
    return pca.transform(scaler.transform(x)).astype(np.float32), meta


def make_time_bins(time_frac: np.ndarray, n_bins: int) -> np.ndarray:
    bins = np.floor(np.clip(time_frac, 0.0, 0.999999) * n_bins).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def make_within_run_block_folds(
    run: np.ndarray,
    sample_id: np.ndarray,
    folds: int,
    seed: int,
    test_frac: float,
    block_size: int,
    gap: int,
    min_train: int,
    min_test: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    run_str = run.astype(str)
    out = []
    for fold in range(1, folds + 1):
        rng = np.random.default_rng(seed + fold * 3571)
        train_parts = []
        test_parts = []
        for r in np.unique(run_str):
            idx = np.flatnonzero(run_str == r)
            idx = idx[np.argsort(sample_id[idx])]
            if idx.size < max(min_train + min_test, block_size * 2):
                continue
            blocks = [idx[start : start + block_size] for start in range(0, idx.size, block_size)]
            blocks = [b for b in blocks if b.size >= max(5, block_size // 4)]
            if len(blocks) < 2:
                continue
            n_test = max(1, int(round(len(blocks) * test_frac)))
            n_test = min(n_test, len(blocks) - 1)
            test_block_ids = set(rng.choice(np.arange(len(blocks)), size=n_test, replace=False).tolist())
            test_run = np.concatenate([b for i, b in enumerate(blocks) if i in test_block_ids])
            train_run = np.concatenate([b for i, b in enumerate(blocks) if i not in test_block_ids])
            train_sid = sample_id[train_run]
            keep = np.ones(train_run.size, dtype=bool)
            for block in [b for i, b in enumerate(blocks) if i in test_block_ids]:
                lo = int(sample_id[block].min())
                hi = int(sample_id[block].max())
                keep &= ~((train_sid >= lo - gap) & (train_sid <= hi + gap))
            train_run = train_run[keep]
            if train_run.size >= min_train and test_run.size >= min_test:
                train_parts.append(train_run)
                test_parts.append(test_run)
        if train_parts and test_parts:
            out.append((fold, np.concatenate(train_parts), np.concatenate(test_parts)))
    return out


def make_subject_folds(subject: np.ndarray, folds: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject.astype(str))
    rng.shuffle(subjects)
    chunks = np.array_split(subjects, min(folds, subjects.size))
    out = []
    for fold, test_subjects in enumerate(chunks, start=1):
        test_idx = np.flatnonzero(np.isin(subject.astype(str), test_subjects))
        train_idx = np.flatnonzero(~np.isin(subject.astype(str), test_subjects))
        if train_idx.size > 0 and test_idx.size > 0:
            out.append((fold, train_idx, test_idx))
    return out


def label_metrics(
    dataset: str,
    control: str,
    split: str,
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    x_pca_dim: int,
) -> ControlRow:
    x_red, _ = maybe_reduce_x(x, train_idx, x_pca_dim, seed + fold)
    clf = RidgeClassifier(class_weight="balanced")
    clf.fit(x_red[train_idx], y[train_idx])
    pred = clf.predict(x_red[test_idx])
    classes = np.unique(y)
    score = float(balanced_accuracy_score(y[test_idx], pred))
    aux = float(accuracy_score(y[test_idx], pred))
    chance = 1.0 / max(1, classes.size)
    if score >= chance * 2 and score >= chance + 0.05:
        interp = "above_chance"
    elif score <= chance + 0.03:
        interp = "near_chance"
    else:
        interp = "weak"
    return ControlRow(
        dataset=dataset,
        control=control,
        split=split,
        fold=fold,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        n_classes=int(classes.size),
        score_name="balanced_accuracy",
        score=score,
        chance=float(chance),
        aux_name="accuracy",
        aux_score=aux,
        interpretation=interp,
    )


def regression_metrics(
    dataset: str,
    control: str,
    split: str,
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    x_pca_dim: int,
) -> ControlRow:
    x_red, _ = maybe_reduce_x(x, train_idx, x_pca_dim, seed + fold)
    reg = RidgeCV(alphas=np.logspace(-2, 4, 10))
    reg.fit(x_red[train_idx], y[train_idx])
    pred = reg.predict(x_red[test_idx])
    corr = float(np.corrcoef(pred, y[test_idx])[0, 1]) if np.std(pred) > 1e-8 and np.std(y[test_idx]) > 1e-8 else math.nan
    r2 = float(r2_score(y[test_idx], pred))
    if np.isfinite(corr) and abs(corr) >= 0.15:
        interp = "above_chance"
    elif np.isfinite(corr) and abs(corr) <= 0.05:
        interp = "near_chance"
    else:
        interp = "weak"
    return ControlRow(
        dataset=dataset,
        control=control,
        split=split,
        fold=fold,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        n_classes=1,
        score_name="corr",
        score=corr,
        chance=0.0,
        aux_name="r2",
        aux_score=r2,
        interpretation=interp,
    )


def temporal_autocorr_rows(
    dataset: str,
    x: np.ndarray,
    run: np.ndarray,
    sample_id: np.ndarray,
    lags: list[int],
    seed: int,
    max_pairs_per_run: int,
) -> list[ControlRow]:
    rng = np.random.default_rng(seed)
    x_norm = x.astype(np.float64)
    x_norm = x_norm - np.nanmean(x_norm, axis=0, keepdims=True)
    x_norm = x_norm / np.maximum(np.linalg.norm(x_norm, axis=1, keepdims=True), 1e-9)
    rows = []
    for lag in lags:
        lag_sims = []
        rand_sims = []
        for r in np.unique(run.astype(str)):
            idx = np.flatnonzero(run.astype(str) == r)
            idx = idx[np.argsort(sample_id[idx])]
            if idx.size <= lag + 5:
                continue
            anchors = np.arange(0, idx.size - lag)
            if anchors.size > max_pairs_per_run:
                anchors = rng.choice(anchors, size=max_pairs_per_run, replace=False)
            a = idx[anchors]
            b = idx[anchors + lag]
            lag_sims.extend(np.sum(x_norm[a] * x_norm[b], axis=1).tolist())
            rand_a = rng.choice(idx, size=anchors.size, replace=True)
            rand_b = rng.choice(idx, size=anchors.size, replace=True)
            rand_sims.extend(np.sum(x_norm[rand_a] * x_norm[rand_b], axis=1).tolist())
        lag_mean = float(np.mean(lag_sims)) if lag_sims else math.nan
        rand_mean = float(np.mean(rand_sims)) if rand_sims else math.nan
        delta = lag_mean - rand_mean if np.isfinite(lag_mean) and np.isfinite(rand_mean) else math.nan
        if np.isfinite(delta) and delta >= 0.05:
            interp = "temporally_smooth"
        elif np.isfinite(delta) and delta <= 0.01:
            interp = "near_random"
        else:
            interp = "weak"
        rows.append(
            ControlRow(
                dataset=dataset,
                control=f"temporal_autocorr_lag{lag}",
                split="no_train",
                fold=0,
                n_train=0,
                n_test=int(len(lag_sims)),
                n_classes=1,
                score_name="lag_cosine",
                score=lag_mean,
                chance=rand_mean,
                aux_name="lag_minus_random",
                aux_score=delta,
                interpretation=interp,
            )
        )
    return rows


def aggregate(rows: list[ControlRow]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[ControlRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.control, row.split), []).append(row)
    out = {}
    for key, vals in grouped.items():
        scores = np.asarray([v.score for v in vals], dtype=np.float64)
        aux = np.asarray([v.aux_score for v in vals], dtype=np.float64)
        chances = np.asarray([v.chance for v in vals], dtype=np.float64)
        out[key] = {
            "score_mean": float(np.nanmean(scores)),
            "score_std": float(np.nanstd(scores)),
            "chance_mean": float(np.nanmean(chances)),
            "aux_mean": float(np.nanmean(aux)),
            "aux_std": float(np.nanstd(aux)),
            "n": float(len(vals)),
        }
    return out


def run_one_dataset(name: str, path: Path, args: argparse.Namespace) -> tuple[list[ControlRow], dict[str, object]]:
    z = np.load(path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    subject = z["subject"].astype(str)
    run = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int32)
    time_frac = z["time_frac"].astype(np.float32)
    if args.max_samples > 0 and x.shape[0] > args.max_samples:
        keep = np.linspace(0, x.shape[0] - 1, args.max_samples).round().astype(np.int64)
        x = x[keep]
        subject = subject[keep]
        run = run[keep]
        sample_id = sample_id[keep]
        time_frac = time_frac[keep]

    rows: list[ControlRow] = []
    time_bin = make_time_bins(time_frac, args.time_bins)
    within_folds = make_within_run_block_folds(
        run,
        sample_id,
        folds=args.folds,
        seed=args.seed,
        test_frac=args.test_frac,
        block_size=args.block_size,
        gap=args.gap,
        min_train=args.min_train_per_run,
        min_test=args.min_test_per_run,
    )
    if args.max_folds > 0:
        within_folds = within_folds[: args.max_folds]

    for fold, train_idx, test_idx in within_folds:
        rows.append(label_metrics(name, "subject_id_from_eeg", "within_run_block", fold, x, subject, train_idx, test_idx, args.seed, args.x_pca_dim))
        rows.append(label_metrics(name, "run_id_from_eeg", "within_run_block", fold, x, run, train_idx, test_idx, args.seed + 101, args.x_pca_dim))
        rows.append(label_metrics(name, "time_bin_from_eeg", "within_run_block", fold, x, time_bin, train_idx, test_idx, args.seed + 202, args.x_pca_dim))
        rows.append(regression_metrics(name, "time_frac_from_eeg", "within_run_block", fold, x, time_frac, train_idx, test_idx, args.seed + 303, args.x_pca_dim))

    subject_folds = make_subject_folds(subject, args.folds, args.seed + 404)
    if args.max_folds > 0:
        subject_folds = subject_folds[: args.max_folds]
    for fold, train_idx, test_idx in subject_folds:
        rows.append(label_metrics(name, "time_bin_from_eeg", "subject_heldout", fold, x, time_bin, train_idx, test_idx, args.seed + 505, args.x_pca_dim))
        rows.append(regression_metrics(name, "time_frac_from_eeg", "subject_heldout", fold, x, time_frac, train_idx, test_idx, args.seed + 606, args.x_pca_dim))

    rows.extend(temporal_autocorr_rows(name, x, run, sample_id, args.autocorr_lags, args.seed + 707, args.max_pairs_per_run))
    meta = {
        "path": str(path),
        "n_samples": int(x.shape[0]),
        "x_dim": int(x.shape[1]),
        "n_subjects": int(len(np.unique(subject))),
        "n_runs": int(len(np.unique(run))),
        "within_run_folds": int(len(within_folds)),
        "subject_folds": int(len(subject_folds)),
    }
    return rows, meta


def write_outputs(args: argparse.Namespace, rows: list[ControlRow], meta: dict[str, object]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ControlRow.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    agg = aggregate(rows)
    serial_agg = {"/".join(key): val for key, val in agg.items()}
    (args.results_dir / "summary.json").write_text(
        json.dumps({"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "meta": meta, "aggregate": serial_agg}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# EEG Positive Controls v1",
        "",
        "These diagnostics use frozen LaBraM EEG features only. They do not use fMRI targets.",
        "",
        "Interpretation:",
        "",
        "- High subject/run decoding means the representation contains stable fingerprints, which may be subject/session artifacts rather than task-relevant neural signal.",
        "- High within-run time decoding means the representation contains slow temporal structure or drift.",
        "- Cross-subject time decoding is the stricter proxy for shared stimulus-locked EEG signal; near-chance scores here support the EEG-bottleneck hypothesis.",
        "- Temporal autocorrelation above random means the features are smooth, but smoothness alone can be artifact/drift.",
        "",
    ]
    for dataset in sorted({row.dataset for row in rows}):
        lines.extend([f"## {dataset}", ""])
        ds_rows = [row for row in rows if row.dataset == dataset]
        ds_agg = aggregate(ds_rows)
        lines.extend(
            [
                "| control | split | score | chance/random | aux | n folds |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for (ds, control, split), vals in sorted(ds_agg.items()):
            del ds
            lines.append(
                f"| {control} | {split} | {vals['score_mean']:.4f} +/- {vals['score_std']:.4f} | "
                f"{vals['chance_mean']:.4f} | {vals['aux_mean']:.4f} +/- {vals['aux_std']:.4f} | {int(vals['n'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Bottom Line",
            "",
            "The key pass/fail criterion for useful EEG is not subject/run decoding; that can be driven by stable artifacts. The important criterion is whether EEG features predict shared time/stimulus structure across held-out subjects, and whether that signal is stronger than within-run drift/fingerprints.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_controls = ["subject_id_from_eeg", "run_id_from_eeg", "time_bin_from_eeg", "time_frac_from_eeg"]
    datasets = sorted({row.dataset for row in rows})
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, max(3, 2.8 * len(datasets))), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        ds_rows = [row for row in rows if row.dataset == dataset and row.control in plot_controls]
        ds_agg = aggregate(ds_rows)
        labels = []
        scores = []
        chances = []
        for (ds, control, split), vals in sorted(ds_agg.items()):
            del ds
            labels.append(f"{control}\n{split}")
            scores.append(vals["score_mean"])
            chances.append(vals["chance_mean"])
        x = np.arange(len(labels))
        ax.bar(x, scores, label="score")
        ax.scatter(x, chances, color="black", s=18, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(dataset)
        ax.set_ylim(min(-0.1, np.nanmin(scores) - 0.05), max(1.0, np.nanmax(scores) + 0.05))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results_dir / "eeg_positive_controls.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature", action="append", default=[], help="Feature spec as name=/path/to/features.npz")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--x-pca-dim", type=int, default=128)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--min-train-per-run", type=int, default=20)
    p.add_argument("--min-test-per-run", type=int, default=10)
    p.add_argument("--autocorr-lags", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--max-pairs-per-run", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    all_rows: list[ControlRow] = []
    meta = {}
    for name, path in parse_feature_spec(args.feature):
        print(f"== {name}: {path}", flush=True)
        rows, ds_meta = run_one_dataset(name, path, args)
        all_rows.extend(rows)
        meta[name] = ds_meta
    write_outputs(args, all_rows, meta)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(all_rows), "datasets": list(meta)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
