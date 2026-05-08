#!/usr/bin/env python3
"""Affective music auxiliary-channel controls.

Use EEG-only bandpower features to predict the `MUSIC` auxiliary channel
envelope/RMS from the Affective dataset.  This is a stricter stimulus-locking
positive control than generic time-bin decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eeg_positive_controls import make_subject_folds, make_within_run_block_folds  # noqa: E402


DEFAULT_RAW_DIR = (
    REPO_ROOT
    / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache/Affective_music_listening_OpenNeuro_ds002725"
)
DEFAULT_FEATURE_DIR = REPO_ROOT / "data/eeg_raw_bandpower_controls_v1"
DEFAULT_RESULTS = REPO_ROOT / "results/eeg_affective_music_aux_controls_v1"


@dataclass
class AuxRow:
    target: str
    model: str
    split: str
    fold: int
    n_train: int
    n_test: int
    corr: float
    r2: float
    y_std_train: float
    y_std_test: float


def scalar(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return str(arr.item()) if arr.shape == () else str(arr)


def window_aux_targets(raw: np.ndarray, channels: list[str], starts: np.ndarray, sfreq: float, window_sec: float) -> dict[str, np.ndarray]:
    win = int(round(window_sec * sfreq))
    good = (starts >= 0) & (starts + win <= raw.shape[1])
    starts = starts[good]
    out: dict[str, list[float]] = {
        "music_rms": [],
        "music_absmean": [],
        "music_std": [],
        "trialtype_rms": [],
        "ft_valence_mean": [],
        "ft_arousal_mean": [],
    }
    upper = [ch.upper() for ch in channels]

    def get_channel(name: str) -> np.ndarray | None:
        if name not in upper:
            return None
        return raw[upper.index(name)].astype(np.float32)

    music = get_channel("MUSIC")
    trialtype = get_channel("TRIALTYPE")
    valence = get_channel("FT_VALANCE")
    arousal = get_channel("FT_AROUSAL")
    for s in starts:
        if music is not None:
            x = np.nan_to_num(music[s : s + win], nan=0.0)
            out["music_rms"].append(float(np.sqrt(np.mean(x * x))))
            out["music_absmean"].append(float(np.mean(np.abs(x))))
            out["music_std"].append(float(np.std(x)))
        else:
            out["music_rms"].append(math.nan)
            out["music_absmean"].append(math.nan)
            out["music_std"].append(math.nan)
        if trialtype is not None:
            x = np.nan_to_num(trialtype[s : s + win], nan=0.0)
            out["trialtype_rms"].append(float(np.sqrt(np.mean(x * x))))
        else:
            out["trialtype_rms"].append(math.nan)
        if valence is not None:
            out["ft_valence_mean"].append(float(np.nanmean(valence[s : s + win])))
        else:
            out["ft_valence_mean"].append(math.nan)
        if arousal is not None:
            out["ft_arousal_mean"].append(float(np.nanmean(arousal[s : s + win])))
        else:
            out["ft_arousal_mean"].append(math.nan)
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def build_aux_cache(args: argparse.Namespace) -> Path:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.glob("*.npz"))
    targets: dict[str, list[np.ndarray]] = {}
    subjects, runs, sample_ids, time_frac = [], [], [], []
    manifest = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        raw = z["raw"].astype(np.float32)
        channels = [str(ch) for ch in z["channels"]]
        starts = z["sample_start"].astype(np.int64)
        sfreq = float(np.asarray(z["sfreq"]).item())
        aux = window_aux_targets(raw, channels, starts, sfreq, args.window_sec)
        n = min(len(next(iter(aux.values()))), starts.size)
        for key, arr in aux.items():
            targets.setdefault(key, []).append(arr[:n])
        subject = scalar(z, "subject", path.stem.split("_")[0])
        run = scalar(z, "run", path.stem)
        subjects.extend([subject] * n)
        runs.extend([run] * n)
        sample_ids.extend(range(n))
        if "time_frac" in z.files:
            tf = z["time_frac"].astype(np.float32)[:n]
        else:
            tf = np.linspace(0, 1, n, dtype=np.float32)
        time_frac.extend(tf.tolist())
        manifest.append({"path": str(path), "subject": subject, "run": run, "n_windows": n})
    out = args.results_dir / "affective_aux_targets.npz"
    np.savez_compressed(
        out,
        subject=np.asarray(subjects, dtype="U32"),
        run=np.asarray(runs, dtype="U160"),
        sample_id=np.asarray(sample_ids, dtype=np.int32),
        time_frac=np.asarray(time_frac, dtype=np.float32),
        **{key: np.concatenate(vals).astype(np.float32) for key, vals in targets.items()},
    )
    with (args.results_dir / "aux_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "subject", "run", "n_windows"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    return out


def time_basis(time_frac: np.ndarray, harmonics: int) -> np.ndarray:
    t = time_frac.reshape(-1, 1).astype(np.float32)
    parts = [np.ones_like(t), t, t * t]
    for k in range(1, harmonics + 1):
        parts.append(np.sin(2 * np.pi * k * t))
        parts.append(np.cos(2 * np.pi * k * t))
    return np.concatenate(parts, axis=1).astype(np.float32)


def reduce_x(x: np.ndarray, train_idx: np.ndarray, dim: int, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    train = scaler.fit_transform(x[train_idx])
    if dim <= 0 or dim >= x.shape[1]:
        return scaler.transform(x).astype(np.float32)
    n_comp = min(dim, x.shape[1], max(1, train_idx.size - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(train)
    return pca.transform(scaler.transform(x)).astype(np.float32)


def eval_regression(
    target_name: str,
    model_name: str,
    split: str,
    fold: int,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    x_pca_dim: int,
) -> AuxRow:
    valid_train = np.isfinite(y[train_idx])
    valid_test = np.isfinite(y[test_idx])
    train_idx = train_idx[valid_train]
    test_idx = test_idx[valid_test]
    y_train = y[train_idx]
    y_test = y[test_idx]
    if train_idx.size < 20 or test_idx.size < 10 or np.nanstd(y_train) < 1e-8 or np.nanstd(y_test) < 1e-8:
        return AuxRow(target_name, model_name, split, fold, int(train_idx.size), int(test_idx.size), math.nan, math.nan, float(np.nanstd(y_train)), float(np.nanstd(y_test)))
    x_red = reduce_x(x, train_idx, x_pca_dim, seed + fold)
    reg = RidgeCV(alphas=np.logspace(-2, 4, 10))
    reg.fit(x_red[train_idx], y_train)
    pred = reg.predict(x_red[test_idx])
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-8 else math.nan
    r2 = float(r2_score(y_test, pred))
    return AuxRow(
        target=target_name,
        model=model_name,
        split=split,
        fold=fold,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        corr=corr,
        r2=r2,
        y_std_train=float(np.nanstd(y_train)),
        y_std_test=float(np.nanstd(y_test)),
    )


def run(args: argparse.Namespace) -> list[AuxRow]:
    aux_path = build_aux_cache(args) if args.force_aux or not (args.results_dir / "affective_aux_targets.npz").exists() else args.results_dir / "affective_aux_targets.npz"
    aux = np.load(aux_path, allow_pickle=True)
    subject = aux["subject"].astype(str)
    run_id = aux["run"].astype(str)
    sample_id = aux["sample_id"].astype(np.int32)
    time_frac = aux["time_frac"].astype(np.float32)
    feature_paths = {
        "summary_bandpower": args.feature_dir / "affective_summary_bandpower.npz",
        "spatial_bandpower": args.feature_dir / "affective_spatial_bandpower.npz",
    }
    rows: list[AuxRow] = []
    within_folds = make_within_run_block_folds(
        run_id,
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
    subject_folds = make_subject_folds(subject, args.folds, args.seed + 404)
    if args.max_folds > 0:
        subject_folds = subject_folds[: args.max_folds]

    target_names = [name for name in aux.files if name not in {"subject", "run", "sample_id", "time_frac"}]
    for target_name in target_names:
        y = aux[target_name].astype(np.float32)
        for fold, train_idx, test_idx in within_folds:
            tb = time_basis(time_frac, args.time_harmonics)
            rows.append(eval_regression(target_name, "time_ridge", "within_run_block", fold, tb, y, train_idx, test_idx, args.seed, 0))
        for fold, train_idx, test_idx in subject_folds:
            tb = time_basis(time_frac, args.time_harmonics)
            rows.append(eval_regression(target_name, "time_ridge", "subject_heldout", fold, tb, y, train_idx, test_idx, args.seed, 0))
        for model_name, feature_path in feature_paths.items():
            f = np.load(feature_path, allow_pickle=True)
            x = f["X"].astype(np.float32)
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Feature/aux length mismatch for {model_name}: {x.shape[0]} vs {y.shape[0]}")
            for fold, train_idx, test_idx in within_folds:
                rows.append(eval_regression(target_name, model_name, "within_run_block", fold, x, y, train_idx, test_idx, args.seed, args.x_pca_dim))
            for fold, train_idx, test_idx in subject_folds:
                rows.append(eval_regression(target_name, model_name, "subject_heldout", fold, x, y, train_idx, test_idx, args.seed + 101, args.x_pca_dim))
    return rows


def write_outputs(args: argparse.Namespace, rows: list[AuxRow]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuxRow.__annotations__.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    grouped = {}
    for row in rows:
        key = (row.target, row.model, row.split)
        grouped.setdefault(key, []).append(row)
    summary = {}
    for key, vals in grouped.items():
        corr = np.asarray([v.corr for v in vals], dtype=float)
        r2 = np.asarray([v.r2 for v in vals], dtype=float)
        summary["/".join(key)] = {
            "corr_mean": float(np.nanmean(corr)),
            "corr_std": float(np.nanstd(corr)),
            "r2_mean": float(np.nanmean(r2)),
            "r2_std": float(np.nanstd(r2)),
            "n": len(vals),
        }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Affective Music Auxiliary Controls",
        "",
        "EEG-only bandpower features are used to predict Affective auxiliary channels, especially the `MUSIC` channel envelope/RMS.",
        "",
        "| target | model | split | corr | R2 | n |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for key, vals in sorted(summary.items()):
        target, model, split = key.split("/")
        lines.append(
            f"| {target} | {model} | {split} | {vals['corr_mean']:.4f} +/- {vals['corr_std']:.4f} | "
            f"{vals['r2_mean']:.4f} +/- {vals['r2_std']:.4f} | {vals['n']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `time_ridge` is an upper-bound nuisance baseline: if the auxiliary target is mostly deterministic over the shared run timeline, time can predict it.",
            "- The EEG models matter only if bandpower predicts auxiliary targets across held-out subjects with positive R2 or robust correlation.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_rows = [row for row in rows if row.target.startswith("music_") and row.split == "subject_heldout"]
    labels = sorted(set((row.target, row.model) for row in plot_rows))
    means, errs = [], []
    for target, model in labels:
        vals = [row.corr for row in plot_rows if row.target == target and row.model == model]
        means.append(float(np.nanmean(vals)))
        errs.append(float(np.nanstd(vals)))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{m}" for t, m in labels], rotation=35, ha="right", fontsize=8)
    ax.set_title("Subject-heldout EEG -> MUSIC auxiliary target")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(args.results_dir / "music_aux_subject_heldout.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--force-aux", action="store_true")
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--x-pca-dim", type=int, default=64)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--min-train-per-run", type=int, default=20)
    p.add_argument("--min-test-per-run", type=int, default=10)
    p.add_argument("--time-harmonics", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    write_outputs(args, rows)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
