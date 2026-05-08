#!/usr/bin/env python3
"""QC and denoise NatView MNI152 rest volumes for fMRI-teacher use.

NatView ships already preprocessed MNI152 3 mm BOLD volumes and Schaefer-100
atlas time series.  This script performs a conservative second-stage cleanup
for teacher extraction: finite-value checks, brain/nonzero masking, nuisance
regression with the shipped Motion24+CompCor design matrix, per-voxel
z-scoring, Schaefer-time alignment, and compact downsampling for fast
NeuroSTORM/teacher smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import signal
from scipy.ndimage import zoom

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from natview_pilot import discover_sessions, read_fmri_tsv  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing"
DEFAULT_OUT = REPO_ROOT / "data/neurostorm_natview_mni_qc"


@dataclass
class QcRow:
    subject: str
    session: str
    status: str
    n_time_mni: int
    n_time_schaefer: int
    kept_time: int
    input_shape: str
    output_shape: str
    brain_voxels: int
    nuisance_cols: int
    nuisance_rank: int
    fd_mean: float
    fd_max: float
    finite_frac: float
    mean_abs_after: float
    std_after: float
    denoise: str
    mni_path: str
    schaefer_path: str
    nuisance_path: str
    motion_path: str
    out_npz: str
    reason: str = ""


def find_mni_for_session(root: Path, subject: str, session: str) -> Path | None:
    base = root / subject / session / "func" / f"{subject}_{session}_task-rest_bold" / "func_preproc"
    path = base / "func_pp_filter_sm0.mni152.3mm.nii.gz"
    return path if path.exists() else None


def find_nuisance_for_session(root: Path, subject: str, session: str) -> tuple[Path | None, Path | None]:
    base = root / subject / session / "func" / f"{subject}_{session}_task-rest_bold" / "func_nuisance"
    nuisance = base / "Model_Motion24_CompCor.txt"
    motion = base / "func_mc.1D"
    return (nuisance if nuisance.exists() else None, motion if motion.exists() else None)


def load_numeric_table(path: Path, n_time: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < n_time:
        raise ValueError(f"{path.name} has {arr.shape[0]} rows, shorter than requested {n_time}")
    arr = arr[:n_time]
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return arr


def standardize_confounds(confounds: np.ndarray, eps: float) -> np.ndarray:
    mean = confounds.mean(axis=0, keepdims=True)
    std = confounds.std(axis=0, keepdims=True)
    keep = std.reshape(-1) > eps
    if not np.any(keep):
        return np.zeros((confounds.shape[0], 0), dtype=np.float32)
    return ((confounds[:, keep] - mean[:, keep]) / np.maximum(std[:, keep], eps)).astype(np.float32)


def design_matrix(confounds: np.ndarray | None, n_time: int, eps: float) -> tuple[np.ndarray, int, int]:
    t = np.linspace(-1.0, 1.0, n_time, dtype=np.float32)
    trends = np.stack([np.ones(n_time, dtype=np.float32), t, t * t], axis=1)
    if confounds is None:
        raw_cols = 0
        x = trends
    else:
        raw_cols = int(confounds.shape[1])
        x = np.concatenate([trends, standardize_confounds(confounds, eps)], axis=1)
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    q, _ = np.linalg.qr(x, mode="reduced")
    rank = int(np.linalg.matrix_rank(q, tol=1e-5))
    return q[:, :rank].astype(np.float32), raw_cols, rank


def framewise_displacement(motion: np.ndarray | None) -> tuple[float, float]:
    if motion is None or motion.shape[0] < 2 or motion.shape[1] < 6:
        return 0.0, 0.0
    diff = np.diff(motion[:, :6].astype(np.float32), axis=0)
    # AFNI-style motion files commonly store rotations in degrees.  Convert the
    # first three columns to arc length on a 50 mm sphere, then add translations.
    rot_mm = np.deg2rad(diff[:, :3]) * 50.0
    trans_mm = diff[:, 3:6]
    fd = np.sum(np.abs(np.concatenate([rot_mm, trans_mm], axis=1)), axis=1)
    return float(np.mean(fd)), float(np.max(fd))


def robust_clean_volume(
    path: Path,
    n_time: int,
    target_shape: tuple[int, int, int],
    eps: float,
    nuisance_path: Path | None,
    motion_path: Path | None,
    regress_nuisance: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"not 4D: {path}")
    finite = np.isfinite(data)
    finite_frac = float(finite.mean())
    data = np.where(finite, data, 0.0).astype(np.float32)
    if data.shape[3] < n_time:
        raise ValueError(f"MNI time {data.shape[3]} shorter than Schaefer time {n_time}")
    data = data[..., :n_time]
    spatial_mean_abs = np.mean(np.abs(data), axis=3)
    spatial_std = np.std(data, axis=3)
    mask = (spatial_mean_abs > eps) & (spatial_std > eps)
    if int(mask.sum()) < 1000:
        raise ValueError(f"too few brain voxels after nonzero/std mask: {int(mask.sum())}")

    confounds = load_numeric_table(nuisance_path, n_time) if regress_nuisance and nuisance_path is not None else None
    motion = load_numeric_table(motion_path, n_time) if motion_path is not None else None
    q, nuisance_cols, nuisance_rank = design_matrix(confounds, data.shape[3], eps)
    fd_mean, fd_max = framewise_displacement(motion)

    # Project out the nuisance/trend design in one QR step, then z-score each
    # voxel.  The projection is kept session-local to avoid leaking test data
    # statistics across subjects or sessions.
    flat = data.reshape(-1, data.shape[3])
    mask_flat = mask.reshape(-1)
    cleaned = np.zeros((flat.shape[0], flat.shape[1]), dtype=np.float32)
    selected = flat[mask_flat].astype(np.float32)
    selected = selected - (selected @ q) @ q.T
    selected = signal.detrend(selected, axis=1, type="linear").astype(np.float32)
    mean = selected.mean(axis=1, keepdims=True)
    std = selected.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    selected = np.clip((selected - mean) / std, -8.0, 8.0).astype(np.float32)
    cleaned[mask_flat] = selected
    cleaned = cleaned.reshape(data.shape)

    factors = [target_shape[i] / cleaned.shape[i] for i in range(3)] + [1.0]
    small = zoom(cleaned, factors, order=1).astype(np.float16)
    info = {
        "input_shape": "x".join(str(x) for x in data.shape),
        "output_shape": "x".join(str(x) for x in small.shape),
        "brain_voxels": int(mask.sum()),
        "nuisance_cols": nuisance_cols,
        "nuisance_rank": nuisance_rank,
        "fd_mean": fd_mean,
        "fd_max": fd_max,
        "finite_frac": finite_frac,
        "mean_abs_after": float(np.mean(np.abs(small))),
        "std_after": float(np.std(small.astype(np.float32))),
        "denoise": "mni_pp_filter+motion24_compcor_qr+trend+voxel_zscore"
        if confounds is not None
        else "mni_pp_filter+trend+voxel_zscore",
    }
    return small, info


def run(args: argparse.Namespace) -> None:
    records = discover_sessions(args.data_root)
    if args.max_sessions > 0:
        records = records[: args.max_sessions]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[QcRow] = []
    for rec in records:
        mni = find_mni_for_session(args.data_root, rec.subject, rec.session)
        nuisance, motion = find_nuisance_for_session(args.data_root, rec.subject, rec.session)
        out_path = args.out_dir / f"{rec.subject}_{rec.session}_task-rest_mni_clean.npz"
        if mni is None:
            rows.append(
                QcRow(
                    rec.subject,
                    rec.session,
                    "missing_mni",
                    0,
                    0,
                    0,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    "",
                    "",
                    str(rec.fmri_path),
                    "",
                    "",
                    "",
                    "missing func_pp_filter_sm0.mni152.3mm.nii.gz",
                )
            )
            continue
        try:
            sch = read_fmri_tsv(rec.fmri_path, expected_rois=100).astype(np.float32)
            vol, info = robust_clean_volume(
                mni,
                sch.shape[0],
                tuple(args.target_shape),
                args.eps,
                nuisance,
                motion,
                not args.skip_nuisance_regression,
            )
            kept = min(vol.shape[3], sch.shape[0])
            sch = sch[:kept]
            vol = vol[..., :kept]
            np.savez_compressed(
                out_path,
                volume=vol,
                schaefer100=sch.astype(np.float32),
                subject=rec.subject,
                session=rec.session,
                mni_path=str(mni),
                schaefer_path=str(rec.fmri_path),
                nuisance_path=str(nuisance) if nuisance is not None else "",
                motion_path=str(motion) if motion is not None else "",
                denoise=str(info["denoise"]),
            )
            rows.append(
                QcRow(
                    rec.subject,
                    rec.session,
                    "ok",
                    int(nib.load(str(mni)).shape[3]),
                    int(sch.shape[0]),
                    int(kept),
                    str(info["input_shape"]),
                    str(info["output_shape"]),
                    int(info["brain_voxels"]),
                    int(info["nuisance_cols"]),
                    int(info["nuisance_rank"]),
                    float(info["fd_mean"]),
                    float(info["fd_max"]),
                    float(info["finite_frac"]),
                    float(info["mean_abs_after"]),
                    float(info["std_after"]),
                    str(info["denoise"]),
                    str(mni),
                    str(rec.fmri_path),
                    str(nuisance) if nuisance is not None else "",
                    str(motion) if motion is not None else "",
                    str(out_path),
                )
            )
            print(
                f"ok {rec.subject} {rec.session} {info['input_shape']} -> {info['output_shape']} "
                f"rank={info['nuisance_rank']} fd_mean={info['fd_mean']:.4f}",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                QcRow(
                    rec.subject,
                    rec.session,
                    "error",
                    0,
                    0,
                    0,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    "",
                    str(mni),
                    str(rec.fmri_path),
                    str(nuisance) if nuisance is not None else "",
                    str(motion) if motion is not None else "",
                    "",
                    str(exc),
                )
            )
            print(f"error {rec.subject} {rec.session}: {exc}", flush=True)

    csv_path = args.out_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(QcRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summary = {
        "n_sessions": len(rows),
        "ok": sum(r.status == "ok" for r in rows),
        "missing_mni": sum(r.status == "missing_mni" for r in rows),
        "error": sum(r.status == "error" for r in rows),
        "target_shape": args.target_shape,
        "denoise_default": "mni_pp_filter+motion24_compcor_qr+trend+voxel_zscore",
        "skip_nuisance_regression": args.skip_nuisance_regression,
        "fd_mean_mean": float(np.mean([r.fd_mean for r in rows if r.status == "ok"])) if any(r.status == "ok" for r in rows) else 0.0,
        "fd_max_max": float(np.max([r.fd_max for r in rows if r.status == "ok"])) if any(r.status == "ok" for r in rows) else 0.0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-shape", type=int, nargs=3, default=(48, 48, 48))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--skip-nuisance-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
