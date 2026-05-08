#!/usr/bin/env python3
"""Build a multi-dataset raw EEG cache with Schaefer-100 fMRI targets.

NatView already ships Schaefer-100 targets.  For BIDS datasets that only have
native-space BOLD locally, this script creates a pragmatic MNI proxy:

1. motion-correct each BOLD run in native space,
2. register the run mean BOLD to subject T1w,
3. register subject T1w to a Schaefer/MNI atlas grid,
4. warp the 4D BOLD to the atlas grid,
5. extract Schaefer-100 ROI time series and detrend/z-score them per run,
6. copy the existing raw EEG windows while replacing their old coarse-grid
   target with the unified Schaefer-100 target.

This is intentionally separate from the final fMRIPrep path.  It is meant to
let us concatenate more datasets for an honest pilot while recording that these
non-NatView targets are "mni_proxy_ants", not full fMRIPrep derivatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pooled_deep import read_tr, slug  # noqa: E402


DEFAULT_IN = REPO_ROOT / "data/pooled_raw_v1_fullsubj/run_cache"
DEFAULT_OUT = REPO_ROOT / "data/pooled_raw_schaefer100_mni_proxy/run_cache"
DEFAULT_WORK = REPO_ROOT / "data/mni_schaefer100_proxy_work"


@dataclass
class CacheRow:
    dataset: str
    subject: str
    session: str
    run: str
    task: str
    cache_path: str
    n_samples: int
    n_channels: int
    target_dim: int
    target_kind: str
    source_cache: str
    target_source: str


@dataclass
class TargetRow:
    dataset: str
    subject: str
    bold_path: str
    t1_path: str
    target_path: str
    n_time: int
    fd_mean: float
    fd_max: float
    roi_std_mean: float
    roi_valid: int
    transform_kind: str


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def hash_path(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def safe_npz_copy_with_target(
    src: Path,
    dst: Path,
    y: np.ndarray,
    target_path: Path,
    target_kind: str,
    y_mask: np.ndarray | None = None,
) -> CacheRow:
    z = np.load(src, allow_pickle=True)
    item = {k: z[k] for k in z.files}
    item["Y"] = y.astype(np.float32)
    if y_mask is not None:
        item["Y_mask"] = y_mask.astype(np.float32)
    item["target_path"] = np.asarray(str(target_path), dtype="U512")
    item["target_kind"] = np.asarray(target_kind, dtype="U64")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **item)
    return CacheRow(
        dataset=scalar_str(item["dataset"]),
        subject=scalar_str(item["subject"]),
        session=scalar_str(item["session"]),
        run=scalar_str(item["run"]),
        task=scalar_str(item["task"]),
        cache_path=str(dst),
        n_samples=int(y.shape[0]),
        n_channels=int(np.asarray(item["raw"]).shape[0]),
        target_dim=int(y.shape[1]),
        target_kind=target_kind,
        source_cache=str(src),
        target_source=str(target_path),
    )


def copy_natview(src: Path, dst: Path) -> CacheRow:
    z = np.load(src, allow_pickle=True)
    item = {k: z[k] for k in z.files}
    if "Y_mask" not in item:
        item["Y_mask"] = np.ones_like(np.asarray(item["Y"], dtype=np.float32), dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        np.savez_compressed(dst, **item)
    y = np.asarray(item["Y"])
    return CacheRow(
        dataset=scalar_str(z["dataset"]),
        subject=scalar_str(z["subject"]),
        session=scalar_str(z["session"]),
        run=scalar_str(z["run"]),
        task=scalar_str(z["task"]),
        cache_path=str(dst),
        n_samples=int(y.shape[0]),
        n_channels=int(np.asarray(z["raw"]).shape[0]),
        target_dim=int(y.shape[1]),
        target_kind="natview_shipped_schaefer100",
        source_cache=str(src),
        target_source=scalar_str(z["target_path"]),
    )


def find_t1(dataset_root: Path, subject: str, bold_path: Path) -> Path | None:
    candidates = []
    subj_dir = dataset_root / subject
    if subj_dir.exists():
        candidates.extend(sorted(p for p in subj_dir.glob("anat/*_T1w.nii.gz") if not p.name.startswith("._")))
        if "ses-" in str(bold_path):
            for parent in bold_path.parents:
                if parent.name.startswith("ses-"):
                    candidates.extend(
                        sorted(p for p in (subj_dir / parent.name / "anat").glob("*_T1w.nii.gz") if not p.name.startswith("._"))
                    )
                    break
    if not candidates:
        candidates.extend(sorted(p for p in dataset_root.glob(f"{subject}/**/*_T1w.nii.gz") if not p.name.startswith("._")))
    return candidates[0] if candidates else None


def atlas_reference(work_dir: Path) -> tuple[Path, Path]:
    from nilearn import datasets, image

    work_dir.mkdir(parents=True, exist_ok=True)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_path = Path(atlas.maps)
    ref_path = work_dir / "mni152_on_schaefer100_grid.nii.gz"
    if not ref_path.exists():
        atlas_img = nib.load(str(atlas_path))
        mni = datasets.load_mni152_template(resolution=2)
        ref = image.resample_to_img(mni, atlas_img, interpolation="continuous", force_resample=True, copy_header=True)
        data = np.asarray(ref.get_fdata(), dtype=np.float32)
        data = np.where(np.isfinite(data), data, 0.0)
        nib.Nifti1Image(data, atlas_img.affine, atlas_img.header).to_filename(str(ref_path))
    return atlas_path, ref_path


def detrend_zscore(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = signal.detrend(y, axis=0, type="linear").astype(np.float32)
    mean = y.mean(axis=0, keepdims=True)
    std = np.maximum(y.std(axis=0, keepdims=True), eps)
    return np.clip((y - mean) / std, -8.0, 8.0).astype(np.float32)


def align_fd(fd: np.ndarray, n_time: int) -> np.ndarray:
    fd = np.asarray(fd, dtype=np.float32).reshape(-1)
    out = np.zeros(n_time, dtype=np.float32)
    if fd.size == 0 or n_time <= 0:
        return out
    if fd.size == n_time:
        out[:] = fd[:n_time]
    elif fd.size == n_time - 1:
        out[1:] = fd
    else:
        n = min(n_time, fd.size)
        out[:n] = fd[:n]
    out[~np.isfinite(out)] = 0.0
    return out


def zscore_col(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - np.nanmean(x)) / max(float(np.nanstd(x)), eps)


def denoise_roi_timeseries(
    y: np.ndarray,
    tr: float,
    fd: np.ndarray,
    high_pass_sec: float,
    fd_spike_threshold: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, str]:
    """ROI-level cleanup for native-BOLD MNI proxy targets.

    Raw OpenNeuro BOLD runs do not usually ship fMRIPrep confounds locally.
    This keeps the pilot conservative by removing run drift, slow scanner
    components, FD-related motion variance, and sparse high-motion spikes
    before per-run ROI z-scoring.
    """

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"expected ROI matrix, got {y.shape}")
    n_time = y.shape[0]
    for j in range(y.shape[1]):
        col = y[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            fill = float(np.nanmean(col[~bad])) if (~bad).any() else 0.0
            col[bad] = fill
            y[:, j] = col

    t = np.linspace(-1.0, 1.0, n_time, dtype=np.float32)
    cols = [np.ones(n_time, dtype=np.float32), t, t * t]
    denoise_bits = ["intercept", "linear_quadratic_trend"]

    if high_pass_sec > 0 and tr > 0:
        duration = n_time * tr
        n_cos = int(np.floor(2.0 * duration / high_pass_sec))
        if n_cos > 0:
            frame = np.arange(n_time, dtype=np.float32) + 0.5
            for k in range(1, n_cos + 1):
                cols.append(np.cos(np.pi * frame * k / n_time).astype(np.float32))
            denoise_bits.append(f"dct_highpass_{high_pass_sec:g}s_{n_cos}cos")

    fd_aligned = align_fd(fd, n_time)
    if np.nanstd(fd_aligned) > eps:
        fd_z = zscore_col(fd_aligned)
        dfd_z = zscore_col(np.r_[0.0, np.diff(fd_aligned)])
        cols.extend([fd_z, dfd_z, fd_z * fd_z, dfd_z * dfd_z])
        denoise_bits.append("fd_deriv_power2")

    if fd_spike_threshold > 0:
        spikes = np.flatnonzero(fd_aligned > fd_spike_threshold)
        max_spikes = max(0, min(50, n_time // 5))
        if spikes.size > max_spikes > 0:
            spikes = spikes[np.argsort(fd_aligned[spikes])[-max_spikes:]]
        for idx in spikes:
            spike = np.zeros(n_time, dtype=np.float32)
            spike[int(idx)] = 1.0
            cols.append(spike)
        if spikes.size:
            denoise_bits.append(f"fd_spikes_{fd_spike_threshold:g}_{spikes.size}")

    x = np.column_stack(cols).astype(np.float32)
    keep = np.isfinite(x).all(axis=0) & (x.std(axis=0) > eps)
    keep[0] = True
    x = x[:, keep]
    rank = int(np.linalg.matrix_rank(x))
    if rank > 0:
        q, _ = np.linalg.qr(x)
        q = q[:, :rank].astype(np.float32)
        y = y - q @ (q.T @ y)
    y = signal.detrend(y, axis=0, type="linear").astype(np.float32)
    mean = y.mean(axis=0, keepdims=True)
    std = np.maximum(y.std(axis=0, keepdims=True), eps)
    return np.clip((y - mean) / std, -8.0, 8.0).astype(np.float32), "+".join(denoise_bits)


def extract_roi_timeseries(
    mni_bold,
    atlas_path: Path,
    tr: float,
    fd: np.ndarray,
    high_pass_sec: float,
    fd_spike_threshold: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    atlas_img = nib.load(str(atlas_path))
    labels = np.asarray(atlas_img.get_fdata(), dtype=np.int16)
    data = mni_bold.numpy()
    if data.ndim != 4:
        raise ValueError(f"warped BOLD is not 4D: {data.shape}")
    if data.shape[:3] != labels.shape:
        raise ValueError(f"BOLD/atlas shape mismatch: {data.shape[:3]} vs {labels.shape}")
    out = np.zeros((data.shape[3], 100), dtype=np.float32)
    roi_mask = np.zeros(100, dtype=np.float32)
    flat = data.reshape(-1, data.shape[3])
    label_flat = labels.reshape(-1)
    finite = np.isfinite(flat).all(axis=1)
    for roi in range(1, 101):
        mask = (label_flat == roi) & finite
        if int(mask.sum()) > 10:
            out[:, roi - 1] = flat[mask].mean(axis=0)
            roi_mask[roi - 1] = 1.0
    cleaned, denoise_kind = denoise_roi_timeseries(out, tr, fd, high_pass_sec, fd_spike_threshold)
    roi_mask = roi_mask * (np.std(cleaned, axis=0) > 1e-6).astype(np.float32)
    return cleaned, roi_mask, denoise_kind


def load_or_register_t1_to_mni(args: argparse.Namespace, dataset: str, subject: str, t1_path: Path, ref_path: Path):
    import ants

    reg_dir = args.work_dir / "registrations" / slug(dataset) / slug(subject)
    reg_json = reg_dir / "t1_to_mni.json"
    if reg_json.exists() and not args.rebuild_registration:
        meta = json.loads(reg_json.read_text(encoding="utf-8"))
        if all(Path(p).exists() for p in meta["fwdtransforms"]):
            return meta
    reg_dir.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(ref_path))
    moving = ants.image_read(str(t1_path))
    tx = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=args.t1_transform,
        outprefix=str(reg_dir / "t1_to_mni_"),
        singleprecision=True,
        verbose=args.verbose,
    )
    meta = {
        "fixed": str(ref_path),
        "moving": str(t1_path),
        "type_of_transform": args.t1_transform,
        "fwdtransforms": [str(p) for p in tx["fwdtransforms"]],
        "invtransforms": [str(p) for p in tx["invtransforms"]],
    }
    reg_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def motion_correct_bold(args: argparse.Namespace, bold_path: Path, out_dir: Path):
    import ants

    mc_path = out_dir / "bold_motion_corrected.nii.gz"
    fd_path = out_dir / "motion_fd.npy"
    if mc_path.exists() and fd_path.exists() and not args.rebuild_target:
        return ants.image_read(str(mc_path)), np.load(fd_path)
    bold = ants.image_read(str(bold_path))
    if not args.motion_correct:
        fd = np.zeros(max(0, bold.shape[-1] - 1), dtype=np.float32)
        ants.image_write(bold, str(mc_path))
        np.save(fd_path, fd)
        return bold, fd
    tx = ants.motion_correction(
        bold,
        type_of_transform=args.motion_transform,
        outprefix=str(out_dir / "motion_"),
        verbose=args.verbose,
    )
    mc = tx["motion_corrected"]
    fd = np.asarray(tx.get("FD", []), dtype=np.float32)
    ants.image_write(mc, str(mc_path))
    np.save(fd_path, fd)
    return mc, fd


def mean_bold_image(mc_bold, out_dir: Path):
    import ants

    mean_path = out_dir / "bold_mean.nii.gz"
    if mean_path.exists():
        return ants.image_read(str(mean_path))
    arr = np.asarray(mc_bold.numpy(), dtype=np.float32)
    mean = np.nanmean(arr, axis=3)
    img = ants.from_numpy(
        mean,
        origin=mc_bold.origin[:3],
        spacing=mc_bold.spacing[:3],
        direction=np.asarray(mc_bold.direction)[:3, :3],
    )
    ants.image_write(img, str(mean_path))
    return img


def extract_schaefer_for_bold(args: argparse.Namespace, dataset: str, subject: str, bold_path: Path, t1_path: Path) -> TargetRow:
    import ants

    atlas_path, ref_path = atlas_reference(args.work_dir)
    target_dir = args.work_dir / "targets" / slug(dataset) / slug(subject) / hash_path(bold_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "schaefer100_mni_proxy.npz"
    if target_path.exists() and not args.rebuild_target:
        z = np.load(target_path, allow_pickle=True)
        y = np.asarray(z["Y"], dtype=np.float32)
        roi_mask = np.asarray(z["roi_mask"], dtype=np.float32) if "roi_mask" in z.files else (np.std(y, axis=0) > 1e-6).astype(np.float32)
        fd = np.asarray(z["fd"], dtype=np.float32)
        return TargetRow(
            dataset,
            subject,
            str(bold_path),
            str(t1_path),
            str(target_path),
            int(y.shape[0]),
            float(np.nanmean(fd)) if fd.size else 0.0,
            float(np.nanmax(fd)) if fd.size else 0.0,
            float(np.nanmean(np.std(y, axis=0))),
            int(np.sum(roi_mask > 0.5)),
            "mni_proxy_ants_cached",
        )

    t1_meta = load_or_register_t1_to_mni(args, dataset, subject, t1_path, ref_path)
    mc_bold, fd = motion_correct_bold(args, bold_path, target_dir)
    bold_mean = mean_bold_image(mc_bold, target_dir)
    t1_img = ants.image_read(str(t1_path))
    b2t = ants.registration(
        fixed=t1_img,
        moving=bold_mean,
        type_of_transform=args.bold_transform,
        outprefix=str(target_dir / "bold_to_t1_"),
        singleprecision=True,
        verbose=args.verbose,
    )
    ref = ants.image_read(str(ref_path))
    chain = list(t1_meta["fwdtransforms"]) + [str(p) for p in b2t["fwdtransforms"]]
    warped = ants.apply_transforms(
        fixed=ref,
        moving=mc_bold,
        transformlist=chain,
        interpolator="linear",
        imagetype=3,
        singleprecision=True,
        verbose=args.verbose,
    )
    tr = read_tr(bold_path)
    y, roi_mask, denoise_kind = extract_roi_timeseries(
        warped,
        atlas_path,
        tr=tr,
        fd=fd,
        high_pass_sec=args.high_pass_sec,
        fd_spike_threshold=args.fd_spike_threshold,
    )
    np.savez_compressed(
        target_path,
        Y=y.astype(np.float32),
        roi_mask=roi_mask.astype(np.float32),
        fd=fd.astype(np.float32),
        bold_path=str(bold_path),
        t1_path=str(t1_path),
        atlas_path=str(atlas_path),
        ref_path=str(ref_path),
        t1_to_mni=json.dumps(t1_meta),
        bold_to_t1=json.dumps({"fwdtransforms": [str(p) for p in b2t["fwdtransforms"]]}),
        target_kind="mni_proxy_ants_schaefer100",
        denoise_kind=denoise_kind,
    )
    if args.cleanup_intermediate:
        for nifti_path in target_dir.glob("*.nii.gz"):
            nifti_path.unlink(missing_ok=True)
    return TargetRow(
        dataset,
        subject,
        str(bold_path),
        str(t1_path),
        str(target_path),
        int(y.shape[0]),
        float(np.nanmean(fd)) if fd.size else 0.0,
        float(np.nanmax(fd)) if fd.size else 0.0,
        float(np.nanmean(np.std(y, axis=0))),
        int(np.sum(roi_mask > 0.5)),
        f"motion={args.motion_transform};bold2t1={args.bold_transform};"
        f"t1tomni={args.t1_transform};denoise={denoise_kind}",
    )


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    include = set(args.include_dataset) if args.include_dataset else None
    selected = []
    seen_subjects: dict[str, set[str]] = {}
    seen_runs: dict[tuple[str, str], int] = {}
    for row in rows:
        dataset = row["dataset"]
        if include is not None and dataset not in include:
            continue
        if dataset == "natview":
            if args.include_natview:
                selected.append(row)
            continue
        subject = row["subject"]
        subjects = seen_subjects.setdefault(dataset, set())
        if subject not in subjects:
            if args.max_subjects_per_dataset > 0 and len(subjects) >= args.max_subjects_per_dataset:
                continue
            subjects.add(subject)
        key = (dataset, subject)
        count = seen_runs.get(key, 0)
        if args.max_runs_per_subject > 0 and count >= args.max_runs_per_subject:
            continue
        seen_runs[key] = count + 1
        selected.append(row)
    return selected


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.in_cache / "manifest.csv")
    selected = select_rows(manifest, args)
    args.out_cache.mkdir(parents=True, exist_ok=True)
    cache_rows: list[CacheRow] = []
    target_rows: list[TargetRow] = []
    errors: list[dict[str, str]] = []
    for i, row in enumerate(selected, start=1):
        dataset = row["dataset"]
        src = REPO_ROOT / row["cache_path"] if not Path(row["cache_path"]).is_absolute() else Path(row["cache_path"])
        dst = args.out_cache / slug(dataset) / src.name
        try:
            if dataset == "natview":
                if not dst.exists() or args.rebuild_cache:
                    cache_rows.append(copy_natview(src, dst))
                else:
                    cache_rows.append(copy_natview(dst, dst))
                print(f"[{i:04d}/{len(selected):04d}] copy natview {row['run']}", flush=True)
                continue
            z = np.load(src, allow_pickle=True)
            bold_path = Path(scalar_str(z["target_path"]))
            if not bold_path.is_absolute():
                bold_path = REPO_ROOT / bold_path
            dataset_root = REPO_ROOT / "downloads/paired_datasets" / dataset
            t1_path = find_t1(dataset_root, row["subject"], bold_path)
            if t1_path is None:
                raise FileNotFoundError(f"missing T1w for {dataset} {row['subject']}")
            target_row = extract_schaefer_for_bold(args, dataset, row["subject"], bold_path, t1_path)
            if target_row.roi_valid < args.min_roi_valid:
                raise ValueError(f"too few valid Schaefer ROIs after MNI proxy: {target_row.roi_valid}")
            target_rows.append(target_row)
            target_npz = np.load(target_row.target_path, allow_pickle=True)
            target = target_npz["Y"].astype(np.float32)
            roi_mask = (
                target_npz["roi_mask"].astype(np.float32)
                if "roi_mask" in target_npz.files
                else (np.std(target, axis=0) > 1e-6).astype(np.float32)
            )
            tr = read_tr(bold_path)
            idx = np.rint(np.asarray(z["sample_time"], dtype=np.float32) / tr - 0.5).astype(int)
            good = (idx >= 0) & (idx < target.shape[0])
            if good.sum() < args.min_samples:
                raise ValueError(f"too few samples after Schaefer alignment: {good.sum()}")
            y = target[idx[good]]
            y_mask = np.broadcast_to(roi_mask.reshape(1, -1), y.shape).astype(np.float32)
            if good.all():
                cache_rows.append(
                    safe_npz_copy_with_target(
                        src,
                        dst,
                        y,
                        Path(target_row.target_path),
                        "mni_proxy_ants_schaefer100",
                        y_mask,
                    )
                )
            else:
                item = {k: z[k] for k in z.files}
                item["sample_start"] = np.asarray(item["sample_start"])[good]
                item["sample_time"] = np.asarray(item["sample_time"])[good]
                item["Y"] = y.astype(np.float32)
                item["Y_mask"] = y_mask.astype(np.float32)
                item["target_path"] = np.asarray(target_row.target_path, dtype="U512")
                item["target_kind"] = np.asarray("mni_proxy_ants_schaefer100", dtype="U64")
                np.savez_compressed(dst, **item)
                cache_rows.append(
                    CacheRow(
                        dataset=dataset,
                        subject=row["subject"],
                        session=row["session"],
                        run=row["run"],
                        task=row["task"],
                        cache_path=str(dst),
                        n_samples=int(y.shape[0]),
                        n_channels=int(np.asarray(item["raw"]).shape[0]),
                        target_dim=100,
                        target_kind="mni_proxy_ants_schaefer100",
                        source_cache=str(src),
                        target_source=target_row.target_path,
                    )
                )
            print(f"[{i:04d}/{len(selected):04d}] built {dataset} {row['subject']} {row['run']}", flush=True)
        except Exception as exc:
            errors.append({**row, "error": str(exc)})
            print(f"[{i:04d}/{len(selected):04d}] error {dataset} {row.get('run', '')}: {exc}", flush=True)

    if cache_rows:
        with (args.out_cache / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(CacheRow.__annotations__.keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in cache_rows)
    if target_rows:
        with (args.out_cache / "target_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(TargetRow.__annotations__.keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in target_rows)
    if errors:
        with (args.out_cache / "errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in errors for k in row}))
            writer.writeheader()
            writer.writerows(errors)
    summary = {
        "selected": len(selected),
        "built": len(cache_rows),
        "targets": len(target_rows),
        "errors": len(errors),
        "out_cache": str(args.out_cache),
    }
    (args.out_cache / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-cache", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-cache", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--include-natview", action="store_true")
    parser.add_argument("--max-subjects-per-dataset", type=int, default=1)
    parser.add_argument("--max-runs-per-subject", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-roi-valid", type=int, default=95)
    parser.add_argument("--t1-transform", default="antsRegistrationSyNQuick[a]")
    parser.add_argument("--bold-transform", default="BOLDAffine")
    parser.add_argument("--motion-transform", default="QuickRigid")
    parser.add_argument("--motion-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--high-pass-sec", type=float, default=128.0)
    parser.add_argument("--fd-spike-threshold", type=float, default=0.5)
    parser.add_argument("--cleanup-intermediate", action="store_true")
    parser.add_argument("--rebuild-target", action="store_true")
    parser.add_argument("--rebuild-registration", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
