#!/usr/bin/env python3
"""Audit local paired datasets for unified Schaefer-100 readiness.

The spatial-distillation branch should only concatenate runs whose fMRI target
has the same semantics.  This audit separates ready Schaefer/MNI data from raw
native-space BIDS data that needs fMRIPrep or another explicit MNI registration
step before it can safely enter a shared Schaefer-100 target.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "downloads/paired_datasets"
DEFAULT_OUT = REPO_ROOT / "data/schaefer_readiness"

EEG_SUFFIXES = ("_eeg.vhdr", "_eeg.set", "_eeg.edf", "_eeg.bdf", "_eeg.fif")


@dataclass(frozen=True)
class BidsEntities:
    sub: str = ""
    ses: str = ""
    task: str = ""
    run: str = ""
    echo: str = ""


@dataclass
class DatasetAudit:
    dataset: str
    n_subject_dirs: int
    n_files: int
    n_eeg_files: int
    n_bold_files: int
    n_bids_paired_runs: int
    n_mni_bold_files: int
    n_mni_paired_runs: int
    n_schaefer_tsv: int
    n_natview_schaefer_runs: int
    n_t1_like: int
    n_archives: int
    status: str
    action: str
    example_eeg: str
    example_bold: str
    example_mni: str
    example_schaefer: str


def parse_bids_entities(path: Path, suffix: str) -> BidsEntities:
    name = path.name
    for ending in (
        f"_{suffix}.nii.gz",
        f"_{suffix}.vhdr",
        f"_{suffix}.set",
        f"_{suffix}.edf",
        f"_{suffix}.bdf",
        f"_{suffix}.fif",
    ):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    out: dict[str, str] = {}
    for part in name.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            out[key] = value
    return BidsEntities(
        sub=out.get("sub", ""),
        ses=out.get("ses", ""),
        task=canonical_task(out.get("task", "")),
        run=out.get("run", ""),
        echo=out.get("echo", ""),
    )


def canonical_task(task: str) -> str:
    upper = task.upper()
    for suffix in ("ON", "OFF"):
        if upper.endswith(suffix) and len(task) > len(suffix):
            return task[: -len(suffix)]
    return task


def bids_key(path: Path, suffix: str) -> tuple[str, str, str, str]:
    ent = parse_bids_entities(path, suffix)
    return (ent.sub, ent.ses, ent.task, ent.run)


def is_eeg(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in EEG_SUFFIXES)


def is_bold(path: Path) -> bool:
    return path.name.endswith("_bold.nii.gz")


def is_mni_like(path: Path) -> bool:
    text = str(path).lower()
    return "mni" in text or "space-mni" in text or "tpl-mni" in text


def is_archive(path: Path) -> bool:
    return path.name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar"))


def is_t1_like(path: Path) -> bool:
    text = path.name.lower()
    return text.endswith(".nii.gz") and ("_t1w" in text or "mprage" in text or "t1" in text)


def first(paths: list[Path]) -> str:
    return str(paths[0]) if paths else ""


def paired_bids_counts(eeg_files: list[Path], bold_files: list[Path], mni_bold_files: list[Path]) -> tuple[int, int]:
    bold_index: dict[tuple[str, str, str, str], Path] = {}
    mni_index: dict[tuple[str, str, str, str], Path] = {}
    for bold in bold_files:
        ent = parse_bids_entities(bold, "bold")
        if ent.echo and ent.echo not in {"1", "01"}:
            continue
        bold_index.setdefault((ent.sub, ent.ses, ent.task, ent.run), bold)
        if bold in mni_bold_files:
            mni_index.setdefault((ent.sub, ent.ses, ent.task, ent.run), bold)
    paired = 0
    mni_paired = 0
    for eeg in eeg_files:
        key = bids_key(eeg, "eeg")
        if key in bold_index:
            paired += 1
        if key in mni_index:
            mni_paired += 1
    return paired, mni_paired


def audit_dataset(dataset_dir: Path) -> DatasetAudit:
    files = [p for p in dataset_dir.rglob("*") if p.is_file() and not p.name.startswith("._")]
    eeg_files = sorted([p for p in files if is_eeg(p)])
    bold_files = sorted([p for p in files if is_bold(p)])
    mni_bold_files = sorted([p for p in files if p.name.endswith(".nii.gz") and is_mni_like(p)])
    schaefer_tsv = sorted([p for p in files if p.suffix == ".tsv" and "schaefer" in str(p).lower()])
    t1_like = sorted([p for p in files if is_t1_like(p)])
    archives = sorted([p for p in files if is_archive(p)])
    n_paired, n_mni_paired = paired_bids_counts(eeg_files, bold_files, mni_bold_files)
    natview_schaefer_runs = len(schaefer_tsv) if eeg_files and "natview" in dataset_dir.name.lower() else 0

    if natview_schaefer_runs > 0:
        status = "ready_schaefer100"
        action = "use shipped Schaefer100 TSV as anchor; MNI volumes are available for teacher/QC"
    elif n_mni_paired > 0:
        status = "ready_mni_extract"
        action = "extract Schaefer100 from paired MNI-space BOLD, then add to unified cache"
    elif n_paired > 0:
        status = "needs_mni_preprocessing"
        action = "run fMRIPrep/native-to-MNI registration before Schaefer100 extraction"
    elif bold_files and not eeg_files:
        status = "fmri_only_visible"
        action = "cannot train EEG-to-fMRI unless matching EEG is located"
    elif eeg_files and not bold_files:
        status = "eeg_only_visible"
        action = "cannot train EEG-to-fMRI unless matching BOLD is located"
    elif archives:
        status = "archives_need_unpack_or_loader"
        action = "unpack archives or write dataset-specific loader, then re-audit"
    elif mni_bold_files:
        status = "mni_fmri_visible_no_bids_eeg_pair"
        action = "write dataset-specific pairing loader if simultaneous EEG exists"
    else:
        status = "not_ready"
        action = "inspect manually"

    return DatasetAudit(
        dataset=dataset_dir.name,
        n_subject_dirs=len([p for p in dataset_dir.glob("sub-*") if p.is_dir()])
        + len([p for p in dataset_dir.glob("sub_*") if p.is_dir()]),
        n_files=len(files),
        n_eeg_files=len(eeg_files),
        n_bold_files=len(bold_files),
        n_bids_paired_runs=n_paired,
        n_mni_bold_files=len(mni_bold_files),
        n_mni_paired_runs=n_mni_paired,
        n_schaefer_tsv=len(schaefer_tsv),
        n_natview_schaefer_runs=natview_schaefer_runs,
        n_t1_like=len(t1_like),
        n_archives=len(archives),
        status=status,
        action=action,
        example_eeg=first(eeg_files),
        example_bold=first(bold_files),
        example_mni=first(mni_bold_files),
        example_schaefer=first(schaefer_tsv),
    )


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_dataset(p) for p in sorted(args.data_root.iterdir()) if p.is_dir()]
    csv_path = args.out_dir / "dataset_schaefer_readiness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DatasetAudit.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary: dict[str, int] = {}
    for row in rows:
        summary[row.status] = summary.get(row.status, 0) + 1
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
