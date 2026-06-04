#!/usr/bin/env python3
"""Build strict THINGS-EEG / THINGS-fMRI image overlap tables.

The external validation should use exact stimulus-image matches when possible.
Concept-only matches are reported separately because they are useful for
inventory, but they should not be used as heldout same-image validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_ROOT = WORKSPACE / "data" / "things_fmri" / "ds004192_metadata"
DEFAULT_MANIFEST_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "manifest"
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
)


def read_flexible_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=",")
    if table.shape[1] == 1:
        table = pd.read_csv(path, sep="\t")
    return table


def normalize_concept(value: object) -> str:
    text = str(value).strip().lower()
    parts = text.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        text = parts[1]
    return text.replace("-", "_").replace(" ", "_")


def load_local_manifest(path: Path, split: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.copy()
    table["split"] = split
    table["image_file"] = table["image_path"].map(lambda value: Path(str(value)).name)
    table["concept_norm"] = table["things_concept"].map(normalize_concept)
    required = [
        "split",
        "image_index",
        "concept",
        "things_concept",
        "concept_norm",
        "image_file",
        "image_path",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table[required].drop_duplicates("image_file")


def load_fmri_stimulus_metadata(root: Path) -> pd.DataFrame:
    paths = sorted(
        root.glob(
            "derivatives/ICA-betas/sub-*/voxel-metadata/*_task-things_stimulus-metadata.tsv"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"No THINGS-fMRI stimulus metadata found under {root}. "
            "Clone the OpenNeuro ds004192 metadata mirror first."
        )
    rows = []
    for path in paths:
        subject = next(part for part in path.parts if part.startswith("sub-"))
        table = read_flexible_table(path)
        table = table.copy()
        table["subject"] = subject
        table["image_file"] = table["stimulus"].map(lambda value: Path(str(value)).name)
        table["concept_norm"] = table["concept"].map(normalize_concept)
        rows.append(table)
    fmri = pd.concat(rows, ignore_index=True)
    required = ["subject", "image_file", "stimulus", "concept", "trial_id"]
    missing = [column for column in required if column not in fmri.columns]
    if missing:
        raise ValueError(f"THINGS-fMRI stimulus metadata missing columns: {missing}")
    return fmri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI_ROOT)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "things_eeg_train_image_manifest.csv",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "things_eeg_test_image_manifest.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    local = pd.concat(
        [
            load_local_manifest(args.train_manifest, "train"),
            load_local_manifest(args.test_manifest, "test"),
        ],
        ignore_index=True,
    )
    fmri = load_fmri_stimulus_metadata(args.fmri_root)

    fmri_by_file = (
        fmri.groupby("image_file")
        .agg(
            fmri_concept=("concept", lambda values: "|".join(sorted(set(map(str, values))))),
            n_fmri_rows=("image_file", "size"),
            fmri_subjects=("subject", lambda values: "|".join(sorted(set(map(str, values))))),
            fmri_trial_ids=("trial_id", lambda values: "|".join(map(str, sorted(set(values))))),
        )
        .reset_index()
    )
    strict = local.merge(fmri_by_file, on="image_file", how="inner")
    strict["match_kind"] = "file"
    strict = strict.sort_values(["split", "image_index", "image_file"]).reset_index(drop=True)

    strict_files = set(strict["image_file"])
    fmri_concepts = set(map(str, fmri["concept_norm"].dropna()))
    concept_only = local[
        local["concept_norm"].map(str).isin(fmri_concepts) & ~local["image_file"].isin(strict_files)
    ].copy()
    concept_only = concept_only.sort_values(["split", "image_index", "image_file"]).reset_index(
        drop=True
    )

    summary = {
        "n_local_total": int(len(local)),
        "n_local_train": int((local["split"] == "train").sum()),
        "n_local_test": int((local["split"] == "test").sum()),
        "n_fmri_rows": int(len(fmri)),
        "n_fmri_unique_files": int(fmri["image_file"].nunique()),
        "n_strict_same_image_matches": int(len(strict)),
        "n_strict_train": int((strict["split"] == "train").sum()),
        "n_strict_test": int((strict["split"] == "test").sum()),
        "n_concept_only_no_strict": int(len(concept_only)),
        "n_concept_only_train": int((concept_only["split"] == "train").sum()),
        "n_concept_only_test": int((concept_only["split"] == "test").sum()),
        "strict_match_kind_counts": strict["match_kind"].value_counts().to_dict(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strict.to_csv(args.out_dir / "things_fmri_same_image_overlap.csv", index=False)
    concept_only.to_csv(args.out_dir / "things_fmri_concept_only_overlap.csv", index=False)
    (args.out_dir / "things_fmri_same_image_overlap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
