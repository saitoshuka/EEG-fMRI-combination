#!/usr/bin/env python3
"""Audit local paired EEG-fMRI dataset metadata.

This is intentionally lightweight: it scans local BIDS-like sidecars and README
files without loading large imaging files.  The output is a first-pass inventory
for deciding dataset-specific preprocessing policies.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def compact_text(text: str, limit: int = 800) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def scalar_values(paths: list[Path], key: str, limit: int = 30) -> list[str]:
    vals: list[str] = []
    for path in paths[:limit]:
        obj = read_json(path)
        if key in obj:
            vals.append(str(obj[key]))
    return sorted(set(vals))


def channel_type_counts(paths: list[Path], limit: int = 10) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths[:limit]:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if "type" not in df:
            continue
        for value, count in df["type"].astype(str).value_counts().items():
            counts[value] = counts.get(value, 0) + int(count)
    return counts


def event_onset_min(paths: list[Path], limit: int = 30) -> float | None:
    vals: list[float] = []
    for path in paths[:limit]:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            continue
        if "onset" not in df:
            continue
        onset = pd.to_numeric(df["onset"], errors="coerce")
        if onset.notna().any():
            vals.append(float(onset.min()))
    return min(vals) if vals else None


def audit_dataset(path: Path) -> dict[str, object]:
    desc = read_json(path / "dataset_description.json")
    bold_jsons = sorted(path.rglob("*_bold.json"))
    eeg_jsons = sorted(path.rglob("*_eeg.json"))
    channel_tsvs = sorted(path.rglob("*_channels.tsv"))
    event_tsvs = sorted(path.rglob("*_events.tsv"))
    bold_files = sorted(path.rglob("*_bold.nii*"))
    eeg_files = sorted(path.rglob("*_eeg.*"))

    readme = ""
    for readme_path in sorted(list(path.glob("README*")) + list(path.glob("README.*")))[:3]:
        try:
            readme += "\n" + readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    authors = desc.get("Authors", "")
    if isinstance(authors, list):
        authors = "; ".join(str(a) for a in authors[:8])
    return {
        "dataset": path.name,
        "name": desc.get("Name", ""),
        "authors": authors,
        "dataset_doi": desc.get("DatasetDOI", ""),
        "license": desc.get("License", ""),
        "n_bold_json": len(bold_jsons),
        "n_bold_files": len(bold_files),
        "tr_values": "|".join(scalar_values(bold_jsons, "RepetitionTime")),
        "echo_values": "|".join(scalar_values(bold_jsons, "EchoTime")),
        "bold_task_names": "|".join(scalar_values(bold_jsons, "TaskName")),
        "n_eeg_json": len(eeg_jsons),
        "n_eeg_files": len(eeg_files),
        "eeg_sampling_frequency": "|".join(scalar_values(eeg_jsons, "SamplingFrequency")),
        "eeg_reference": "|".join(scalar_values(eeg_jsons, "EEGReference")),
        "eeg_task_names": "|".join(scalar_values(eeg_jsons, "TaskName")),
        "channel_type_counts": json.dumps(channel_type_counts(channel_tsvs), sort_keys=True),
        "event_onset_min": event_onset_min(event_tsvs),
        "readme_hint": compact_text(readme),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("downloads/paired_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_source_metadata_audit"))
    args = parser.parse_args()

    rows = [audit_dataset(path) for path in sorted(args.root.iterdir()) if path.is_dir()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "local_metadata_summary.csv", index=False)

    lines = [
        "# Local Dataset Source Metadata Audit",
        "",
        f"- Root: `{args.root}`",
        f"- Datasets: {len(frame)}",
        f"- CSV: `{args.output_dir / 'local_metadata_summary.csv'}`",
        "",
        "| dataset | bold files | eeg files | TR | EEG Hz | EEG ref | event min |",
        "| --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in frame.to_dict("records"):
        event_min = row["event_onset_min"]
        event_txt = "" if pd.isna(event_min) else f"{float(event_min):.3f}"
        lines.append(
            f"| {row['dataset']} | {row['n_bold_files']} | {row['n_eeg_files']} | "
            f"{row['tr_values']} | {row['eeg_sampling_frequency']} | {row['eeg_reference']} | {event_txt} |"
        )
    (args.output_dir / "local_metadata_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"datasets": len(frame), "output_dir": str(args.output_dir)}, flush=True)


if __name__ == "__main__":
    main()
