#!/usr/bin/env python3
"""Write the focused residual ROI scaling table requested for the 16k run."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FOCUS_ROWS = [
    ("group", "all_visual", "group12 residual all_visual"),
    ("parcel", "all_visual", "parcel38 residual all_visual"),
    ("group", "early_visual_combined", "group12 residual early_visual_combined"),
    ("parcel", "early_visual_combined", "parcel38 residual early_visual_combined"),
    ("group", "high_level_combined", "group12 residual high_level_combined"),
    ("parcel", "high_level_combined", "parcel38 residual high_level_combined"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_float(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return math.nan
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "..."
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.scaling_csv)
    by_key = {(row["roi_kind"], row["roi_group"]): row for row in rows}

    table_rows = []
    for roi_kind, roi_group, label in FOCUS_ROWS:
        row = by_key.get((roi_kind, roi_group))
        rank_8192 = get_float(row, "rank_8192")
        rank_16k = get_float(row, "rank_16540")
        delta = rank_16k - rank_8192 if not math.isnan(rank_16k) and not math.isnan(rank_8192) else math.nan
        table_rows.append(
            [
                label,
                fmt(get_float(row, "rank_4096")),
                fmt(rank_8192),
                fmt(rank_16k),
                fmt(get_float(row, "shifted_rank_4096")),
                fmt(get_float(row, "shifted_rank_8192")),
                fmt(get_float(row, "shifted_rank_16540")),
                fmt(delta),
            ]
        )

    lines = [
        "# Residual ROI Scaling Table",
        "",
        "| target | 4096 rank | 8192 rank | 16k rank | 4096 shifted | 8192 shifted | 16k shifted | 8192->16k delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "Best 16k run is selected per ROI kind/group by final ROI rank when multiple lambda values are available.",
            f"Source CSV: `{args.scaling_csv}`",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
