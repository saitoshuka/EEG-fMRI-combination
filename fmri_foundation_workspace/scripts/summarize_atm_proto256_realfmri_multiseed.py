#!/usr/bin/env python3
"""Summarize multiseed ATM proto256 real-fMRI probe results.

The probe directory contains one summary.json per seed/target-label run. This
script intentionally accepts both older labels such as
``proto256_pooled_residual_seed33_to_realfmri_visual64`` and newer labels such
as ``proto256_pooled_residual_seed11_bestroi_visual64``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_feature_to_realfmri_probe"
)
DEFAULT_OUT_DIR = DEFAULT_PROBE_DIR / "multiseed_summary"


def infer_target_label(name: str) -> str | None:
    if "visual64" in name:
        return "visual64"
    if "shared207" in name:
        return "shared207"
    return None


def infer_seed(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def discover_summaries(probe_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for path in sorted(probe_dir.glob(f"{prefix}*/summary.json")):
        name = path.parent.name
        if infer_seed(name) is None or infer_target_label(name) is None:
            continue
        paths.append(path)
    return paths


def load_rows(summary_path: Path) -> list[dict[str, object]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    name = summary_path.parent.name
    seed = infer_seed(name)
    target_label = infer_target_label(name)
    if seed is None or target_label is None:
        return []
    rows = []
    for result in payload["results"]:
        test = result["test"]
        rows.append(
            {
                "run": name,
                "seed": seed,
                "target_label": target_label,
                "feature_set": result["feature_set"],
                "feature_dim": result["feature_dim"],
                "rank_percentile": test["rank_percentile"],
                "shifted_rank_percentile": test["shifted_rank_percentile"],
                "rank_minus_shifted": test["rank_percentile"] - test["shifted_rank_percentile"],
                "top1": test["top1"],
                "top5": test["top5"],
                "diag_minus_offdiag": test["diag_minus_offdiag"],
                "image_pattern_corr_mean": test["image_pattern_corr_mean"],
                "roi_corr_fisher_mean": test["roi_corr_fisher_mean"],
                "selected_alpha": test["selected_alpha"],
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["target_label"]), str(row["feature_set"]))].append(row)

    summary_rows = []
    metrics = [
        "rank_percentile",
        "shifted_rank_percentile",
        "rank_minus_shifted",
        "top1",
        "top5",
        "diag_minus_offdiag",
        "image_pattern_corr_mean",
        "roi_corr_fisher_mean",
    ]
    for (target_label, feature_set), items in sorted(groups.items()):
        seeds = sorted(int(row["seed"]) for row in items)
        out: dict[str, object] = {
            "target_label": target_label,
            "feature_set": feature_set,
            "n_seeds": len(items),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in items]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# ATM proto256 real-fMRI multiseed summary",
        "",
        "## Per-run metrics",
        "",
        "| seed | target | feature | rank | shifted | delta | top1 | roi corr | image corr |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (str(r["target_label"]), str(r["feature_set"]), int(r["seed"]))):
        lines.append(
            "| {seed} | {target_label} | {feature_set} | {rank_percentile:.4f} | "
            "{shifted_rank_percentile:.4f} | {rank_minus_shifted:.4f} | {top1:.4f} | "
            "{roi_corr_fisher_mean:.4f} | {image_pattern_corr_mean:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Aggregate metrics",
            "",
            "| target | feature | n | seeds | rank mean | rank std | delta mean | delta std | top1 mean | roi corr mean |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {target_label} | {feature_set} | {n_seeds} | {seeds} | "
            "{rank_percentile_mean:.4f} | {rank_percentile_std:.4f} | "
            "{rank_minus_shifted_mean:.4f} | {rank_minus_shifted_std:.4f} | "
            "{top1_mean:.4f} | {roi_corr_fisher_mean_mean:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--prefix", default="proto256_pooled_residual_seed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="pooled_proto256_residual_realfmri_multiseed")
    args = parser.parse_args()

    summaries = discover_summaries(args.probe_dir, args.prefix)
    rows: list[dict[str, object]] = []
    for summary_path in summaries:
        rows.extend(load_rows(summary_path))

    summary_rows = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = args.out_dir / f"{args.label}_per_run.csv"
    aggregate_csv = args.out_dir / f"{args.label}_aggregate.csv"
    markdown_path = args.out_dir / f"{args.label}.md"
    write_csv(per_run_csv, rows)
    write_csv(aggregate_csv, summary_rows)
    write_markdown(markdown_path, rows, summary_rows)
    print(
        json.dumps(
            {
                "n_summary_files": len(summaries),
                "n_rows": len(rows),
                "summary_files": [str(path) for path in summaries],
                "per_run_csv": str(per_run_csv),
                "aggregate_csv": str(aggregate_csv),
                "markdown": str(markdown_path),
                "aggregate": summary_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
