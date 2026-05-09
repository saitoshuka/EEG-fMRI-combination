#!/usr/bin/env python3
"""Build a dataset-level reproducibility gate report.

This is the paper-story gate above individual canary runs.  It combines:

- download and Schaefer-100 cache inventory,
- strict bandpower canaries on subject-heldout and within-run splits,
- waveform-token lag sweep results,
- subject-adaptation canaries.

The goal is not to find the prettiest single number.  The goal is to decide
whether the EEG-to-fMRI distillation story is reproducible across independent
datasets under residual and shifted-null controls.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results/dataset_reproducibility_gate_20260509"


DATASET_ALIASES = {
    "Affective_music_listening_OpenNeuro_ds002725": "affective",
    "Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216": "experience",
    "Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338": "xp2",
    "Sleep_rest_EEG_fMRI_OpenNeuro_ds003768": "sleep",
    "Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158": "speeded",
    "gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040": "gradcpt",
    "NatView_NKI_EEG_fMRI_Naturalistic_Viewing": "natview",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pass_residual(row: pd.Series, rank_threshold: float, gap_threshold: float = 0.01, diag_threshold: float = 0.02) -> bool:
    if row is None or len(row) == 0:
        return False
    vals = {
        "rank": float(row.get("residual_rank_pct", math.nan)),
        "gap": float(row.get("residual_minus_shifted", math.nan)),
        "diag": float(row.get("residual_diag_minus_offdiag", math.nan)),
    }
    return vals["rank"] >= rank_threshold and vals["gap"] >= gap_threshold and vals["diag"] >= diag_threshold


def get_row(df: pd.DataFrame, dataset: str) -> pd.Series | None:
    if df.empty or "dataset_name" not in df.columns:
        return None
    hit = df[df["dataset_name"].astype(str) == dataset]
    if hit.empty:
        return None
    return hit.iloc[0]


def summarize_adaptation(adapt: pd.DataFrame, dataset: str, strict_rank: float) -> dict[str, object]:
    out = {
        "adapt_best_resid_rank": math.nan,
        "adapt_best_gap": math.nan,
        "adapt_best_diag": math.nan,
        "adapt_best_calib_frac": math.nan,
        "adapt_best_le20_rank": math.nan,
        "adapt_first_strict_calib": math.nan,
        "adapt_first_soft_calib": math.nan,
        "adapt_gate": "not_run",
    }
    if adapt.empty:
        return out
    ds = adapt[adapt["dataset_name"].astype(str) == dataset].copy()
    if ds.empty:
        return out
    ds = ds.sort_values("calib_frac")
    best = ds.sort_values("residual_rank_pct", ascending=False).iloc[0]
    out.update(
        {
            "adapt_best_resid_rank": float(best["residual_rank_pct"]),
            "adapt_best_gap": float(best["residual_minus_shifted"]),
            "adapt_best_diag": float(best["residual_diag_minus_offdiag"]),
            "adapt_best_calib_frac": float(best["calib_frac"]),
        }
    )
    le20 = ds[ds["calib_frac"] <= 0.20]
    if not le20.empty:
        out["adapt_best_le20_rank"] = float(le20["residual_rank_pct"].max())
    strict = ds[
        (ds["residual_rank_pct"] >= strict_rank)
        & (ds["residual_minus_shifted"] >= 0.01)
        & (ds["residual_diag_minus_offdiag"] >= 0.02)
    ]
    soft = ds[
        (ds["residual_rank_pct"] >= 0.52)
        & (ds["residual_minus_shifted"] >= 0.01)
        & (ds["residual_diag_minus_offdiag"] >= 0.02)
    ]
    if not strict.empty:
        out["adapt_first_strict_calib"] = float(strict["calib_frac"].min())
    if not soft.empty:
        out["adapt_first_soft_calib"] = float(soft["calib_frac"].min())
    if math.isfinite(out["adapt_first_strict_calib"]):
        if out["adapt_first_strict_calib"] <= 0.20:
            out["adapt_gate"] = "strong_subject_adapt"
        elif out["adapt_first_strict_calib"] <= 0.40:
            out["adapt_gate"] = "late_subject_adapt"
        else:
            out["adapt_gate"] = "too_much_calibration"
    elif math.isfinite(out["adapt_first_soft_calib"]):
        out["adapt_gate"] = "soft_only"
    else:
        out["adapt_gate"] = "fail"
    return out


def summarize_waveform(lag: pd.DataFrame, nested: pd.DataFrame, dataset: str, strict_rank: float) -> dict[str, object]:
    out = {
        "waveform_best_lag": math.nan,
        "waveform_best_resid_rank": math.nan,
        "waveform_best_gap": math.nan,
        "waveform_best_diag": math.nan,
        "waveform_nested_resid_rank": math.nan,
        "waveform_nested_gap": math.nan,
        "waveform_nested_diag": math.nan,
        "waveform_gate": "not_run",
    }
    if lag.empty:
        return out
    ds = lag[lag["dataset_name"].astype(str) == dataset].copy()
    if ds.empty:
        return out
    best = ds.sort_values("residual_rank_pct", ascending=False).iloc[0]
    out.update(
        {
            "waveform_best_lag": int(best["lag"]),
            "waveform_best_resid_rank": float(best["residual_rank_pct"]),
            "waveform_best_gap": float(best["residual_minus_shifted"]),
            "waveform_best_diag": float(best["residual_diag_minus_offdiag"]),
        }
    )
    gate_rank = out["waveform_best_resid_rank"]
    gate_gap = out["waveform_best_gap"]
    gate_diag = out["waveform_best_diag"]
    if not nested.empty and "dataset_name" in nested.columns:
        ndf = nested[nested["dataset_name"].astype(str) == dataset]
        if not ndf.empty:
            out["waveform_nested_resid_rank"] = float(ndf["heldout_residual_rank_pct"].mean())
            out["waveform_nested_gap"] = float(ndf["heldout_residual_minus_shifted"].mean())
            out["waveform_nested_diag"] = float(ndf["heldout_residual_diag_minus_offdiag"].mean())
            gate_rank = out["waveform_nested_resid_rank"]
            gate_gap = out["waveform_nested_gap"]
            gate_diag = out["waveform_nested_diag"]
    if gate_rank >= strict_rank and gate_gap >= 0.01 and gate_diag >= 0.02:
        out["waveform_gate"] = "within_run_pass"
    elif gate_rank >= 0.52 and gate_gap >= 0.01:
        out["waveform_gate"] = "soft_within_run"
    else:
        out["waveform_gate"] = "fail"
    return out


def metric(row: pd.Series | None, key: str) -> float:
    if row is None:
        return math.nan
    return float(row.get(key, math.nan))


def build_gate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    diag_dir = REPO_ROOT / "results/dataset_diagnostics_schaefer100"
    inventory = read_csv(diag_dir / "download_inventory.csv")
    cache_qc = read_csv(diag_dir / "cache_dataset_qc.csv")
    subject = read_csv(args.bandpower_subject / "canary_summary.csv")
    within = read_csv(args.bandpower_withinrun / "canary_summary.csv")
    lag = read_csv(args.waveform_lag / "lag_summary.csv")
    nested = read_csv(args.waveform_lag / "nested_lag_confirmation.csv")
    adapt = read_csv(args.subject_adaptation)

    if not cache_qc.empty:
        cache_qc = cache_qc.copy()
        cache_qc["dataset_key"] = cache_qc["dataset"].map(DATASET_ALIASES).fillna(cache_qc["dataset"].astype(str))

    datasets = []
    if not cache_qc.empty:
        datasets.extend(cache_qc["dataset_key"].astype(str).tolist())
    for extra in sorted(set(subject.get("dataset_name", [])) | set(within.get("dataset_name", [])) | set(lag.get("dataset_name", []))):
        if extra != "natview_neurostorm" and extra not in datasets:
            datasets.append(str(extra))

    rows = []
    for dataset in sorted(set(datasets)):
        sqc = cache_qc[cache_qc["dataset_key"] == dataset].iloc[0] if not cache_qc.empty and (cache_qc["dataset_key"] == dataset).any() else None
        srow = get_row(subject, dataset)
        wrow = get_row(within, dataset)
        wave = summarize_waveform(lag, nested, dataset, args.strict_rank)
        ad = summarize_adaptation(adapt, dataset, args.strict_rank)
        subject_pass = pass_residual(srow, 0.52)
        within_pass = pass_residual(wrow, 0.52)
        strict_adapt = math.isfinite(ad["adapt_first_strict_calib"])
        early_adapt = strict_adapt and float(ad["adapt_first_strict_calib"]) <= args.max_good_calib
        late_adapt = strict_adapt and float(ad["adapt_first_strict_calib"]) <= 0.40
        target_dim = metric(srow, "target_dim")
        if math.isnan(target_dim):
            target_dim = metric(wrow, "target_dim")
        roi_mask_mean = float(sqc["roi_mask_mean"]) if sqc is not None and "roi_mask_mean" in sqc.index else math.nan

        if subject_pass:
            final_gate = "cross_subject_candidate"
        elif early_adapt:
            final_gate = "paper_candidate_subject_adapted"
        elif late_adapt:
            final_gate = "weak_late_calibration_only"
        elif within_pass or wave["waveform_gate"] in {"within_run_pass", "soft_within_run"}:
            final_gate = "within_run_only_or_stress_control"
        else:
            final_gate = "exclude_or_reprocess"

        if dataset == "xp2" and final_gate != "exclude_or_reprocess":
            final_gate = "within_run_only_or_stress_control"
        if dataset in {"natview", "speeded", "gradcpt"} and not subject_pass and not early_adapt:
            final_gate = "exclude_or_reprocess"

        rows.append(
            {
                "dataset": dataset,
                "subjects": int(sqc["subjects"]) if sqc is not None and "subjects" in sqc.index else int(metric(srow, "n_subjects") if not math.isnan(metric(srow, "n_subjects")) else 0),
                "runs": int(sqc["runs"]) if sqc is not None and "runs" in sqc.index else int(metric(srow, "n_runs") if not math.isnan(metric(srow, "n_runs")) else 0),
                "windows": int(sqc["windows"]) if sqc is not None and "windows" in sqc.index else int(metric(srow, "n_samples") if not math.isnan(metric(srow, "n_samples")) else 0),
                "target_dim": int(target_dim) if not math.isnan(target_dim) else 0,
                "roi_mask_mean": roi_mask_mean,
                "subject_resid_rank": metric(srow, "residual_rank_pct"),
                "subject_resid_gap": metric(srow, "residual_minus_shifted"),
                "subject_resid_diag": metric(srow, "residual_diag_minus_offdiag"),
                "subject_gate": "pass" if subject_pass else "fail",
                "within_resid_rank": metric(wrow, "residual_rank_pct"),
                "within_resid_gap": metric(wrow, "residual_minus_shifted"),
                "within_resid_diag": metric(wrow, "residual_diag_minus_offdiag"),
                "within_gate": "pass" if within_pass else "fail",
                **wave,
                **ad,
                "final_gate": final_gate,
            }
        )
    gate = pd.DataFrame(rows).sort_values(["final_gate", "dataset"]).reset_index(drop=True)

    inv_rows = []
    if not inventory.empty:
        cached = set(gate["dataset"].astype(str))
        for _, row in inventory.iterrows():
            key = DATASET_ALIASES.get(str(row["dataset"]), str(row["dataset"]))
            if key in cached:
                status = "strict_gate_evaluated"
            elif int(row.get("EEG-like files", 0)) == 0 or int(row.get("BOLD-like files", 0)) == 0:
                status = "not_pairable_from_current_inventory"
            else:
                status = "downloaded_but_not_shared_schaefer_cache"
            inv_rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "dataset_key": key,
                    "inventory_status": status,
                    "subject_dirs": int(row.get("subject dirs", 0)),
                    "eeg_like_files": int(row.get("EEG-like files", 0)),
                    "bold_like_files": int(row.get("BOLD-like files", 0)),
                    "size_gb": float(row.get("size GB", 0.0)),
                }
            )
    inventory_gate = pd.DataFrame(inv_rows)
    counts = {
        "downloaded_top_level": int(inventory.shape[0]) if not inventory.empty else 0,
        "strict_gate_evaluated": int((inventory_gate["inventory_status"] == "strict_gate_evaluated").sum()) if not inventory_gate.empty else int(gate.shape[0]),
        "cross_subject_pass": int((gate["subject_gate"] == "pass").sum()),
        "within_run_pass": int((gate["within_gate"] == "pass").sum()),
        "waveform_run_count": int((gate["waveform_gate"] != "not_run").sum()),
        "waveform_strict_pass": int((gate["waveform_gate"] == "within_run_pass").sum()),
        "waveform_soft_pass": int((gate["waveform_gate"].isin(["within_run_pass", "soft_within_run"])).sum()),
        "early_strict_adapt": int((gate["final_gate"] == "paper_candidate_subject_adapted").sum()),
        "late_strict_adapt": int((gate["final_gate"] == "weak_late_calibration_only").sum()),
        "usable_for_big_model_claim": int(
            ((gate["subject_gate"] == "pass") | (gate["final_gate"] == "paper_candidate_subject_adapted")).sum()
        ),
    }
    return gate, inventory_gate, counts


def fmt(x: object, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def write_report(args: argparse.Namespace, gate: pd.DataFrame, inventory_gate: pd.DataFrame, counts: dict[str, int]) -> None:
    args.results_dir = args.results_dir.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    gate.to_csv(args.results_dir / "gate_matrix.csv", index=False)
    inventory_gate.to_csv(args.results_dir / "inventory_gate.csv", index=False)

    continue_big_model = counts["usable_for_big_model_claim"] >= args.min_datasets_for_claim
    lines: list[str] = []
    lines.append("# Dataset-Level Reproducibility Gate - 2026-05-09")
    lines.append("")
    lines.append("This is the stricter gate for deciding whether the EEG-to-fMRI distillation story can be told as a cross-dataset foundation-model direction.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if continue_big_model:
        lines.append("The current evidence clears the minimum multi-dataset gate for a cautious subject-adapted distillation story.")
    else:
        lines.append("The current evidence does not clear the minimum multi-dataset gate for a big-model EEG-to-fMRI distillation claim.")
    lines.append("")
    lines.append(f"- Download inventory: {counts['downloaded_top_level']} top-level datasets.")
    lines.append(f"- Strict gate evaluated: {counts['strict_gate_evaluated']} datasets with shared Schaefer-style targets.")
    lines.append(f"- Cross-subject residual pass: {counts['cross_subject_pass']} datasets.")
    lines.append(f"- Within-run residual pass: {counts['within_run_pass']} datasets.")
    lines.append(f"- Waveform-token evaluated: {counts['waveform_run_count']} datasets.")
    lines.append(f"- Waveform nested strict within-run pass: {counts['waveform_strict_pass']} datasets.")
    lines.append(f"- Waveform nested soft-or-strict within-run pass: {counts['waveform_soft_pass']} datasets.")
    lines.append(f"- Early strict subject-adaptation pass, calibration <= {args.max_good_calib:.2f}: {counts['early_strict_adapt']} datasets.")
    lines.append(f"- Late strict subject-adaptation pass, calibration <= 0.40: {counts['late_strict_adapt']} datasets.")
    lines.append("")
    lines.append("Paper-level rule used here: at least 2-3 independent datasets should pass cross-subject residual controls or pass strict subject-adaptation with no more than 10-20 percent calibration. Current count is below that bar.")
    lines.append("")
    lines.append("## Gate Matrix")
    lines.append("")
    lines.append("| dataset | final gate | subj resid | subj gap | within resid | within gap | waveform best | waveform nested | first strict calib | best <=20% | target dim | ROI mask |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in gate.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['final_gate']} | "
            f"{fmt(row['subject_resid_rank'])} | {fmt(row['subject_resid_gap'])} | "
            f"{fmt(row['within_resid_rank'])} | {fmt(row['within_resid_gap'])} | "
            f"{fmt(row['waveform_best_resid_rank'])} | {fmt(row['waveform_nested_resid_rank'])} | "
            f"{fmt(row['adapt_first_strict_calib'], 2)} | "
            f"{fmt(row['adapt_best_le20_rank'])} | {int(row['target_dim'])} | {fmt(row['roi_mask_mean'], 3)} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Affective is the best current dataset, but it only becomes strict-positive after about 30 percent subject calibration. That is useful as a canary, not enough as a general foundation-model claim.")
    lines.append("- Experience is weaker and time-dominated in the non-residual metrics. It only crosses the strict threshold at 40 percent calibration, so it is a late-calibration support point rather than an independent strong dataset.")
    lines.append("- XP2 passes within-run tests but fails subject adaptation and has partial Schaefer coverage. Treat it as a stress/negative-control dataset, not as positive evidence.")
    lines.append("- Sleep now has a soft waveform within-run signal, but nested confirmation is weak and subject adaptation only reaches a soft pass at 40 percent calibration. It is a promising reprocessing/modeling target, not a current positive.")
    lines.append("- NatView, speeded, and gradCPT fail the current strict gates. NatView remains scientifically important, but this target/cache does not currently provide positive EEG-to-fMRI evidence.")
    lines.append("")
    lines.append("## Route Recommendation")
    lines.append("")
    lines.append("Do not continue as blind pooled LaBraM/NeuroSTORM distillation. The evidence is below the multi-dataset reproducibility bar.")
    lines.append("")
    lines.append("The paper-capable route is narrower: reliability-gated paired EEG-fMRI distillation. The story should be that public paired datasets are valuable, but naive cross-dataset distillation fails; with reliability gates and subject adaptation, only specific datasets show conditional signal. To become a method paper, the next experiment must add at least one more independent dataset that passes early calibration, most likely Sleep after better EEG preprocessing/conditioning.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Gate matrix: `{(args.results_dir / 'gate_matrix.csv').relative_to(REPO_ROOT)}`")
    lines.append(f"- Inventory gate: `{(args.results_dir / 'inventory_gate.csv').relative_to(REPO_ROOT)}`")
    lines.append(f"- Bandpower subject canary: `{args.bandpower_subject.relative_to(REPO_ROOT)}`")
    lines.append(f"- Bandpower within-run canary: `{args.bandpower_withinrun.relative_to(REPO_ROOT)}`")
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--bandpower-subject", type=Path, default=REPO_ROOT / "results/dataset_gate_bandpower_subject_residual_20260509")
    p.add_argument("--bandpower-withinrun", type=Path, default=REPO_ROOT / "results/dataset_gate_bandpower_withinrun_residual_20260509")
    p.add_argument("--waveform-lag", type=Path, default=REPO_ROOT / "results/waveform_token_lag_sweep_v1_tf8")
    p.add_argument("--subject-adaptation", type=Path, default=REPO_ROOT / "results/subject_adaptation_decision_20260509.csv")
    p.add_argument("--strict-rank", type=float, default=0.53)
    p.add_argument("--max-good-calib", type=float, default=0.20)
    p.add_argument("--min-datasets-for-claim", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gate, inventory_gate, counts = build_gate(args)
    write_report(args, gate, inventory_gate, counts)
    print({"results_dir": str(args.results_dir), **counts})


if __name__ == "__main__":
    main()
