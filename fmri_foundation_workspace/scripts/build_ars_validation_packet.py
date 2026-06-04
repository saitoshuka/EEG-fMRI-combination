#!/usr/bin/env python3
"""Build an ARS-compatible validation packet from current EEG bridge results."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WS = Path(__file__).resolve().parents[1]
ROOT = WS.parent
RESULTS = WS / "results" / "eeg_image_bridge"
NOTES = WS / "notes" / "eeg_image_bridge"


SOURCES = {
    "reviewer_risk": RESULTS
    / "target_space_ablation"
    / "reviewer_risk_summary.json",
    "target_space_metrics": RESULTS
    / "target_space_ablation"
    / "target_space_ablation_metrics.csv",
    "paired_sig_lam005": RESULTS
    / "atm_roi_spatial_branch"
    / "paired_significance_16k_residual_parcel_lam005_vs_semantic"
    / "summary.json",
    "residual_parcel_lam005_summary": RESULTS
    / "atm_roi_spatial_branch"
    / "atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005"
    / "summary.json",
    "vjepa_residual_bootstrap": RESULTS
    / "short_validation"
    / "vjepa2_residual_bootstrap_16k.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def f4(x: Any) -> str:
    if x is None or x == "":
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x: Any) -> str:
    if x is None or x == "":
        return "n/a"
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return str(x)


def ci_text(ci: list[float] | tuple[float, float] | None) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def row_for(rows: list[dict[str, str]], model: str, train_images: int) -> dict[str, str]:
    for row in rows:
        if row.get("model") == model and int(float(row.get("train_images", -1))) == train_images:
            return row
    raise KeyError((model, train_images))


def bullet_sources() -> str:
    lines = []
    for name, path in SOURCES.items():
        rel = path.relative_to(ROOT)
        lines.append(f"- `{name}`: `{rel}`; sha256[0:12]=`{sha256_short(path)}`")
    return "\n".join(lines)


def make_report() -> str:
    reviewer = load_json(SOURCES["reviewer_risk"])
    target_rows = load_csv(SOURCES["target_space_metrics"])
    sig = load_json(SOURCES["paired_sig_lam005"])
    residual_summary = load_json(SOURCES["residual_parcel_lam005_summary"])
    vjepa_resid = load_json(SOURCES["vjepa_residual_bootstrap"])

    boot = {int(item["size"]): item for item in reviewer["target_space_bootstrap"]}
    frozen_16k = row_for(target_rows, "frozen_atm_clip_direct", 16540)
    vjepa_16k = row_for(target_rows, "atm_to_vjepa_ridge", 16540)
    vjepa_to_clip_16k = row_for(target_rows, "atm_to_vjepa_to_clip", 16540)
    combo_16k = row_for(target_rows, "atm_to_clip_plus_vjepa_concat", 16540)

    sem = sig["metrics"]["semantic_final"]
    res = sig["metrics"]["residual_parcel_lam005_final"]
    comp = sig["comparisons"]["final_vs_semantic"]
    res_roi_row = residual_summary["best_checkpoints"]["best_roi_rank"]["row"]

    cortical = reviewer["cortical_binding"]
    raw_strong = cortical["raw_parcel_strong"]
    res_parcel = cortical["residual_parcel"]

    raw_time = {x["roi_group"]: x for x in reviewer["raw_time"]}
    residual_time = {x["roi_group"]: x for x in reviewer["residual_time"]}
    raw_channel = reviewer["raw_channel"]
    residual_channel = reviewer["residual_channel"]

    vjepa_latent32 = vjepa_resid["vjepa2_only"]["latent32_surface"]
    clip_vjepa_latent32 = vjepa_resid["clip_plus_vjepa2"]["latent32_surface"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: {now}
- Verification Status: ANALYZED
- Version Label: eeg_visual_cortical_bridge_validation_v1
- Working Directory: `{ROOT}`

## Validation Report

- **Source**: EEG visual decoding / TRIBE-V-JEPA cortical supervision experiments
- **Overall Confidence**: CAUTION
- **Validation Question**: Does cortical or V-JEPA-derived supervision improve EEG image retrieval and preserve query-specific pseudo-cortical structure, or is V-JEPA alone enough?

### Evidence Sources

{bullet_sources()}

### Statistical Findings

| Finding | Test / control | Value | Effect Size | Confidence |
|---|---|---:|---:|---|
| Residual parcel38 branch vs same-framework semantic-only | paired 200-image test; final checkpoint | top1 {f4(sem["top1"])} -> {f4(res["top1"])} | diff {f4(comp["top1"]["diff"])}; CI {ci_text(comp["top1"]["ci95"])}; McNemar p {f4(comp["counts"]["mcnemar_p_two_sided"])} | CAUTION |
| Residual parcel38 ROI signal | shifted-null control | ROI rank {f4(res_roi_row["roi_rank_percentile"])} vs shifted {f4(res_roi_row["roi_shifted_rank_percentile"])} | gap {f4(float(res_roi_row["roi_rank_percentile"]) - float(res_roi_row["roi_shifted_rank_percentile"]))} | SOLID for non-random ROI signal |
| V-JEPA-only target-space retrieval scales but remains below frozen ATM->CLIP | 16540 train images, 200-image test | V-JEPA top1 {f4(vjepa_16k["vjepa_test_top1"])} vs frozen CLIP top1 {f4(frozen_16k["clip_test_top1"])} | gap {f4(float(vjepa_16k["vjepa_test_top1"]) - float(frozen_16k["clip_test_top1"]))} | SOLID against "V-JEPA alone is enough" |
| V-JEPA->CLIP is not a CLIP replacement | 16540 train images, 200-image test | CLIP top1 {f4(vjepa_to_clip_16k["clip_test_top1"])} vs frozen {f4(frozen_16k["clip_test_top1"])} | large negative gap | SOLID |
| Frozen CLIP + V-JEPA->CLIP blend | bootstrap paired differences | 8192 top1 +{f4(boot[8192]["top1_diff"]["diff"])} CI {ci_text(boot[8192]["top1_diff"]["ci95"])}; 16540 top1 +{f4(boot[16540]["top1_diff"]["diff"])} CI {ci_text(boot[16540]["top1_diff"]["ci95"])} | small, CI crosses 0 | CAUTION |
| CLIP+V-JEPA concat target-space retrieval | 16540 train images | combo top1 {f4(combo_16k["combo_test_top1"])}, top5 {f4(combo_16k["combo_test_top5"])}, rank {f4(combo_16k["combo_test_rank_percentile"])} | useful but below frozen CLIP top1/rank | CAUTION |
| Raw parcel38 query identity | query-target correlation, shuffled query order | diag-offdiag {f4(raw_strong["diag_minus_offdiag_mean"])} vs shuffled mean {f4(raw_strong["shuffled_diag_minus_offdiag_mean"])} | diag rank pct {f4(raw_strong["diag_rank_percentile_mean"])} | SOLID |
| Residual parcel38 query identity | query-target correlation, shuffled query order | diag-offdiag {f4(res_parcel["diag_minus_offdiag_mean"])} vs shuffled mean {f4(res_parcel["shuffled_diag_minus_offdiag_mean"])} | target-geometry corr {f4(res_parcel["query_target_vs_target_self_matrix_corr_all"])} | SOLID but weaker amplitude |
| V-JEPA residual spatial prototypes | shifted-null bootstrap | V-JEPA latent32 surface gap {f4(vjepa_latent32["gap"])} CI {ci_text(vjepa_latent32["bootstrap_ci95"])} p {f4(vjepa_latent32["signflip_p_two_sided"])} | positive residual signal | CAUTION/SOLID |
| CLIP+V-JEPA residual spatial prototypes | shifted-null bootstrap | latent32 surface gap {f4(clip_vjepa_latent32["gap"])} CI {ci_text(clip_vjepa_latent32["bootstrap_ci95"])} p {f4(clip_vjepa_latent32["signflip_p_two_sided"])} | positive residual signal | CAUTION/SOLID |

### Cortical Structure Check

| Probe | Raw parcel38 strong | Residual parcel38 | Interpretation |
|---|---:|---:|---|
| Query-target binding | diag-offdiag {f4(raw_strong["diag_minus_offdiag_mean"])} | diag-offdiag {f4(res_parcel["diag_minus_offdiag_mean"])} | raw is visibly stronger; residual remains above shuffled null |
| Target geometry preservation | {f4(raw_strong["query_target_vs_target_self_matrix_corr_all"])} | {f4(res_parcel["query_target_vs_target_self_matrix_corr_all"])} | residual preserves target-space geometry surprisingly well |
| Early calcarine timing | best keep {raw_time["early_calcarine"]["best_keep_window"]}; drop {raw_time["early_calcarine"]["most_important_drop_window"]} | best keep {residual_time["early_calcarine"]["best_keep_window"]}; drop {residual_time["early_calcarine"]["most_important_drop_window"]} | early visual structure is present in both |
| High-level timing | best keep {raw_time["high_level_combined"]["best_keep_window"]}; drop {raw_time["high_level_combined"]["most_important_drop_window"]} | best keep {residual_time["high_level_combined"]["best_keep_window"]}; drop {residual_time["high_level_combined"]["most_important_drop_window"]} | high-level staging is clearer in raw than residual |
| Posterior EEG channel concentration | posterior/all {f4(raw_channel["posterior_over_all"])}; top {raw_channel["top"][0][0]} | posterior/all {f4(residual_channel["posterior_over_all"])}; top {residual_channel["top"][0][0]} | O/PO/P dominance matches visual EEG, but is not proof of ROI-specific channel selection |

### Claim Readiness

| Candidate claim | Status | Why |
|---|---|---|
| ROI-query cortical supervision can add retrieval benefit within a trainable ATM-style EEG decoder | PARTIAL | top1 improves 0.385 -> 0.430, but CI touches zero and McNemar is not conventionally significant |
| The ROI branch learns non-random pseudo-cortical structure rather than arbitrary ROI labels | SUPPORTED | query-target diagonal advantage, target-geometry correlation, shifted/shuffled nulls, and posterior channel maps agree |
| Residual targets provide beyond-CLIP evidence | PARTIAL | residual ROI rank and query identity remain above null, but visual/time maps are weaker and effect sizes are modest |
| V-JEPA alone can replace CLIP as the EEG retrieval/generation target | NOT SUPPORTED | V-JEPA-space retrieval works, but CLIP-space via V-JEPA is much worse than frozen ATM->CLIP |
| V-JEPA is a useful auxiliary teacher or reviewer-control feature | SUPPORTED | V-JEPA features explain/organize pseudo-cortical targets and provide small blend gains, but not a standalone interface |
| The current results prove real fMRI/neuroscience alignment | NOT YET SUPPORTED | targets are pseudo-cortical outputs from foundation teachers; real fMRI external validation remains required |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Small test set | Main retrieval tests use 200 averaged THINGS-EEG test images; top1 CIs are wide. | retrieval gains |
| Pseudo-target teacher | TRIBE/V-JEPA targets are model-generated cortical proxies, not measured fMRI. | neuroscience claims |
| Best-checkpoint protocol | Some historical ATM-style runs selected best checkpoints on test-like metrics; final-checkpoint comparisons are cleaner. | model comparison |
| Multiple exploratory comparisons | Many target spaces, losses, and probes have been tried. | confirmatory p-values |
| Baseline mismatch | Trainable ATM semantic-only is weaker than frozen/pretrained ATM->CLIP embedding. | SOTA comparison |
| Gated DINOv3 access | Official `facebook/dinov3-*` weights were not locally runnable without authenticated access. | DINO/V-JEPA generality |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | No subgroup reversal test across subjects/classes has been run in this packet. | Add per-subject and per-class stratified retrieval/ROI analyses before strong general claims. |
| Ecological fallacy | CAUTION | Current cortical claims use group-level pseudo-targets and averaged test EEG. | Avoid individual-subject neuroscience claims unless validated per subject. |
| Berkson's paradox | NOTE | THINGS-EEG stimulus/test set is curated, not a natural image sample. | Phrase as visual decoding on THINGS-like stimuli. |
| Collider bias | NOTE | No explicit causal/control regression is used in the main evidence. | Keep controls descriptive. |
| Base-rate neglect | NOTE | Retrieval top-k is balanced over 200 test images. | Report chance level and candidate set size. |
| Regression to mean | NOTE | No pre/post extreme-group design. | Not central. |
| Survivorship bias | CAUTION | Trial averaging and usable EEG preprocessing can select higher-SNR data. | Report trial inclusion, repeats, and averaging explicitly. |
| Look-elsewhere effect | CAUTION | Many windows, ROI groups, and losses were explored. | Mark cortical maps as exploratory unless repeated on heldout analysis choices. |
| Garden of forking paths | CAUTION | Pipeline evolved through many target definitions and loss variants. | Freeze a small confirmatory protocol before paper numbers. |
| Correlation != causation | CAUTION | Query/ROI correlations do not prove biological causality or true fMRI encoding. | Use "associated with", "predicts proxy target", and "consistent with". |
| Reverse causality | NOTE | Direction is stimulus -> EEG/model target by design, but teacher targets are derived from stimulus. | Be explicit that stimulus mediation, not paired fMRI causality, is being tested. |

### Reproducibility

- **Method**: structured-result re-read plus deterministic report regeneration; no full GPU retraining in this validation pass.
- **Verdict**: PARTIALLY_REPRODUCIBLE

| Metric | Source | Re-run | Diff | Status |
|---|---:|---:|---:|---|
| ARS packet generation | source JSON/CSV files | this report | n/a | REGENERATED |
| Model training | historical GPU runs | not re-run | n/a | CANNOT_VERIFY in this pass |

### Decision Output

The strongest paper direction is **not** "V-JEPA replaces CLIP" and not yet
"we beat SOTA". The defensible direction is:

> Keep CLIP as the semantic retrieval/generation interface, and use
> ROI-query cortical supervision as an auxiliary branch that contributes
> modest retrieval gains and query-specific pseudo-cortical structure. Use raw
> targets for visual explanation and residual targets as a stricter beyond-CLIP
> control.

Minimum next confirmatory step:

1. Freeze one protocol: semantic-only vs residual parcel38 vs raw parcel38 under the same trainable ATM budget.
2. Repeat on at least 3 seeds or bootstrap subjects/classes.
3. Add real-fMRI or external cortical-response validation before making neuroscience-alignment claims.
"""


def main() -> None:
    missing = [str(p) for p in SOURCES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing sources:\n" + "\n".join(missing))
    NOTES.mkdir(parents=True, exist_ok=True)
    out = NOTES / "ars_validation_eeg_visual_cortical_bridge_20260604.md"
    out.write_text(make_report(), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
