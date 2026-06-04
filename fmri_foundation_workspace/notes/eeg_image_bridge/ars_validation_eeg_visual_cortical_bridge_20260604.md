## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-04T08:29:16Z
- Verification Status: ANALYZED
- Version Label: eeg_visual_cortical_bridge_validation_v1
- Working Directory: `/home/sudaxin/projects/paired_data`

## Validation Report

- **Source**: EEG visual decoding / TRIBE-V-JEPA cortical supervision experiments
- **Overall Confidence**: CAUTION
- **Validation Question**: Does cortical or V-JEPA-derived supervision improve EEG image retrieval and preserve query-specific pseudo-cortical structure, or is V-JEPA alone enough?

### Evidence Sources

- `reviewer_risk`: `fmri_foundation_workspace/results/eeg_image_bridge/target_space_ablation/reviewer_risk_summary.json`; sha256[0:12]=`883fe2248ace`
- `target_space_metrics`: `fmri_foundation_workspace/results/eeg_image_bridge/target_space_ablation/target_space_ablation_metrics.csv`; sha256[0:12]=`09f60c14788a`
- `paired_sig_lam005`: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/paired_significance_16k_residual_parcel_lam005_vs_semantic/summary.json`; sha256[0:12]=`125572486a1c`
- `residual_parcel_lam005_summary`: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/summary.json`; sha256[0:12]=`83a820cdf02f`
- `vjepa_residual_bootstrap`: `fmri_foundation_workspace/results/eeg_image_bridge/short_validation/vjepa2_residual_bootstrap_16k.json`; sha256[0:12]=`5056d07c3019`

### Statistical Findings

| Finding | Test / control | Value | Effect Size | Confidence |
|---|---|---:|---:|---|
| Residual parcel38 branch vs same-framework semantic-only | paired 200-image test; final checkpoint | top1 0.3850 -> 0.4300 | diff 0.0450; CI [-0.0050, 0.0950]; McNemar p 0.1221 | CAUTION |
| Residual parcel38 ROI signal | shifted-null control | ROI rank 0.6207 vs shifted 0.4818 | gap 0.1388 | SOLID for non-random ROI signal |
| V-JEPA-only target-space retrieval scales but remains below frozen ATM->CLIP | 16540 train images, 200-image test | V-JEPA top1 0.3750 vs frozen CLIP top1 0.5200 | gap -0.1450 | SOLID against "V-JEPA alone is enough" |
| V-JEPA->CLIP is not a CLIP replacement | 16540 train images, 200-image test | CLIP top1 0.1450 vs frozen 0.5200 | large negative gap | SOLID |
| Frozen CLIP + V-JEPA->CLIP blend | bootstrap paired differences | 8192 top1 +0.0350 CI [-0.0100, 0.0800]; 16540 top1 +0.0150 CI [-0.0050, 0.0400] | small, CI crosses 0 | CAUTION |
| CLIP+V-JEPA concat target-space retrieval | 16540 train images | combo top1 0.4000, top5 0.7250, rank 0.9566 | useful but below frozen CLIP top1/rank | CAUTION |
| Raw parcel38 query identity | query-target correlation, shuffled query order | diag-offdiag 0.4136 vs shuffled mean -0.0034 | diag rank pct 0.8883 | SOLID |
| Residual parcel38 query identity | query-target correlation, shuffled query order | diag-offdiag 0.1615 vs shuffled mean -0.0009 | target-geometry corr 0.6681 | SOLID but weaker amplitude |
| V-JEPA residual spatial prototypes | shifted-null bootstrap | V-JEPA latent32 surface gap 0.0648 CI [0.0070, 0.1209] p 0.0252 | positive residual signal | CAUTION/SOLID |
| CLIP+V-JEPA residual spatial prototypes | shifted-null bootstrap | latent32 surface gap 0.0656 CI [0.0080, 0.1225] p 0.0254 | positive residual signal | CAUTION/SOLID |

### Cortical Structure Check

| Probe | Raw parcel38 strong | Residual parcel38 | Interpretation |
|---|---:|---:|---|
| Query-target binding | diag-offdiag 0.4136 | diag-offdiag 0.1615 | raw is visibly stronger; residual remains above shuffled null |
| Target geometry preservation | 0.5153 | 0.6681 | residual preserves target-space geometry surprisingly well |
| Early calcarine timing | best keep w100_200; drop w100_200 | best keep w200_300; drop w200_300 | early visual structure is present in both |
| High-level timing | best keep w600_700; drop w300_400 | best keep w200_300; drop w400_500 | high-level staging is clearer in raw than residual |
| Posterior EEG channel concentration | posterior/all 2.9140; top Oz | posterior/all 3.0530; top Oz | O/PO/P dominance matches visual EEG, but is not proof of ROI-specific channel selection |

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
