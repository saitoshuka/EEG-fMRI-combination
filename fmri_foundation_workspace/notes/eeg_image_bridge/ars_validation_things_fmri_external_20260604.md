# ARS Validation: THINGS-fMRI External Validation

## Material Passport

- Passport ID: `things_fmri_external_validation_20260604`
- Mode: ARS experiment-agent `validate` plus reviewer-style gate check
- Verification Status: `ANALYZED`
- Data access level: local derived results; no external upload
- Primary result report: `fmri_foundation_workspace/notes/eeg_image_bridge/things_fmri_external_validation_results_20260604.md`
- Primary result directory: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation`
- Reproducibility artifacts:
  - `scripts/build_things_fmri_overlap.py`
  - `scripts/extract_things_fmri_roi_betas.py`
  - `scripts/evaluate_things_fmri_external_validation.py`
  - `scripts/analyze_things_fmri_external_roi_breakdown.py`
  - `scripts/evaluate_raw_eeg_to_realfmri_overlap_holdout.py`
  - `scripts/evaluate_image_features_to_realfmri_overlap_holdout.py`

## Statistical Findings

| Finding | Evidence | Interpretation |
|---|---:|---|
| Raw TRIBE/parcel38 teacher aligns with real THINGS-fMRI207 on 77 exact heldout test images | rank 0.6528 vs shifted 0.4880, p=0.0002 | Strong support that the pseudo-cortical teacher captures stimulus-driven real fMRI structure. |
| CLIP and V-JEPA also align with real fMRI | CLIP rank 0.6224, p=0.0004; V-JEPA rank 0.5668, p=0.0222 | TRIBE is not the only visual representation aligned with fMRI. The claim should be comparative, not exclusive. |
| Current ROI-query deep model predictions are weak on 77 exact test images | EEG raw-strong rank 0.5113, p=0.1038; EEG residual rank 0.5436, p=0.0836 | Not sufficient for a main EEG-to-real-fMRI claim. |
| Larger 1000-image overlap probe shows EEG contains usable real-fMRI signal | raw EEG waveform visual-family rank 0.6456, p=0.0010 | Strong evidence against the hypothesis that visual EEG is pure noise for fMRI-like targets. |
| Fit-target shuffle control fails | visual-family rank 0.4933, p=0.7632 | Supports that the raw EEG heldout result is not a trivial metric or split artifact. |
| Signal is visual-family concentrated | raw EEG visual-family rank 0.6456 vs nonvisual rank 0.5165 | Supports a visual neuroscience interpretation rather than an all-brain/global shortcut. |
| Raw EEG overlap finding is stable over heldout seeds | 4-seed visual-family rank mean 0.6399, std 0.0148; shuffle rank mean 0.4976 | Strengthens the result beyond one favorable split. |

## Fallacy Scan

Coverage: 11/11 ARS statistical fallacy classes checked.

| Fallacy | Status | Note |
|---|---|---|
| Simpson's paradox | NOTE | No subgroup reversal observed yet, but subject-level and ROI-family stratification should remain mandatory. |
| Ecological fallacy | CAUTION | Many metrics use subject-averaged fMRI and averaged EEG; avoid claims about individual-subject neural mechanisms. |
| Berkson's paradox | CAUTION | Exact-image overlap is a selected subset of THINGS images; results may not generalize to all image categories. |
| Collider bias | NOTE | No explicit covariate control creating collider risk in current ridge evaluations. |
| Base-rate neglect | NOTE | Not a diagnostic/probability task. |
| Regression to the mean | NOTE | No extreme-group pre/post design. |
| Survivorship bias | NOTE | No participant attrition analysis, but fMRI has only 3 subjects; report this limitation. |
| Look-elsewhere effect | CAUTION | Multiple ROI families/predictors are explored. Treat family analyses as structured exploratory unless corrected. |
| Garden of forking paths | CAUTION | Several design choices exist: ROI family definitions, time pooling, ridge alpha, holdout split. Need preregistered final protocol or robustness across seeds. |
| Correlation is not causation | CAUTION | Use associational language: "predicts", "aligns with", "captures stimulus-driven structure"; avoid causal distillation claims unless intervention/ablation supports them. |
| Reverse causality | NOTE | Not a causal temporal design; no reverse-causality claim should be made. |

## Main-Conference Gate

Current gate decision: `NOT READY`, but the direction became materially stronger.

Passes:

- External real fMRI validation exists and is reproducible from OpenNeuro ds004192 ICA beta derivatives.
- Pseudo-cortical raw teacher is validated against real fMRI and concentrated in visual ROIs.
- Raw EEG waveform baseline predicts real fMRI visual-family targets on a 1000-image heldout split, with a fit-target shuffle control near chance.
- The raw EEG heldout effect is stable across four random seeds.
- CLIP/V-JEPA controls were run, preventing an overclaim that TRIBE is uniquely fMRI-like.

Fails or incomplete:

- The proposed ROI-query deep architecture does not yet turn the available EEG signal into a robust heldout EEG-to-real-fMRI result.
- The strongest EEG result is currently a ridge baseline on image-averaged EEG, not the final trainable cortical distillation model.
- The original 77-image exact THINGS-EEG test external validation is too small and has low subject-pattern reliability.
- No final image retrieval or generation improvement has been shown under the new real-fMRI-aligned supervision.
- ROI family definitions are curated and must be justified or replaced by atlas-defined visual masks in the final paper.

## Required Next Experiments

1. Run temporal/channel ablations for raw EEG -> real fMRI visual-family prediction to show the signal comes from plausible post-stimulus windows and posterior channels.
2. Train a small raw-waveform neural model on the same overlap split and test whether ROI-query/cortical supervision beats ridge or at least matches it with interpretable time/channel/ROI structure.
3. Repeat image-feature ceilings over the same multi-seed splits or freeze seed 33 as the final protocol with a clear rationale.
4. Freeze the split/protocol, then run a final statistical correction or clearly label ROI-family tests as exploratory.
5. Only after the strict overlap protocol is stable, reconnect the best cortical branch to visual decoding/generation and measure whether it improves or at least preserves retrieval while adding cortical interpretability.

## Claim Status

Supported now:

- "Stimulus-derived cortical teachers align with real image-evoked fMRI, especially in visual ROIs."
- "Averaged visual EEG contains a measurable signal for predicting real fMRI visual-family patterns under image-heldout evaluation."

Not supported yet:

- "The current ROI-query deep model robustly distills fMRI spatial knowledge into EEG."
- "The method is ready as an AAAI main-conference contribution."
- "The approach improves image generation or retrieval under strict final evaluation."
