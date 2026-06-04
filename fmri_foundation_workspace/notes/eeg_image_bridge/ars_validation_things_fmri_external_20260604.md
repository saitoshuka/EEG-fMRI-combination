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
| Raw EEG signal has plausible temporal/channel structure | best 100 ms keep-window 300-400 ms rank 0.6164; posterior P/PO/O-only rank 0.6744 vs nonposterior-only 0.6112; top single channel Oz rank 0.6242 | Supports a visual-evoked interpretation and motivates channel-aware trainable models. |
| ROI-family timing shows a limited visual hierarchy trend | early_visual best keep 100-200 ms; mid/ventral/all-visual best keep 300-400 ms | Useful neuroscience-facing evidence, but not a clean feed-forward latency cascade. |
| Trainable factorized-query model predicts real visual fMRI | posterior factorized query rank 0.6604 vs shifted 0.4764; full-ridge reference 0.6456; posterior-ridge ceiling 0.6744 | A promising trainable ordered-query result; still below the strongest closed-form posterior ridge. |
| Trainable factorized-query is stable over heldout seeds | 4-seed rank mean 0.6461, shifted mean 0.4937, full-ridge mean 0.6399 | Stable above null, modestly above full-ridge on average, but not a large performance jump. |

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
- A temporal/channel ablation shows plausible visual EEG structure: 300-400 ms is the strongest 100 ms keep-window, and posterior P/PO/O channels outperform full EEG.
- ROI-family time analysis shows early_visual peaking earlier (100-200 ms) than mid/ventral/all-visual families (300-400 ms), supporting a modest visual hierarchy story.
- A small trainable factorized-query readout now beats full-channel ridge and MLP/linear trainable baselines on the seed-33 heldout split.
- The factorized-query readout remains above shifted/shuffled controls across four heldout seeds, with a small mean improvement over full-channel ridge.
- CLIP/V-JEPA controls were run, preventing an overclaim that TRIBE is uniquely fMRI-like.

Fails or incomplete:

- The proposed ROI-query deep architecture does not yet turn the available EEG signal into a robust heldout EEG-to-real-fMRI result.
- The strongest EEG result remains the closed-form posterior ridge ceiling; the trainable factorized-query model is multi-seed stable but its improvement over full ridge is modest.
- The original 77-image exact THINGS-EEG test external validation is too small and has low subject-pattern reliability.
- No final image retrieval or generation improvement has been shown under the new real-fMRI-aligned supervision.
- ROI family definitions are curated and must be justified or replaced by atlas-defined visual masks in the final paper.

## Required Next Experiments

1. Run temporal/channel/ROI-family weight and ablation analysis for the best trainable model across seeds if this becomes a central claim.
2. Repeat image-feature ceilings over the same multi-seed splits or freeze seed 33 as the final protocol with a clear rationale.
3. Test whether the factorized-query cortical branch improves or preserves THINGS image retrieval/generation when added to the existing semantic EEG route.
4. Freeze the split/protocol, then run a final statistical correction or clearly label ROI-family tests as exploratory.
5. Only after the strict overlap protocol is stable, reconnect the best cortical branch to visual decoding/generation and measure whether it improves or at least preserves retrieval while adding cortical interpretability.

## Claim Status

Supported now:

- "Stimulus-derived cortical teachers align with real image-evoked fMRI, especially in visual ROIs."
- "Averaged visual EEG contains a measurable signal for predicting real fMRI visual-family patterns under image-heldout evaluation."
- "The raw EEG signal has plausible visual temporal/channel structure under the overlap heldout protocol."
- "A trainable ordered-query readout can extract this signal across multiple image-heldout splits, with modest average improvement over full-channel ridge and strong improvement over shifted/shuffled controls."

Not supported yet:

- "The current ROI-query deep model robustly distills fMRI spatial knowledge into EEG."
- "The method is ready as an AAAI main-conference contribution."
- "The approach improves image generation or retrieval under strict final evaluation."
