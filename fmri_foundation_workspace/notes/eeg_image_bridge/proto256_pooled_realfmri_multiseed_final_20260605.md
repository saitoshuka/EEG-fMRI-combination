# Proto256 pooled residual to real-fMRI probe: final three-seed result

Date: 2026-06-05

This note replaces the interim two-seed note for the pooled proto256 residual
ATM branch. Seeds 11, 33, and 77 have all completed training, export, and
THINGS-fMRI external validation.

## Setup

- Model family: ATM semantic branch plus pooled proto256 residual ROI branch.
- Training target: CLIP+V-JEPA2 residualized proto256 cortical prototype target.
- Completed seeds: 11, 33, 77.
- Feature probe: train-only ridge from exported EEG features to measured
  THINGS-fMRI ROI targets.
- Train overlap: 6330 images.
- Test overlap: 77 images.
- Metrics: retrieval-style rank percentile against measured fMRI targets, with
  shifted rank as the null.

Artifacts:

- Multiseed summary:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/pooled_proto256_residual_realfmri_multiseed.md`
- Aggregate CSV:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/pooled_proto256_residual_realfmri_multiseed_aggregate.csv`
- Per-run CSV:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/pooled_proto256_residual_realfmri_multiseed_per_run.csv`

## Three-seed aggregate

### visual64

| feature | rank mean | rank std | shifted mean | delta mean | delta std | top1 mean | ROI corr mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic | 0.6005 | 0.0103 | 0.4809 | 0.1196 | 0.0080 | 0.0606 | 0.2308 |
| roi | 0.6202 | 0.0346 | 0.4827 | 0.1375 | 0.0315 | 0.0563 | 0.1582 |
| semantic_roi | 0.5968 | 0.0103 | 0.4813 | 0.1156 | 0.0084 | 0.0606 | 0.2276 |
| image_clip oracle | 0.6765 | 0.0000 | 0.4892 | 0.1873 | 0.0000 | 0.0649 | 0.2210 |

Per-seed rank:

| seed | semantic | roi | semantic_roi |
|---:|---:|---:|---:|
| 11 | 0.5902 | 0.6391 | 0.5868 |
| 33 | 0.6003 | 0.6413 | 0.5964 |
| 77 | 0.6109 | 0.5803 | 0.6073 |

### shared207

| feature | rank mean | rank std | shifted mean | delta mean | delta std | top1 mean | ROI corr mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic | 0.5587 | 0.0099 | 0.4993 | 0.0594 | 0.0039 | 0.0519 | 0.0919 |
| roi | 0.5995 | 0.0084 | 0.5110 | 0.0885 | 0.0093 | 0.0346 | 0.0466 |
| semantic_roi | 0.5559 | 0.0079 | 0.5002 | 0.0557 | 0.0018 | 0.0476 | 0.0830 |
| image_clip oracle | 0.6224 | 0.0000 | 0.4954 | 0.1270 | 0.0000 | 0.0909 | 0.1418 |

Per-seed rank:

| seed | semantic | roi | semantic_roi |
|---:|---:|---:|---:|
| 11 | 0.5496 | 0.5964 | 0.5490 |
| 33 | 0.5572 | 0.6090 | 0.5542 |
| 77 | 0.5692 | 0.5931 | 0.5646 |

## Interpretation

The strongest positive result is shared207. Across all three seeds, the pooled
proto256 residual ROI feature maps to measured fMRI patterns better than the
semantic EEG embedding:

- shared207 rank: ROI 0.5995 +- 0.0084 vs semantic 0.5587 +- 0.0099.
- The shifted null is near chance.
- The effect holds for seeds 11, 33, and 77 individually.

The visual64 result is weaker but still useful. The mean rank is higher for the
ROI feature than semantic-only:

- visual64 rank: ROI 0.6202 vs semantic 0.6005.

However, the visual64 effect is not seed-stable in the strict sense. Seeds 11
and 33 show ROI > semantic, but seed 77 reverses the order. This means visual64
should be described as a trend or partial support, not as a robust three-seed
win.

The result does not support a broad claim that the ROI branch improves all fMRI
metrics. ROI-wise correlation is lower for the ROI feature than for semantic
features on both target spaces. The claim should be specifically about
retrieval-style fMRI pattern alignment / rank, not about ROI-wise correlation.

Simple feature concatenation does not help. The semantic+ROI ridge feature is
not better than semantic-only, suggesting that fusion should be learned inside
the model or with a better regularized adapter, not post-hoc concatenation.

## ARS-style validation status

Verification Status: ANALYZED.

The result was not reproduced by rerunning the full three-seed queue from
scratch in this validation step; instead, it was analyzed from completed
training/export/probe artifacts. The evidence is sufficient for a project-level
decision but should not yet be called a final paper claim without additional
checks.

Fallacy scan:

- Metric cherry-picking risk: present. Rank supports the ROI branch more than
  ROI-wise correlation. The paper should report both.
- Seed instability risk: present for visual64, weaker for shared207.
- Overclaiming risk: present. The supported claim is external fMRI pattern-rank
  transfer, not general decoding superiority.
- Leakage risk: mitigated by train-only ridge and shifted-null comparison, but
  the split/probe code should still be audit-reviewed before paper submission.
- Multiple comparison risk: present because several target spaces/features were
  checked. Treat this as validation evidence, not definitive hypothesis testing.

## Decision

This is a meaningful positive result for the cortical-supervision story:

"A stimulus-mediated cortical-prototype branch learns EEG features whose
representation transfers to measured fMRI pattern retrieval better than a
semantic-only EEG embedding, robustly on shared visual-cortical targets and
partially on visual64 targets."

This is not yet enough by itself for a main-conference performance claim. For a
stronger AAAI-style story, the next high-value work should be:

1. Audit the train-only ridge/probe pipeline for leakage and split correctness.
2. Add query-specific interpretability or prototype/cortical map analysis for
   this pooled proto256 model family.
3. Improve fusion, since post-hoc semantic+ROI concatenation currently fails.
4. Test whether the same ROI/prototype signal gives a downstream retrieval or
   reranking gain under a fair semantic baseline.
