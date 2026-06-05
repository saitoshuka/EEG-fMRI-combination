# Proto256 pooled residual to real-fMRI probe: interim multiseed result

Date: 2026-06-05

This note records the interim status after completing seed 11 for the pooled
proto256 residual ATM branch. Seed 77 is still running, so this is not the final
multiseed table.

## Setup

- Model family: ATM semantic branch plus pooled proto256 residual ROI branch.
- Training target: CLIP+V-JEPA2 residualized proto256 cortical prototype target.
- Completed seeds in this note: 33 and 11.
- Current running seed: 77.
- Feature probe: train-only ridge from exported EEG features to measured
  THINGS-fMRI ROI targets.
- Train overlap: 6330 images.
- Test overlap: 77 images.
- Metrics below are image retrieval-style rank percentile against measured
  fMRI targets, with shifted rank as the null.

Artifacts:

- Multiseed summary:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/pooled_proto256_residual_realfmri_multiseed.md`
- Seed 11 visual64 probe:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_pooled_residual_seed11_bestroi_visual64/summary.json`
- Seed 11 shared207 probe:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_pooled_residual_seed11_bestroi_shared207/summary.json`

## Seed 11 probe result

### visual64

| feature | rank | shifted | delta | top1 | ROI corr | image corr |
|---|---:|---:|---:|---:|---:|---:|
| semantic | 0.5902 | 0.4793 | 0.1109 | 0.0649 | 0.2265 | 0.2294 |
| roi | 0.6391 | 0.4901 | 0.1490 | 0.0649 | 0.1491 | 0.1809 |
| semantic_roi | 0.5868 | 0.4802 | 0.1066 | 0.0649 | 0.2209 | 0.2304 |
| image_clip oracle | 0.6765 | 0.4892 | 0.1873 | 0.0649 | 0.2210 | 0.2576 |

### shared207

| feature | rank | shifted | delta | top1 | ROI corr | image corr |
|---|---:|---:|---:|---:|---:|---:|
| semantic | 0.5496 | 0.4944 | 0.0552 | 0.0390 | 0.0918 | 0.1044 |
| roi | 0.5964 | 0.5149 | 0.0815 | 0.0260 | 0.0333 | 0.0723 |
| semantic_roi | 0.5490 | 0.4954 | 0.0537 | 0.0390 | 0.0837 | 0.1033 |
| image_clip oracle | 0.6224 | 0.4954 | 0.1270 | 0.0909 | 0.1418 | 0.1207 |

## Interim aggregate across seeds 11 and 33

### visual64

| feature | rank mean | rank std | delta mean | delta std | top1 mean | ROI corr mean |
|---|---:|---:|---:|---:|---:|---:|
| semantic | 0.5953 | 0.0071 | 0.1161 | 0.0074 | 0.0584 | 0.2314 |
| roi | 0.6402 | 0.0016 | 0.1553 | 0.0089 | 0.0649 | 0.1668 |
| semantic_roi | 0.5916 | 0.0068 | 0.1117 | 0.0071 | 0.0584 | 0.2252 |
| image_clip oracle | 0.6765 | 0.0000 | 0.1873 | 0.0000 | 0.0649 | 0.2210 |

### shared207

| feature | rank mean | rank std | delta mean | delta std | top1 mean | ROI corr mean |
|---|---:|---:|---:|---:|---:|---:|
| semantic | 0.5534 | 0.0054 | 0.0577 | 0.0035 | 0.0455 | 0.0869 |
| roi | 0.6027 | 0.0089 | 0.0903 | 0.0124 | 0.0195 | 0.0570 |
| semantic_roi | 0.5516 | 0.0036 | 0.0549 | 0.0018 | 0.0390 | 0.0765 |
| image_clip oracle | 0.6224 | 0.0000 | 0.1270 | 0.0000 | 0.0909 | 0.1418 |

## Interpretation

This is a useful result for the main story, but it is not yet final.

The positive part is that the pooled proto256 residual ROI feature transfers to
measured THINGS-fMRI better than the semantic EEG embedding on rank percentile
for both visual64 and shared207 targets. The effect is stable across the two
completed seeds:

- visual64 rank: ROI 0.6402 vs semantic 0.5953.
- shared207 rank: ROI 0.6027 vs semantic 0.5534.
- shifted-null stays near chance.

This supports the claim that the cortical/prototype branch is not just a pretty
visualization; it carries stimulus-linked information that maps to measured
fMRI better than semantic-only features under a train-only ridge probe.

The limitation is that ROI-wise correlation does not uniformly improve. In
visual64, semantic features still have higher ROI corr than the ROI feature.
So the strongest claim should use retrieval-style fMRI pattern rank, not
voxel/ROI-wise correlation alone. Also, simple semantic+ROI concatenation is
not helping yet, which means fusion remains unsolved.

## Decision

Continue the seed 77 run. If seed 77 agrees, this becomes a credible
three-seed external-validation result:

"A stimulus-mediated cortical branch learns EEG features whose predicted
cortical-prototype representation transfers more strongly to measured visual
fMRI patterns than a semantic-only EEG embedding."

This would still need to be paired with retrieval/generation or stronger
query-specific interpretability before making a main-conference-level claim.
