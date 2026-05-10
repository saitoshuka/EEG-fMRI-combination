# ATM ROI Late-Fusion Reranking, 4096 Images

Date: 2026-05-10

## Question

Can the ATM spatial ROI branch convert TRIBE visual-ROI pseudo targets into an
image retrieval gain, instead of only improving ROI retrieval metrics?

## Setup

- Semantic baseline: `atm_semantic_group_train_seed33_budget4096_faststill_n4096_d256_none`
- Spatial models:
  - `atm_spatial_group_train_seed33_budget4096_faststill_n4096_d256_none`
  - `atm_spatial_parcel_train_seed33_budget4096_faststill_n4096_d256_none`
- Test set: 200 THING-EEG test images, averaged across 10 subjects.
- Spatial score:
  - `corr/cosine(EEG_pred_ROI_i, TRIBE_ROI_image_j)`
- Fusion score:
  - `zscore(semantic_score_i) + alpha * zscore(spatial_score_i)`
- Reranking:
  - either all candidates or only semantic top-K.
- Tuning rule:
  - even-index test rows used as validation for `alpha/topK`
  - odd-index test rows held out for reporting

## Main Heldout Result

| source | selected alpha/topK | base top1 | fused top1 | base top5 | fused top5 | base rank | fused rank | rescue | damage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group own semantic + group ROI | 0.05 / 150 | 0.220 | 0.220 | 0.510 | 0.530 | 0.9508 | 0.9516 | 0 | 0 |
| semantic-only + group ROI | 0.00 / 150 | 0.240 | 0.240 | 0.570 | 0.570 | 0.9548 | 0.9548 | 0 | 0 |
| parcel own semantic + parcel ROI | 0.05 / 5 | 0.180 | 0.190 | 0.560 | 0.560 | 0.9459 | 0.9460 | 2 | 1 |
| semantic-only + parcel ROI | 0.00 / 150 | 0.240 | 0.240 | 0.570 | 0.570 | 0.9548 | 0.9548 | 0 | 0 |

## Full-Test Exploratory Sweep

The best full-test parcel own-semantic fusion improved its own weaker semantic
head:

- `parcel_own_semantic_plus_roi`
- alpha/topK: `0.10 / all`
- top1: `0.210 -> 0.225`
- top5: `0.575 -> 0.575`
- rank: `0.9486 -> 0.9500`
- rescue/damage: `5 / 2`

But spatial ROI did not robustly improve the stronger standalone semantic-only
baseline:

- `semantic_only_plus_parcel_roi`
- best positive-alpha full-test rank case:
  - alpha/topK: `0.08 / all`
  - top1: `0.250 -> 0.245`
  - top5: `0.560 -> 0.565`
  - rank: `0.9502 -> 0.9516`
  - rescue/damage: `0 / 1`

There was one full-test group-ROI case with top1 `0.250 -> 0.255` using a
negative alpha, but top5/rank worsened and validation did not select it. This is
not convincing evidence of useful spatial gain.

## Interpretation

The spatial ROI score is real enough to help the weaker jointly-trained spatial
model slightly. However, it is not yet adding reliable incremental information
on top of the stronger semantic-only EEG-to-CLIP model.

Current evidence supports:

> The ROI branch learns a TRIBE pseudo-cortical target and can slightly rerank
> its own semantic output.

Current evidence does not yet support:

> TRIBE ROI supervision improves the best image retrieval model.

## Artifacts

- Sweep CSV:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/late_fusion_sweep.csv`
- Heldout CSV:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/late_fusion_heldout.csv`
- Score matrices:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/late_fusion_scores.npz`
- Visualizations:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/semantic_only_score.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/group_spatial_score.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/parcel_spatial_score.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n4096_clean256_fine_alpha/heldout_rescue_damage.png`

## Next Step

Do not claim spatial retrieval improvement from this 4096 run. The clean next
experiment is to train the new attention semantic head and spatial branch under
the same budget, then rerun this exact late-fusion evaluator. If attention
semantic pooling closes the semantic gap while preserving ROI signal, spatial
late fusion may become useful.
