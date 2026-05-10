# ATM ROI Spatial Branch, 8192-Image Budget

Date: 2026-05-10

## Setup

- Train images: 8192 THING-EEG training images.
- EEG samples: 8192 images x 10 subjects x 4 repeats = 327,680 training samples.
- Test set: 200 THING-EEG test images, averaged over 10 subjects.
- Backbone: ATM/iTransformer channel tokens, `d_model=256`, `heads=4`, `layers=1`.
- Subject token: disabled, `subject_mode=none`.
- Semantic head: original shallow/ATMS-style head.
- Spatial branch: ordered ROI-query branch supervised by TRIBE-derived visual ROI pseudo-targets.
- TRIBE target: 4096 existing fp32 targets + 4096 new fp16 fast-still tail targets.

## Main Metrics

Final epoch metrics:

| budget | model | CLIP top1 | CLIP top5 | CLIP rank | ROI rank | shifted ROI rank |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | semantic-only | 0.250 | 0.560 | 0.9502 | - | - |
| 4096 | spatial-group12 | 0.225 | 0.525 | 0.9478 | 0.6876 | 0.5195 |
| 4096 | spatial-parcel38 | 0.210 | 0.575 | 0.9486 | 0.7715 | 0.5026 |
| 8192 | semantic-only | 0.350 | 0.680 | 0.9700 | - | - |
| 8192 | spatial-group12 | 0.335 | 0.695 | 0.9692 | 0.7108 | 0.5193 |
| 8192 | spatial-parcel38 | 0.310 | 0.645 | 0.9658 | 0.7816 | 0.4984 |

Best CLIP top1 epoch:

| model | best epoch | CLIP top1 | CLIP top5 | CLIP rank |
|---|---:|---:|---:|---:|
| semantic-only | 20 | 0.350 | 0.680 | 0.9700 |
| spatial-group12 | 17 | 0.335 | 0.665 | 0.9669 |
| spatial-parcel38 | 18 | 0.335 | 0.650 | 0.9655 |

## Late Fusion / Reranking

Heldout split results using ROI score as a post-hoc reranker:

| source | alpha | topK | base top1 | fused top1 | base top5 | fused top5 | rescue | damage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| group own semantic + group ROI | 0.20 | all | 0.300 | 0.310 | 0.680 | 0.670 | 5 | 4 |
| semantic-only + group ROI | 0.05 | 100 | 0.310 | 0.310 | 0.690 | 0.700 | 0 | 0 |
| parcel own semantic + parcel ROI | 0.10 | all | 0.250 | 0.260 | 0.630 | 0.650 | 1 | 0 |
| semantic-only + parcel ROI | 0.05 | 100 | 0.310 | 0.320 | 0.690 | 0.700 | 1 | 0 |

## Interpretation

8192 clearly improves the semantic-only baseline over 4096. The spatial branch
continues to learn TRIBE-derived ROI pseudo-targets, especially parcel38, but it
does not beat the stronger semantic-only CLIP retrieval baseline when trained
with the current fixed loss weights.

The most useful positive signal is not raw semantic top1. It is:

- ROI prediction remains above shifted-null at 8192.
- late fusion gives a small heldout rerank improvement for `semantic-only +
  parcel ROI` from top1 0.310 to 0.320 and top5 0.690 to 0.700.
- parcel38 has stronger ROI retrieval than group12, but also competes more with
  CLIP semantic training.

Conservative conclusion:

> Scaling to 8192 helps EEG-to-CLIP retrieval substantially. TRIBE ROI
> supervision learns a distinct pseudo-cortical signal, but the current
> multi-task training does not yet convert that signal into a robust
> end-to-end semantic retrieval gain. ROI supervision is currently more useful
> as an interpretable auxiliary branch and weak reranking cue than as a direct
> replacement for semantic-only training.

## Artifacts

- TRIBE target:
  - `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192.npz`
- Visual ROI target:
  - `fmri_foundation_workspace/results/eeg_image_bridge/visual_roi_targets/visual_roi_targets_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192.npz`
- Model runs:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_semantic_group_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192_d256_none/`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_group_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192_d256_none/`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192_d256_none/`
- Late fusion:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_late_fusion/n8192_reuse4096_clean256_shallow/`

## Next

The next most informative check is to repeat the temporal hierarchy ablation on
the 8192 group12/parcel38 models. If the early-ROI/late-ROI timing structure
becomes stronger with the larger budget, that is a better mechanism story than
expecting the spatial branch to always improve CLIP top1.
