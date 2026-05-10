# CLIP-Residualized ROI Scaling: 4096 vs 8192

Date: 2026-05-10

## Question

Does CLIP-residualized ROI prediction improve when increasing the training
budget from 4096 to 8192 images?

## Setup

- Target: TRIBE visual ROI targets residualized against ViT-H/14 image CLIP.
- Backbone: ATM/iTransformer, `d_model=256`, no subject token.
- Loss: same semantic CLIP loss plus residual ROI losses.
- Group12:
  - 20 epochs, batch size 128.
- Parcel38:
  - 10 epochs, batch size 256 quick check.
- Test: 200 THING-EEG test images.

## Residual Target Diagnostics

Before training, image CLIP still explains a large part of raw ROI target:

| train budget | ROI kind | image CLIP -> raw ROI rank | shifted rank | ROI-wise corr | R2 |
|---:|---|---:|---:|---:|---:|
| 4096 | group12 | 0.781 | 0.514 | 0.676 | 0.434 |
| 4096 | parcel38 | 0.839 | 0.511 | 0.693 | 0.461 |
| 8192 | group12 | 0.772 | 0.515 | 0.691 | 0.458 |
| 8192 | parcel38 | 0.837 | 0.508 | 0.713 | 0.492 |

Semantic-only EEG probe on CLIP-residualized ROI is weak:

| train budget | ROI kind | semantic probe residual rank | shifted rank | ROI-wise corr |
|---:|---|---:|---:|---:|
| 4096 | group12 | 0.561 | 0.504 | 0.206 |
| 4096 | parcel38 | 0.566 | 0.498 | 0.202 |
| 8192 | group12 | 0.540 | 0.499 | 0.253 |
| 8192 | parcel38 | 0.561 | 0.488 | 0.269 |

## Residual-Trained ROI-Query Results

| train budget | ROI kind | best epoch | best ROI rank | shifted rank | final ROI rank | final shifted rank |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | group12 | 15 | 0.614 | 0.519 | 0.610 | 0.515 |
| 8192 | group12 | 11 | 0.631 | 0.513 | 0.614 | 0.514 |
| 4096 | parcel38 | 7 | 0.618 | 0.502 | 0.614 | 0.498 |
| 8192 | parcel38 | 8 | 0.639 | 0.501 | 0.629 | 0.517 |

## Interpretation

The residual ROI rank does increase from 4096 to 8192, but only modestly:

- group12 best rank: `0.614 -> 0.631` (`+0.016`)
- parcel38 best rank: `0.618 -> 0.639` (`+0.021`)

This suggests that extra data helps a little, but the residual spatial signal is
still weak. The improvement is not large enough to justify blindly scaling to
60k images without first checking 16k.

Conservative conclusion:

> CLIP-residualized ROI prediction shows a weak positive scaling trend from
> 4096 to 8192 images, but the signal remains much weaker than raw ROI
> prediction. This supports a staged scaling test at 16k, not immediate full
> 60k training.

## Artifacts

- 4096 residual target/diagnostics:
  - `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_ridge100_n4096/`
- 4096 residual group12 model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_group_clip_residual_train_seed33_budget4096_n4096_d256_none/`
- 4096 residual parcel38 quick model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_clip_residual_train_seed33_budget4096_n4096_d256_none_fast10/`
- 8192 residual target/diagnostics:
  - `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_ridge100_n8192/`
- 8192 residual group12 model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_group_clip_residual_train_seed33_budget8192_n8192_d256_none/`
- 8192 residual parcel38 quick model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_clip_residual_train_seed33_budget8192_n8192_d256_none_fast10/`

## Next

Run a 16k residual target and short residual ROI-query training only if the
compute budget is acceptable. The decision rule should be:

- continue scaling if 16k residual rank reaches roughly `0.67+`;
- stop prioritizing residual spatial distillation if it stays around `0.63-0.64`.
