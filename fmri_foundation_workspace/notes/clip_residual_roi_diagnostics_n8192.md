# CLIP-Residualized ROI Diagnostics, 8192 Images

Date: 2026-05-10

## Question

Are the TRIBE visual ROI pseudo-targets mostly predictable from image semantics,
or do they contain ROI-specific residual structure that EEG can still learn
after removing the image-CLIP component?

## Method

Residualization:

```text
TRIBE ROI target = image-CLIP-predictable component + residual
```

- Input for residualization: ViT-H/14 image CLIP features.
- Ridge fit: train 8192 images.
- Test: 200 THING-EEG test images.
- Residual targets are z-scored using train residual mean/std.

Then evaluate:

- image CLIP -> raw ROI target
- semantic-only EEG embedding -> CLIP-residualized ROI target
- existing raw-trained ROI-query branch -> CLIP-residualized ROI target
- newly trained ROI-query branch -> CLIP-residualized ROI target

## Main Diagnostics

Image CLIP explains a large part of raw TRIBE ROI targets:

| source | ROI kind | rank | shifted rank | ROI-wise corr | R2 |
|---|---|---:|---:|---:|---:|
| image CLIP -> raw ROI | group12 | 0.772 | 0.515 | 0.691 | 0.458 |
| image CLIP -> raw ROI | parcel38 | 0.837 | 0.508 | 0.713 | 0.492 |

After removing the image-CLIP component, EEG-to-residual ROI signal is much
weaker:

| source | ROI kind | rank | shifted rank | ROI-wise corr | R2 |
|---|---|---:|---:|---:|---:|
| semantic EEG probe -> residual ROI | group12 | 0.540 | 0.499 | 0.253 | -0.993 |
| semantic EEG probe -> residual ROI | parcel38 | 0.561 | 0.488 | 0.269 | -0.834 |
| existing ROI-query -> residual ROI | group12 | 0.532 | 0.506 | 0.130 | -54.777 |
| existing ROI-query -> residual ROI | parcel38 | 0.530 | 0.504 | 0.134 | -10.171 |

## Residual-Trained ROI-Query Models

Training directly on CLIP-residualized ROI targets improves over the frozen/raw
ROI-query predictions, but the signal remains weak.

| target | setting | best ROI rank | shifted rank | final ROI rank | final shifted rank |
|---|---|---:|---:|---:|---:|
| group12 residual | 20 epochs, batch 128 | 0.631 | 0.513 | 0.614 | 0.514 |
| parcel38 residual | 10 epochs, batch 256 | 0.639 | 0.501 | 0.629 | 0.517 |

For comparison, the raw TRIBE ROI targets were much easier:

| target | raw ROI rank | shifted rank |
|---|---:|---:|
| group12 raw | 0.711 | 0.519 |
| parcel38 raw | 0.782 | 0.498 |

## Interpretation

This is a strong diagnostic result:

- TRIBE visual ROI pseudo-targets contain substantial image-semantic shared
  variance.
- A semantic-only EEG embedding can recover much of the raw pseudo-ROI target
  because the target is strongly stimulus-conditioned.
- Once the image-CLIP-predictable component is removed, the remaining residual
  ROI signal is weak and only modestly above shifted/null.
- Current ROI-query spatial supervision therefore cannot yet support a strong
  claim that EEG learns detailed fMRI-like spatial knowledge beyond image
  semantics.

Conservative claim:

> TRIBE pseudo-cortical targets provide useful anatomical labels and auxiliary
> interpretability, but most of the current learnable ROI signal is shared with
> image semantics. CLIP-residualized ROI targets expose only a weak residual EEG
> signal under the current ATM/ROI-query setup.

## Artifacts

- Residual diagnostics script:
  - `fmri_foundation_workspace/scripts/evaluate_clip_residual_roi_signal.py`
- Residual target/results directory:
  - `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_ridge100_n8192/`
- Residual-trained group12 model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_group_clip_residual_train_seed33_budget8192_n8192_d256_none/`
- Residual-trained parcel38 quick model:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_clip_residual_train_seed33_budget8192_n8192_d256_none_fast10/`

## Practical Note

The full 8192 training set expands to 327,680 EEG trials because it includes
10 subjects and 4 repeats per image. With 20 epochs and batch 128, this is about
6.55 million trial-level forward/backward passes, so one residual branch can
take tens of minutes on the local GPU. The parcel residual check was shortened
to 10 epochs with batch 256 after the group12 residual result already showed the
main pattern.
