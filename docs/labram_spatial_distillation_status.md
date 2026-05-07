# LaBraM Spatial Distillation Status

Date: 2026-05-08

This note records the pivot from frozen LaBraM feature evaluation to actual
fMRI-to-LaBraM spatial distillation.

## Why The Architecture Changed

The earlier frozen LaBraM run used global mean/std features from LaBraM patch
tokens.  That is useful as a diagnostic baseline, but it mostly erases which
EEG channel produced which token.  For this project, the important target is not
only "predict fMRI latent"; it is to inject fMRI spatial structure into the EEG
foundation model.

The new distillation path is therefore:

```text
EEG window
  -> LaBraM channel/time patch tokens
  -> channel tokens with electrode coordinate information
  -> fMRI spatial coordinate queries cross-attend to EEG channel tokens
  -> 4x4x4 fMRI grid prediction
  -> loss backpropagates into LaBraM final blocks / position embeddings
```

This lets fMRI spatial supervision update the EEG encoder instead of only
training a downstream regression head.

## Implemented Scripts

- `scripts/labram_frozen.py`
  - frozen LaBraM feature extraction and evaluation
  - includes MSE/correlation loss and optional map-level contrastive loss

- `scripts/labram_spatial.py`
  - frozen LaBraM channel-token spatial-query decoder
  - keeps channel tokens and electrode coordinates

- `scripts/labram_distill.py`
  - true spatial distillation path
  - raw EEG window enters LaBraM during training
  - fMRI spatial-grid loss can update LaBraM final blocks
  - optional unfreezing of LaBraM `pos_embed` and `time_embed`
  - optional map-level contrastive loss

## Frozen LaBraM Diagnostic

Result directory:
`results/labram_frozen_v1_contrast_w005_f1_e20`

This run used frozen LaBraM mean/std features plus a trainable MLP head.  It is
not the final distillation objective, but it checks whether LaBraM features are
already stronger than the small raw Transformer.

Important pattern:

- The full fMRI scores on task datasets remain mostly explained by
  `time_dataset_ridge`.
- NatView stays weak but has a small positive sign:
  - `pooled_labram_residual_mlp`: ROI r `0.0118`
  - `pooled_labram_residual_shifted_null`: ROI r `-0.0165`
  - residual latent r: `0.0024` vs `-0.0010`

This is slightly better shaped than random, but still too small to claim
successful spatial distillation.

## True LaBraM Distillation Pilots

### gradCPT, unfreeze last 2 LaBraM blocks

Result directory:
`results/labram_distill_gradcpt_unfreeze2_contrast_f1_e6`

Settings:

- full gradCPT held-out-subject fold
- 1,900 train windows, 1,213 test windows
- update LaBraM final 2 blocks plus spatial decoder
- map-level contrastive weight `0.02`

Result:

| model | ROI r | residual r |
| --- | ---: | ---: |
| time baseline | 0.0745 | n/a |
| distilled spatial | -0.0065 | n/a |
| shifted null | 0.0083 | n/a |
| distilled residual | 0.0704 | -0.0032 |
| residual shifted null | 0.0751 | 0.0068 |

Interpretation: not successful.  The residual model does not beat residual
shifted-null, so there is no evidence that this pilot distilled EEG-specific
fMRI spatial residual into LaBraM.

### Affective, unfreeze last 2 blocks + pos/time embeddings

Result directory:
`results/labram_distill_affective_pos_unfreeze_contrast_f1_e5`

Settings:

- Affective held-out-subject fold
- sampled 4,096 train windows, 6,500 test windows
- update LaBraM final 2 blocks, `pos_embed`, `time_embed`, and spatial decoder
- map-level contrastive weight `0.02`

Result:

| model | ROI r | residual r |
| --- | ---: | ---: |
| time baseline | 0.0479 | n/a |
| distilled spatial | 0.0085 | n/a |
| shifted null | 0.0040 | n/a |
| distilled residual | 0.0235 | -0.0303 |
| residual shifted null | 0.0418 | -0.0028 |

Interpretation: also not successful.  Updating channel position embeddings did
not produce a reliable EEG-specific fMRI residual signal in this small pilot.

## Current Conclusion

The architectural direction is now aligned with the real goal: spatial fMRI
supervision can update LaBraM parameters.  However, the first pilots are
negative.  In the strict residual/shifted-null setting, the model is not yet
learning a robust EEG-specific spatial fMRI signal.

This does not mean the direction is dead.  It means the next useful step is not
just "bigger model"; it should improve the teacher signal and distillation
target.

## Next Changes Worth Trying

- Use a real fMRI teacher encoder instead of raw 4x4x4 grid regression.  A
  spatial autoencoder/VAE target should be smoother and more meaningful than
  coarse voxel averages.
- Distill fMRI spatial representations at multiple levels:
  - global map embedding contrast
  - region/voxel query prediction
  - spatial smoothness or neighborhood consistency
- Add EEG channel graph/geodesic bias before cross-attention, so nearby
  electrodes share information before fMRI query decoding.
- Train on all grid datasets jointly for the distillation phase, then evaluate
  subject-heldout per dataset.
- After a useful signal appears, save a LaBraM+adapter checkpoint and test
  whether the adapted EEG encoder improves downstream EEG-only representations.
