# EEG-to-fMRI Validation Status

Date: 2026-05-08

## Current verdict

Do not scale the current LaBraM distillation setup to a larger GPU yet. The available evidence says there is weak EEG-conditioned signal in NatView, but the present LaBraM Schaefer-100 training recipe has not passed the essential shifted-null controls.

## Data scale used

- Masked unified Schaefer-100 cache: 164 runs, 57,985 windows, 7 datasets.
- Training for NatView-heldout evaluation: 55,921 train windows, 2,064 NatView test windows.
- XP2 was rescued with ROI masks: mean ROI coverage 0.868, minimum 0.800.

## LaBraM Schaefer-100 results

| Setting | Real ROI r | Shifted-null ROI r | Time baseline ROI r | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Full masked pooled, all datasets | -0.0192 | -0.0067 | 0.0083 | Fails null and time baseline |
| Full masked residual target | 0.0081 | 0.0089 | 0.0083 | Residual signal not learned; residual r real -0.0099 vs null 0.0175 |
| Pooled pretrain + NatView fine-tune | -0.0245 | -0.0144 | 0.0083 | Fine-tune did not rescue |
| NatView-only + contrastive queue | 0.0074 | 0.0177 | 0.0083 | Queue helps optimization slightly, but null is better |

## Positive controls from simpler models

| Setting | Real ROI r | Shifted-null ROI r | Time-only ROI r | Interpretation |
| --- | ---: | ---: | ---: | --- |
| NatView band/lag ridge, ROI target, subject-CV | 0.0257 | -0.0011 | n/a | Weak but clean EEG-conditioned signal |
| NatView band/lag ridge, PCA8 target, subject-CV | 0.0291 | -0.0010 | 0.0174 | EEG beats time-only and shifted-null |
| NatView band/lag deep MLP, PCA16 target | 0.0135 | -0.0032 | n/a | Weak positive, less strong than ridge |

## Interpretation

The current evidence does not support spending larger-GPU compute on simply increasing batch size or training the same LaBraM head longer. Larger batch contrastive was tested approximately with a 1024 target queue; it still failed the shifted-null control in the NatView-only anchor run.

This does not kill the project. It narrows the problem: the dataset likely contains a weak EEG-to-fMRI signal, but it is easiest to see with carefully engineered lagged EEG features. The current foundation-model distillation head is probably losing the signal because of some combination of subject-domain shift, target noise, weak fMRI SNR, and insufficiently diagnostic loss/validation design.

## Next go/no-go standard

Before larger-GPU training, require all of the following on NatView subject-heldout:

- Real EEG model ROI r must exceed shifted-null by at least 0.01.
- Real EEG model must beat time-only baseline.
- Residual-target real model should exceed residual shifted-null.
- The effect should repeat over more than one subject fold.

## Recommended next experiments

1. Reproduce the ridge-positive setting with a small neural model that consumes the same lagged band features. This checks whether the deep training code can recover the known weak signal.
2. Add raw waveform tokens only after the band-lag neural model passes shifted-null, not before.
3. Keep ROI mask loss for partial-coverage datasets, but do not treat it as a performance fix by itself.
4. Use multi-dataset data only after the NatView anchor is positive; naive concatenation currently hurts.
5. Treat NeuroSTORM/fMRI teacher latents as a second-stage target only after the Schaefer-100 anchor passes controls.

## Addendum: Spatial Distillation

A stronger spatial distillation variant was added after the initial negative LaBraM runs:

- ROI spatial correlation-matrix loss: match the ROI-by-ROI correlation structure of predicted Schaefer-100 activity to the true fMRI target within each training batch.
- Attention geometry loss: make ROI queries attend to EEG channels according to a soft electrode-to-ROI geometry prior, so the bridge cannot remain purely name/id based.
- Existing masked MSE, ROI temporal correlation, contrastive queue, and electrode positional smoothness were retained.

NatView subject-heldout direct Schaefer-100 results improved:

| Setting | Folds | Real ROI r | Shifted-null ROI r | Time baseline ROI r | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| NatView-only contrastive queue, before spatial distillation | 1 | 0.0074 | 0.0177 | 0.0083 | Failed null |
| NatView spatial distillation | 1 | 0.0221 | 0.0087 | 0.0083 | Passed direct null/time on this split |
| NatView spatial distillation | 3 | 0.0215 | 0.0124 | -0.0070 | First multi-fold LaBraM direct run above null |

This is a useful positive signal, but still not enough to declare success. The residual-target spatial-distillation split remained weaker than residual shifted-null, so the current model is learning some fMRI spatially structured signal, but not yet a clean residual EEG-to-fMRI mapping.
