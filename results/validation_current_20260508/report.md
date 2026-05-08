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

## Addendum: NeuroSTORM Manifold Bridge

The NeuroSTORM branch was upgraded from plain latent regression to a manifold bridge:

- LaBraM EEG tokens predict compressed NeuroSTORM 2x2x2 spatial latent tokens.
- A frozen ridge decoder is fit from NeuroSTORM latent to Schaefer-100 on training fMRI only; EEG-predicted latent is penalized through this frozen decoder.
- Latent token Gram loss preserves NeuroSTORM token-token spatial topology.
- Latent query attention gets the same electrode-to-query geometry regularization.
- Contrastive queue is used for larger negative pools without increasing CUDA batch size.

NatView one-fold results with PCA32 NeuroSTORM teacher:

| Setting | latent r | shifted latent r | decoded ROI r | shifted decoded ROI r | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Manifold bridge, moderate decoder weight | 0.0018 | 0.0024 | 0.0068 | -0.0232 | Decoder path beats null, latent alignment does not |
| Manifold bridge, strong decoder weight | 0.0012 | 0.0005 | 0.0031 | -0.0190 | Slight latent-null gap, weaker decoded ROI |

This means the bridge is now connected to a frozen fMRI latent-to-ROI manifold, but the current NeuroSTORM latent target is still too hard/noisy for robust EEG alignment. At this stage Schaefer spatial distillation gives the stronger measurable signal, while NeuroSTORM should remain a second-stage objective after improving the EEG-to-spatial-ROI anchor.

## Addendum: Multi-Lag EEG Token Memory

The Schaefer spatial distiller now supports multiple EEG windows around the existing BOLD-aligned window. For each fMRI target the model receives five 8 s EEG windows at offsets -4, -2, 0, +2, and +4 s relative to the current hemodynamic alignment. LaBraM extracts channel tokens for every lag, a learned lag embedding is added, and ROI queries cross-attend to the expanded lag-by-channel memory. Windows that cannot support all five lags are dropped rather than padded.

NatView one-fold result:

| Setting | Real ROI r | Shifted-null ROI r | Time baseline ROI r | Note |
| --- | ---: | ---: | ---: | --- |
| Single-lag spatial distillation | 0.0221 | 0.0087 | 0.0083 | Earlier fold-1 run |
| Five-lag spatial distillation | 0.0223 | -0.0081 | -0.0094 | Similar real r, much cleaner null separation |
| Five-lag hybrid raw+bandpower distillation | 0.0290 | -0.0280 | -0.0094 | Best fold-1 ROI r so far; R2 still negative |
| Five-lag hybrid raw+bandpower distillation | -0.0047 | -0.0027 | -0.0040 | 3-fold repeat failed; single-fold gain not stable |
| Five-lag pooled 7-dataset training | 0.0182 | n/a | -0.0094 | 21,000 balanced training windows, worse than NatView-only |
| Five-lag pooled pretrain + NatView fine-tune | 0.0099 | n/a | -0.0094 | NatView validation MSE improved, held-out subjects worsened |

This did not materially raise the absolute NatView ROI correlation, but it made the control cleaner: the real EEG-fMRI alignment stays positive while circularly shifted fMRI supervision becomes negative. The next useful test is pooled multi-dataset training with the same five-lag memory, because the current result suggests the architecture can preserve alignment but may still be data-limited.

Adding online bandpower features to each lag-by-channel LaBraM token produced the strongest fold-1 temporal correlation so far: ROI r rose to 0.0290 while shifted-null fell to -0.0280. However, the 3-fold repeat failed to preserve this gain: mean real ROI r was -0.0047 and mean shifted-null ROI r was -0.0027. The useful lesson is not that this hybrid is solved, but that raw waveform tokens alone may be missing an easier time-frequency route. The next implementation issue is checkpoint selection: the hybrid run shows strong decoupling between validation MSE and held-out temporal correlation, so selecting checkpoints by MSE is probably the wrong criterion for this objective.

The first pooled run did not support naive concatenation: even after unifying targets to Schaefer-100 and balancing each dataset to 3,000 windows, NatView held-out ROI r fell from 0.0223 to 0.0182. Pooled pretraining followed by NatView-specific fine-tuning also failed held-out subjects: NatView validation MSE improved strongly, but test ROI r fell to 0.0099. This points to subject/domain overfitting rather than a simple lack of training data.
