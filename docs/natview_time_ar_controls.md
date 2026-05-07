# NatView Time and BOLD Self-Prior Controls

## Goal

Add nuisance controls to the best NatView EEG-to-fMRI pilot:

- `time_only_ridge_pca`: no EEG, only within-session time basis features.
- `bold_ar_oracle_ridge_pca`: no EEG, previous true fMRI ROI values from the same held-out session.

The second model is intentionally not deployable for EEG-only inference. It is an oracle BOLD self-prior control that asks how high the score can get if the model has access to fMRI's own temporal autocorrelation, similar to how denoising diffusion can exploit a noisy fMRI input `x_t`.

## Command

```bash
conda activate eeg
python scripts/natview_pilot.py eval \
  --features data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz \
  --out-dir results/natview_pilot_pca8_time_ar_controls \
  --folds 5 \
  --group-by subject \
  --target pca \
  --n-components 8 \
  --null \
  --time-baseline \
  --bold-ar-baseline
```

## Result

| Model | ROI r mean | Spatial r mean | R2 weighted | Retrieval top-1 | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_mean | 0.0000 | -0.0008 | 0.0000 | 0.0038 | n/a |
| shifted_null_pca | -0.0010 | 0.0017 | -0.1421 | 0.0040 | 0.0015 |
| time_only_ridge_pca | 0.0174 | -0.0065 | -0.0018 | 0.0030 | 0.0055 |
| eeg_ridge_pca | 0.0291 | 0.0039 | -0.1798 | 0.0051 | 0.0123 |
| bold_ar_oracle_ridge_pca | 0.8282 | 0.6198 | 0.6869 | 0.4153 | 0.9632 |

Output directory:

```text
results/natview_pilot_pca8_time_ar_controls/
```

## Interpretation

The time-only baseline is nonzero but weaker than EEG, so NatView still has a small EEG-linked signal above simple within-run time trends.

The BOLD autoregressive oracle is enormous. It uses previous true fMRI volumes from the held-out session and reaches ROI r mean 0.8282. This is a warning for diffusion-style EEG-to-fMRI papers: if the training/evaluation setup lets the denoising model rely heavily on noisy fMRI `x_t`, high reconstruction scores may mostly reflect fMRI self-denoising and temporal autocorrelation rather than EEG condition strength.

For our project, the honest statement is:

```text
EEG predicts a weak but non-null component in NatView. However, fMRI self-priors are orders of magnitude stronger, so any generative method must include no-EEG, shuffled-EEG, and x_t-only controls.
```
