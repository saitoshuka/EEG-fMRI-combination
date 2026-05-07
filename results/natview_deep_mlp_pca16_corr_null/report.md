# NatView Deep EEG-to-fMRI Pilot

This run predicts training-fold PCA fMRI latents from the same NatView EEG bandpower-lag features used by the ridge baseline.
The query-attention model is CATD-inspired: each lag-band EEG slice is a token and learnable fMRI latent queries cross-attend to the encoded EEG tokens.

## Configuration

- Feature file: `data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz`
- Model: `mlp`
- PCA components: `16`
- CV: `grouped` grouped by `subject` with `5` folds requested
- Metrics: `results/natview_deep_mlp_pca16_corr_null/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Positive ROI frac | Spatial r mean | R2 weighted | Latent r mean | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_deep_mlp | 0.0135 | 0.0144 | 0.7340 | -0.0005 | -0.0383 | 0.0028 | 8.8 |
| shifted_null_mlp | -0.0032 | -0.0026 | 0.4640 | -0.0010 | -0.0174 | -0.0004 | 2.8 |
| train_mean | 0.0000 | 0.0000 | 0.4940 | -0.0008 | 0.0000 | nan | nan |

## Notes

- Subject-heldout folds match the ridge pilot and avoid session leakage.
- The target is session-z-scored Schaefer-100 activity projected to a PCA latent fitted only on training subjects.
- `shifted_null_*` circularly shifts the training fMRI target within each session before fitting target transforms and training the neural network.
- Early stopping uses held-out training subjects, not samples from the test subjects.
