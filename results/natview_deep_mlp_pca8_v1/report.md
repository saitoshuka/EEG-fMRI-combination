# NatView Deep EEG-to-fMRI Pilot

This run predicts training-fold PCA fMRI latents from the same NatView EEG bandpower-lag features used by the ridge baseline.
The query-attention model is CATD-inspired: each lag-band EEG slice is a token and learnable fMRI latent queries cross-attend to the encoded EEG tokens.

## Configuration

- Feature file: `data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz`
- Model: `mlp`
- PCA components: `8`
- CV: `grouped` grouped by `subject` with `5` folds requested
- Metrics: `results/natview_deep_mlp_pca8_v1/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Positive ROI frac | Spatial r mean | R2 weighted | Latent r mean | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_deep_mlp | 0.0056 | 0.0055 | 0.6060 | 0.0008 | -0.0071 | -0.0006 | 3.0 |
| train_mean | 0.0000 | 0.0000 | 0.4940 | -0.0008 | 0.0000 | nan | nan |

## Notes

- Subject-heldout folds match the ridge pilot and avoid session leakage.
- The target is session-z-scored Schaefer-100 activity projected to a PCA latent fitted only on training subjects.
- `shifted_null_*` circularly shifts the training fMRI target within each session before fitting target transforms and training the neural network.
- Early stopping uses held-out training subjects, not samples from the test subjects.
