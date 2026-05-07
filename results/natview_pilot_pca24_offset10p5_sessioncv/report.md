# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `/home/sudaxin/projects/paired_data/data/derived/natview_rest_features_papertr_offset10p5.npz`
- Target mode: `pca`
- PCA components: `24`
- CV: `grouped`, grouped by `session` when grouped, `5` folds requested
- Metrics: `/home/sudaxin/projects/paired_data/results/natview_pilot_pca24_offset10p5_sessioncv/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_ridge_pca | 0.0183 | 0.0198 | -0.0010 | -0.0074 | 0.0028 | 0.0199 |
| shifted_null_pca | 0.0020 | 0.0014 | 0.0017 | -0.0053 | 0.0036 | 0.0202 |
| train_mean | -0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0038 | 0.0188 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
