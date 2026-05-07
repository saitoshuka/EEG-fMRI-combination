# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `/home/sudaxin/projects/paired_data/data/derived/natview_smoke_features.npz`
- Target mode: `pca`
- PCA components: `4`
- CV: grouped by subject, `2` folds requested
- Metrics: `/home/sudaxin/projects/paired_data/results/natview_smoke/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_ridge_pca | 0.0462 | 0.0463 | 0.0136 | -0.0371 | 0.0040 | 0.0238 |
| shifted_null_pca | -0.0020 | -0.0032 | -0.0019 | -0.0506 | 0.0048 | 0.0185 |
| train_mean | 0.0000 | 0.0000 | -0.0004 | -0.0000 | 0.0035 | 0.0176 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
