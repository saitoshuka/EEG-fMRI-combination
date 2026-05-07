# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `/home/sudaxin/projects/paired_data/data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz`
- Target mode: `roi`
- PCA components: `24`
- CV: `grouped`, grouped by `subject` when grouped, `5` folds requested
- Metrics: `/home/sudaxin/projects/paired_data/results/natview_pilot_roi_offset10p5_w8_lags4to12_subjectcv/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_ridge_roi | 0.0257 | 0.0265 | 0.0024 | -0.2388 | 0.0044 | 0.0218 |
| shifted_null_roi | -0.0011 | -0.0018 | 0.0009 | -0.1715 | 0.0038 | 0.0203 |
| train_mean | 0.0000 | 0.0000 | -0.0008 | 0.0000 | 0.0038 | 0.0189 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
