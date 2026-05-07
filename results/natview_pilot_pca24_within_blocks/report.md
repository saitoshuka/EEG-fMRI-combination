# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `/home/sudaxin/projects/paired_data/data/derived/natview_rest_features.npz`
- Target mode: `pca`
- PCA components: `24`
- CV: `within_session_blocks`, grouped by `subject` when grouped, `5` folds requested
- Metrics: `/home/sudaxin/projects/paired_data/results/natview_pilot_pca24_within_blocks/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_ridge_pca | -0.0030 | -0.0021 | -0.0153 | -0.0031 | 0.0158 | 0.0848 |
| shifted_null_pca | -0.0079 | -0.0076 | -0.0129 | -0.0020 | 0.0177 | 0.0887 |
| train_mean | 0.0000 | 0.0000 | -0.0137 | -0.0003 | 0.0176 | 0.0881 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
