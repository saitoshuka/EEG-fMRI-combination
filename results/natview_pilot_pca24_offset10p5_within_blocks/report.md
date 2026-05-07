# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `/home/sudaxin/projects/paired_data/data/derived/natview_rest_features_papertr_offset10p5.npz`
- Target mode: `pca`
- PCA components: `24`
- CV: `within_session_blocks`, grouped by `subject` when grouped, `5` folds requested
- Metrics: `/home/sudaxin/projects/paired_data/results/natview_pilot_pca24_offset10p5_within_blocks/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_ridge_pca | 0.0088 | 0.0107 | -0.0175 | -0.0137 | 0.0175 | 0.0905 |
| shifted_null_pca | -0.0240 | -0.0246 | -0.0154 | -0.0097 | 0.0195 | 0.0959 |
| train_mean | 0.0000 | 0.0000 | -0.0105 | -0.0003 | 0.0188 | 0.0942 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
