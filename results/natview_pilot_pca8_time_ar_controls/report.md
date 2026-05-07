# NatView Rest EEG-to-fMRI Pilot

This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.
EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.

## Configuration

- Feature file: `data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz`
- Target mode: `pca`
- PCA components: `8`
- CV: `grouped`, grouped by `subject` when grouped, `5` folds requested
- Metrics: `results/natview_pilot_pca8_time_ar_controls/metrics.csv`

## Mean Metrics Across Folds

| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bold_ar_oracle_ridge_pca | 0.8282 | 0.8342 | 0.6198 | 0.6869 | 0.4153 | 0.7532 |
| eeg_ridge_pca | 0.0291 | 0.0293 | 0.0039 | -0.1798 | 0.0051 | 0.0215 |
| shifted_null_pca | -0.0010 | -0.0011 | 0.0017 | -0.1421 | 0.0040 | 0.0208 |
| time_only_ridge_pca | 0.0174 | 0.0175 | -0.0065 | -0.0018 | 0.0030 | 0.0150 |
| train_mean | 0.0000 | 0.0000 | -0.0008 | 0.0000 | 0.0038 | 0.0189 |

## Interpretation Notes

- Subject-level folds are used to avoid session leakage.
- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.
- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.
- `time_only_*` uses only within-run time bases; `bold_ar_oracle_*` uses previous true fMRI volumes and is a non-deployable BOLD self-prior control.
- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.
