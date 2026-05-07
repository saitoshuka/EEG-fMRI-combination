# CATD Audit: NODDI

This is an audit-style reproduction, not a full CATD diffusion rerun.
It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.

## Data

- Feature cache: `catd_reproduction/noddi/features.npz`
- Metrics: `catd_reproduction/noddi/results_residual/generation_metrics.csv`
- Train subjects: `32, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 50`
- Test subjects: `48, 49`
- Shapes: EEG `[4752, 315]`, schedule/time `[4752, 18]`, fMRI grid `[4752, 256]`

## Generation Metrics

| Model | Grid r mean | Spatial r mean | RMSE | R2 weighted | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_mean | 0.0000 | -0.0009 | 1.0000 | -0.0000 | nan |
| schedule_or_time_ridge | 0.3210 | 0.4871 | 0.8756 | 0.2334 | 0.0556 |
| eeg_ridge | 0.1234 | 0.0690 | 1.1101 | -0.2323 | 0.0213 |
| eeg_shifted_null | -0.1258 | -0.0716 | 1.1244 | -0.2642 | -0.0324 |
| eeg_plus_schedule_ridge | 0.2774 | 0.3515 | 0.9564 | 0.0852 | 0.0402 |

## Residual Prediction Metrics

Residual target is `real fMRI - schedule/time prediction`; this asks whether EEG explains fMRI variance beyond the no-EEG nuisance baseline.
Residual metrics: `catd_reproduction/noddi/results_residual/residual_metrics.csv`

| Model | Residual grid r mean | Residual spatial r mean | RMSE | R2 weighted | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_train_mean | -0.0000 | 0.0012 | 0.8768 | 0.0000 | nan |
| residual_schedule_or_time_ridge | 0.0502 | 0.0850 | 0.8768 | 0.0000 | 0.0021 |
| residual_eeg_ridge | -0.0187 | 0.0385 | 0.9557 | -0.1880 | -0.0283 |
| residual_eeg_shifted_null | -0.0006 | 0.0116 | 0.9240 | -0.1106 | 0.0027 |
| removed_schedule_or_time_component | 0.3224 | 0.4827 | 0.8768 | 0.2312 | nan |

## Audit Interpretation

- If schedule/time-only is close to or better than EEG, high generation/classification scores can be explained without EEG-to-fMRI neural information.
- If EEG is not above shifted-null, the model is likely using fMRI slow structure or split artifacts rather than temporal correspondence.
- If generated-BOLD classification is high while reconstruction correlation is weak, downstream classification is not sufficient evidence of faithful fMRI generation.
