# CATD Audit: XP1

This is an audit-style reproduction, not a full CATD diffusion rerun.
It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.

## Data

- Feature cache: `catd_reproduction/xp1_all/features.npz`
- Metrics: `catd_reproduction/xp1_all/results_residual/generation_metrics.csv`
- Train subjects: `sub-xp101, sub-xp102, sub-xp103, sub-xp107, sub-xp108, sub-xp109, sub-xp110`
- Test subjects: `sub-xp104, sub-xp105, sub-xp106`
- Shapes: EEG `[8648, 320]`, schedule/time `[8648, 18]`, fMRI grid `[8648, 256]`

## Generation Metrics

| Model | Grid r mean | Spatial r mean | RMSE | R2 weighted | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_mean | 0.0000 | -0.0035 | 1.0000 | -0.0000 | nan |
| schedule_or_time_ridge | 0.4460 | 0.2726 | 0.8478 | 0.2813 | 0.1017 |
| eeg_ridge | 0.0285 | 0.0270 | 3.1799 | -9.1118 | -0.0056 |
| eeg_shifted_null | -0.0202 | -0.0275 | 1.4203 | -1.0172 | -0.0107 |
| eeg_plus_schedule_ridge | 0.2237 | 0.0655 | 1.8646 | -2.4766 | 0.0420 |

## Residual Prediction Metrics

Residual target is `real fMRI - schedule/time prediction`; this asks whether EEG explains fMRI variance beyond the no-EEG nuisance baseline.
Residual metrics: `catd_reproduction/xp1_all/results_residual/residual_metrics.csv`

| Model | Residual grid r mean | Residual spatial r mean | RMSE | R2 weighted | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_train_mean | 0.0000 | 0.0036 | 0.8485 | -0.0000 | nan |
| residual_schedule_or_time_ridge | 0.0622 | -0.0199 | 0.8485 | 0.0000 | 0.0406 |
| residual_eeg_ridge | -0.0039 | -0.0041 | 1.7411 | -3.2100 | -0.0115 |
| residual_eeg_shifted_null | -0.0031 | -0.0028 | 1.2794 | -1.2734 | 0.0074 |
| removed_schedule_or_time_component | 0.4441 | 0.2717 | 0.8485 | 0.2800 | nan |

## Rest-vs-Task Classification

| Input | Accuracy | F1 |
| --- | ---: | ---: |
| real_fmri | 0.6153 | 0.6291 |
| schedule_generated | 0.6141 | 0.6168 |
| eeg_generated | 0.5077 | 0.4509 |
| eeg_shifted_null_generated | 0.4889 | 0.2872 |
| eeg_plus_schedule_generated | 0.5138 | 0.1854 |
| schedule_direct | 1.0000 | 1.0000 |

## Audit Interpretation

- If schedule/time-only is close to or better than EEG, high generation/classification scores can be explained without EEG-to-fMRI neural information.
- If EEG is not above shifted-null, the model is likely using fMRI slow structure or split artifacts rather than temporal correspondence.
- If generated-BOLD classification is high while reconstruction correlation is weak, downstream classification is not sufficient evidence of faithful fMRI generation.
