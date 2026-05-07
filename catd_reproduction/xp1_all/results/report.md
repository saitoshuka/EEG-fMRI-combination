# CATD Audit: XP1

This is an audit-style reproduction, not a full CATD diffusion rerun.
It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.

## Data

- Feature cache: `catd_reproduction/xp1_all/features.npz`
- Metrics: `catd_reproduction/xp1_all/results/generation_metrics.csv`
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
