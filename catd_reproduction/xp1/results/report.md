# CATD Audit: XP1

This is an audit-style reproduction, not a full CATD diffusion rerun.
It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.

## Data

- Feature cache: `/home/sudaxin/projects/paired_data/catd_reproduction/xp1/features.npz`
- Metrics: `/home/sudaxin/projects/paired_data/catd_reproduction/xp1/results/generation_metrics.csv`
- Train subjects: `sub-xp101, sub-xp102, sub-xp103, sub-xp107, sub-xp108, sub-xp109, sub-xp110`
- Test subjects: `sub-xp104, sub-xp105, sub-xp106`
- Shapes: EEG `[5742, 320]`, schedule/time `[5742, 18]`, fMRI grid `[5742, 256]`

## Generation Metrics

| Model | Grid r mean | Spatial r mean | RMSE | R2 weighted | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_mean | -0.0000 | -0.0046 | 1.0000 | -0.0000 | nan |
| schedule_or_time_ridge | 0.4354 | 0.2474 | 0.8507 | 0.2763 | 0.1042 |
| eeg_ridge | 0.0350 | 0.0186 | 2.8162 | -6.9309 | -0.0096 |
| eeg_shifted_null | -0.0462 | -0.0277 | 2.1782 | -3.7446 | -0.0107 |
| eeg_plus_schedule_ridge | 0.2398 | 0.0545 | 2.0883 | -3.3609 | 0.0497 |

## Rest-vs-Task Classification

| Input | Accuracy | F1 |
| --- | ---: | ---: |
| real_fmri | 0.5892 | 0.5982 |
| schedule_generated | 0.6869 | 0.6630 |
| eeg_generated | 0.4714 | 0.1180 |
| eeg_shifted_null_generated | 0.5006 | 0.3338 |
| eeg_plus_schedule_generated | 0.5112 | 0.1486 |
| schedule_direct | 1.0000 | 1.0000 |

## Audit Interpretation

- If schedule/time-only is close to or better than EEG, high generation/classification scores can be explained without EEG-to-fMRI neural information.
- If EEG is not above shifted-null, the model is likely using fMRI slow structure or split artifacts rather than temporal correspondence.
- If generated-BOLD classification is high while reconstruction correlation is weak, downstream classification is not sufficient evidence of faithful fMRI generation.
