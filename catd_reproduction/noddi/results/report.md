# CATD Audit: NODDI

This is an audit-style reproduction, not a full CATD diffusion rerun.
It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.

## Data

- Feature cache: `/home/sudaxin/projects/paired_data/catd_reproduction/noddi/features.npz`
- Metrics: `/home/sudaxin/projects/paired_data/catd_reproduction/noddi/results/generation_metrics.csv`
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

## Audit Interpretation

- If schedule/time-only is close to or better than EEG, high generation/classification scores can be explained without EEG-to-fMRI neural information.
- If EEG is not above shifted-null, the model is likely using fMRI slow structure or split artifacts rather than temporal correspondence.
- If generated-BOLD classification is high while reconstruction correlation is weak, downstream classification is not sufficient evidence of faithful fMRI generation.
