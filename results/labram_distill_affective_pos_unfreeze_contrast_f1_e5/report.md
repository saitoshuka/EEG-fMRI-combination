# Spatial fMRI Distillation Into LaBraM

This run updates LaBraM parameters/adapters using fMRI spatial-grid supervision.

- Metrics: `results/labram_distill_affective_pos_unfreeze_contrast_f1_e5/metrics.csv`

| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | residual r |
| --- | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725/distilled_residual | 0.0235 | -0.0679 | -0.1188 | -0.0303 |
| Affective_music_listening_OpenNeuro_ds002725/distilled_residual_shifted_null | 0.0418 | -0.0603 | -0.0867 | -0.0028 |
| Affective_music_listening_OpenNeuro_ds002725/distilled_shifted_null | 0.0040 | 0.0094 | -0.0183 | nan |
| Affective_music_listening_OpenNeuro_ds002725/distilled_spatial | 0.0085 | 0.0107 | -0.0137 | nan |
| Affective_music_listening_OpenNeuro_ds002725/time_dataset_ridge | 0.0479 | -0.0579 | -0.0756 | nan |
