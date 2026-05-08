# LaBraM NeuroSTORM Teacher Distillation

Target latent is NeuroSTORM encoder output with 2x2x2 spatial tokens and 288 features per token.

- Metrics: `results/labram_neurostorm_official_natview_labram_eeg_frozen_pca32_e10_b32_f1/metrics.csv`

| dataset/model | latent r | row r | direct ROI r | decoded ROI r | decoded R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| natview/neurostorm_shifted_null | 0.0011 | 0.0005 | -0.0187 | -0.0296 | -0.0073 |
| natview/neurostorm_spatial | -0.0009 | 0.0002 | -0.0072 | -0.0049 | -0.0034 |
| natview/time_dataset_ridge | 0.0104 | 0.0094 | nan | -0.0012 | -0.0000 |
