# LaBraM NeuroSTORM Teacher Distillation

Target latent is NeuroSTORM encoder output with 2x2x2 spatial tokens and 288 features per token.

- Metrics: `results/labram_neurostorm_natview_manifold_pca32_e12_b32/metrics.csv`

| dataset/model | latent r | row r | direct ROI r | decoded ROI r | decoded R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| natview/neurostorm_shifted_null | 0.0024 | -0.0003 | 0.0063 | -0.0232 | -0.0021 |
| natview/neurostorm_spatial | 0.0018 | -0.0012 | 0.0120 | 0.0068 | -0.0022 |
| natview/time_dataset_ridge | 0.0102 | 0.0096 | nan | -0.0122 | -0.0001 |
