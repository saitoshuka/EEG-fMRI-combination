# LaBraM NeuroSTORM Teacher Distillation

Target latent is NeuroSTORM encoder output with 2x2x2 spatial tokens and 288 features per token.

- Metrics: `results/labram_neurostorm_natview_manifold_pca32_decstrong_e14_b32/metrics.csv`

| dataset/model | latent r | row r | direct ROI r | decoded ROI r | decoded R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| natview/neurostorm_shifted_null | 0.0005 | -0.0019 | -0.0016 | -0.0190 | -0.0039 |
| natview/neurostorm_spatial | 0.0012 | -0.0005 | 0.0118 | 0.0031 | -0.0047 |
| natview/time_dataset_ridge | 0.0102 | 0.0096 | nan | -0.0122 | -0.0001 |
