# LaBraM Schaefer100 Distillation

All included fMRI targets share the same Schaefer2018 100-parcel atlas.

- Metrics: `results/labram_schaefer100_natview_spatialdistill_e12_b32/metrics.csv`

| dataset/model | ROI r mean | spatial r mean | R2 weighted | residual r |
| --- | ---: | ---: | ---: | ---: |
| natview/schaefer_residual | 0.0097 | 0.0027 | -0.0089 | 0.0065 |
| natview/schaefer_residual_shifted_null | 0.0129 | 0.0014 | -0.0116 | 0.0160 |
| natview/schaefer_shifted_null | 0.0087 | -0.0028 | -0.0017 | nan |
| natview/schaefer_spatial | 0.0221 | 0.0032 | 0.0001 | nan |
| natview/time_dataset_ridge | 0.0083 | 0.0045 | -0.0083 | nan |
