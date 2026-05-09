# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/pooled_affective_experience_spatial_bandpower.npz`
- Context steps: 16
- Target space: pca64
- Target variance retained: 0.9450
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 16466 | 7283 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 16466 | 7283 | -0.0067 | -0.0399 | -0.0003 | 0.0049 | 0.0272 | 0.0308 | 0.4828 | 0.0116 |
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0078 | 0.0256 | 0.0008 | 0.0070 | 0.0356 | 0.0364 | 0.5132 | 0.0200 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0081 | 0.0142 | 0.0001 | 0.0067 | 0.0330 | 0.0352 | 0.5016 | 0.0081 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | 0.0088 | 0.0047 | -0.0011 | 0.0074 | 0.0352 | 0.0365 | 0.5148 | 0.0041 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | 0.0005 | 0.0021 | -0.0015 | 0.0067 | 0.0308 | 0.0345 | 0.5026 | 0.0011 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | -0.0026 | -0.0313 | -0.0002 | 0.0033 | 0.0268 | 0.0298 | 0.4808 | 0.0194 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0119 | 0.0196 | 0.0009 | 0.0077 | 0.0369 | 0.0372 | 0.5134 | 0.0199 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0097 | 0.0067 | 0.0001 | 0.0061 | 0.0312 | 0.0344 | 0.5061 | 0.0098 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0074 | 0.0047 | -0.0011 | 0.0066 | 0.0324 | 0.0344 | 0.5111 | 0.0031 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | -0.0015 | 0.0025 | -0.0013 | 0.0059 | 0.0294 | 0.0331 | 0.4981 | -0.0002 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0057 | -0.0383 | -0.0004 | 0.0045 | 0.0271 | 0.0310 | 0.4798 | 0.0145 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0087 | 0.0298 | 0.0017 | 0.0080 | 0.0315 | 0.0369 | 0.5191 | 0.0226 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0073 | 0.0052 | -0.0001 | 0.0065 | 0.0360 | 0.0360 | 0.5057 | 0.0140 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0090 | 0.0032 | -0.0011 | 0.0069 | 0.0333 | 0.0355 | 0.5140 | 0.0029 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | -0.0029 | 0.0020 | -0.0020 | 0.0061 | 0.0305 | 0.0338 | 0.5007 | 0.0000 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
