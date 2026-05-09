# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/pooled_affective_experience_spatial_bandpower.npz`
- Context steps: 16
- Target space: pca32
- Target variance retained: 0.8495
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 16466 | 7283 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 16466 | 7283 | -0.0057 | -0.0394 | -0.0003 | 0.0052 | 0.0269 | 0.0312 | 0.4831 | 0.0145 |
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0167 | 0.0329 | 0.0026 | 0.0060 | 0.0325 | 0.0354 | 0.5172 | 0.0317 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0203 | 0.0296 | -0.0041 | 0.0081 | 0.0420 | 0.0400 | 0.5219 | 0.0539 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | 0.0182 | 0.0228 | -0.0027 | 0.0084 | 0.0419 | 0.0403 | 0.5277 | 0.0365 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | -0.0017 | 0.0000 | -0.0044 | 0.0049 | 0.0297 | 0.0329 | 0.5005 | -0.0000 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | 0.0005 | -0.0309 | -0.0002 | 0.0033 | 0.0279 | 0.0299 | 0.4810 | 0.0226 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0178 | 0.0277 | 0.0010 | 0.0073 | 0.0389 | 0.0379 | 0.5174 | 0.0309 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0198 | 0.0240 | -0.0042 | 0.0081 | 0.0403 | 0.0390 | 0.5158 | 0.0575 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0225 | 0.0249 | -0.0030 | 0.0083 | 0.0413 | 0.0398 | 0.5299 | 0.0343 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | -0.0034 | -0.0035 | -0.0048 | 0.0057 | 0.0283 | 0.0327 | 0.4918 | -0.0035 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0031 | -0.0377 | -0.0004 | 0.0040 | 0.0290 | 0.0309 | 0.4807 | 0.0183 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0192 | 0.0333 | 0.0042 | 0.0067 | 0.0355 | 0.0366 | 0.5172 | 0.0319 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0223 | 0.0358 | -0.0028 | 0.0081 | 0.0382 | 0.0388 | 0.5198 | 0.0555 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0212 | 0.0248 | -0.0035 | 0.0091 | 0.0425 | 0.0403 | 0.5270 | 0.0359 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | 0.0066 | 0.0072 | -0.0040 | 0.0056 | 0.0344 | 0.0354 | 0.5100 | 0.0096 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
