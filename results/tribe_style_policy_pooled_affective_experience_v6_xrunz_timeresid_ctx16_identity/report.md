# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/pooled_affective_experience_spatial_bandpower.npz`
- Context steps: 16
- Target space: identity
- Target variance retained: 1.0000
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 16466 | 7283 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 16466 | 7283 | -0.0054 | -0.0392 | -0.0003 | 0.0047 | 0.0261 | 0.0305 | 0.4826 | 0.0100 |
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0595 | 0.0104 | 0.0018 | 0.0056 | 0.0325 | 0.0352 | 0.5181 | 0.0294 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0296 | 0.0217 | -0.0041 | 0.0087 | 0.0423 | 0.0405 | 0.5226 | 0.0491 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | 0.0207 | 0.0143 | -0.0037 | 0.0097 | 0.0433 | 0.0427 | 0.5302 | 0.0268 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | 0.0043 | 0.0014 | -0.0054 | 0.0070 | 0.0343 | 0.0355 | 0.5071 | 0.0031 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | -0.0033 | -0.0292 | -0.0002 | 0.0033 | 0.0262 | 0.0296 | 0.4805 | 0.0179 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0502 | 0.0141 | 0.0004 | 0.0077 | 0.0366 | 0.0383 | 0.5190 | 0.0294 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0281 | 0.0182 | -0.0041 | 0.0090 | 0.0403 | 0.0396 | 0.5163 | 0.0534 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0173 | 0.0117 | -0.0048 | 0.0086 | 0.0404 | 0.0400 | 0.5308 | 0.0246 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | -0.0017 | -0.0005 | -0.0058 | 0.0069 | 0.0329 | 0.0341 | 0.4972 | -0.0023 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0083 | -0.0335 | -0.0003 | 0.0043 | 0.0282 | 0.0308 | 0.4797 | 0.0131 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0627 | 0.0181 | 0.0031 | 0.0067 | 0.0366 | 0.0371 | 0.5179 | 0.0306 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0271 | 0.0277 | -0.0028 | 0.0078 | 0.0391 | 0.0389 | 0.5203 | 0.0518 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0209 | 0.0125 | -0.0035 | 0.0069 | 0.0373 | 0.0385 | 0.5319 | 0.0292 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | -0.0014 | 0.0028 | -0.0060 | 0.0063 | 0.0301 | 0.0338 | 0.4975 | 0.0008 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
