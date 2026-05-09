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
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0133 | 0.0315 | 0.0020 | 0.0052 | 0.0321 | 0.0350 | 0.5178 | 0.0302 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0139 | 0.0278 | -0.0041 | 0.0082 | 0.0412 | 0.0401 | 0.5224 | 0.0507 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | 0.0114 | 0.0175 | -0.0031 | 0.0091 | 0.0395 | 0.0400 | 0.5291 | 0.0274 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | -0.0007 | 0.0020 | -0.0040 | 0.0060 | 0.0331 | 0.0351 | 0.5042 | 0.0020 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | -0.0026 | -0.0313 | -0.0002 | 0.0033 | 0.0268 | 0.0298 | 0.4808 | 0.0194 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0164 | 0.0268 | 0.0006 | 0.0075 | 0.0385 | 0.0380 | 0.5185 | 0.0301 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0153 | 0.0227 | -0.0042 | 0.0090 | 0.0409 | 0.0396 | 0.5163 | 0.0548 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0148 | 0.0154 | -0.0033 | 0.0081 | 0.0373 | 0.0387 | 0.5260 | 0.0270 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | -0.0031 | -0.0017 | -0.0048 | 0.0041 | 0.0261 | 0.0311 | 0.4956 | -0.0054 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0057 | -0.0383 | -0.0004 | 0.0045 | 0.0271 | 0.0310 | 0.4798 | 0.0145 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0150 | 0.0319 | 0.0035 | 0.0072 | 0.0369 | 0.0372 | 0.5178 | 0.0311 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0172 | 0.0337 | -0.0028 | 0.0081 | 0.0395 | 0.0388 | 0.5203 | 0.0532 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0159 | 0.0191 | -0.0021 | 0.0061 | 0.0341 | 0.0371 | 0.5298 | 0.0298 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | 0.0025 | 0.0037 | -0.0042 | 0.0056 | 0.0314 | 0.0339 | 0.5047 | 0.0056 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
