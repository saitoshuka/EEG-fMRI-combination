# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/affective_spatial_bandpower.npz`
- Context steps: 16
- Target space: identity
- Target variance retained: 1.0000
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 11625 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 11625 | 5376 | 0.0081 | -0.0002 | 0.0000 | 0.0026 | 0.0192 | 0.0238 | 0.5066 | 0.0039 |
| 1 | ridge_last | time_residual | 11625 | 5376 | 0.0714 | 0.0089 | 0.0031 | 0.0037 | 0.0253 | 0.0267 | 0.5222 | 0.0347 |
| 1 | ridge_context_mean | time_residual | 11625 | 5376 | 0.0179 | 0.0032 | -0.0007 | 0.0037 | 0.0205 | 0.0248 | 0.5052 | 0.0058 |
| 1 | longctx_lowrank_transformer | time_residual | 11625 | 5376 | 0.0189 | 0.0095 | -0.0037 | 0.0048 | 0.0251 | 0.0279 | 0.5299 | 0.0155 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 11625 | 5376 | 0.0004 | 0.0058 | -0.0055 | 0.0037 | 0.0199 | 0.0235 | 0.4963 | -0.0015 |
| 2 | train_mean | time_residual | 11655 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 11655 | 5376 | 0.0067 | 0.0015 | -0.0000 | 0.0030 | 0.0192 | 0.0237 | 0.5089 | 0.0087 |
| 2 | ridge_last | time_residual | 11655 | 5376 | 0.0656 | 0.0125 | 0.0030 | 0.0041 | 0.0242 | 0.0270 | 0.5243 | 0.0341 |
| 2 | ridge_context_mean | time_residual | 11655 | 5376 | 0.0250 | 0.0084 | 0.0003 | 0.0043 | 0.0266 | 0.0265 | 0.5080 | 0.0123 |
| 2 | longctx_lowrank_transformer | time_residual | 11655 | 5376 | 0.0168 | 0.0118 | -0.0041 | 0.0045 | 0.0223 | 0.0263 | 0.5329 | 0.0146 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 11655 | 5376 | -0.0054 | 0.0017 | -0.0063 | 0.0045 | 0.0164 | 0.0227 | 0.4888 | -0.0056 |
| 3 | train_mean | time_residual | 11520 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 11520 | 5376 | 0.0030 | -0.0010 | -0.0001 | 0.0041 | 0.0206 | 0.0245 | 0.5022 | 0.0025 |
| 3 | ridge_last | time_residual | 11520 | 5376 | 0.0786 | 0.0106 | 0.0059 | 0.0056 | 0.0244 | 0.0280 | 0.5240 | 0.0380 |
| 3 | ridge_context_mean | time_residual | 11520 | 5376 | 0.0209 | 0.0020 | 0.0003 | 0.0028 | 0.0210 | 0.0243 | 0.5067 | 0.0134 |
| 3 | longctx_lowrank_transformer | time_residual | 11520 | 5376 | 0.0171 | 0.0104 | -0.0035 | 0.0043 | 0.0219 | 0.0264 | 0.5224 | 0.0116 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 11520 | 5376 | 0.0051 | 0.0048 | -0.0044 | 0.0045 | 0.0225 | 0.0252 | 0.4995 | 0.0003 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
