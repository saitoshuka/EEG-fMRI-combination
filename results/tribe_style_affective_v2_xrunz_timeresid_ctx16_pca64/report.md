# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/affective_spatial_bandpower.npz`
- Context steps: 16
- Target space: pca64
- Target variance retained: 0.9450
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 11625 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 11625 | 5376 | 0.0003 | -0.0382 | -0.0001 | 0.0026 | 0.0156 | 0.0217 | 0.4827 | -0.0296 |
| 1 | ridge_last | time_residual | 11625 | 5376 | 0.0141 | 0.0357 | 0.0039 | 0.0047 | 0.0260 | 0.0274 | 0.5228 | 0.0361 |
| 1 | ridge_context_mean | time_residual | 11625 | 5376 | 0.0052 | 0.0063 | -0.0005 | 0.0045 | 0.0214 | 0.0256 | 0.5062 | 0.0075 |
| 1 | longctx_lowrank_transformer | time_residual | 11625 | 5376 | 0.0108 | 0.0157 | -0.0028 | 0.0048 | 0.0247 | 0.0283 | 0.5297 | 0.0157 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 11625 | 5376 | -0.0024 | -0.0007 | -0.0045 | 0.0024 | 0.0177 | 0.0217 | 0.4947 | -0.0012 |
| 2 | train_mean | time_residual | 11655 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 11655 | 5376 | -0.0001 | -0.0360 | -0.0001 | 0.0024 | 0.0175 | 0.0221 | 0.4829 | -0.0304 |
| 2 | ridge_last | time_residual | 11655 | 5376 | 0.0185 | 0.0346 | 0.0033 | 0.0033 | 0.0253 | 0.0262 | 0.5223 | 0.0347 |
| 2 | ridge_context_mean | time_residual | 11655 | 5376 | 0.0097 | 0.0112 | 0.0004 | 0.0039 | 0.0251 | 0.0257 | 0.5066 | 0.0105 |
| 2 | longctx_lowrank_transformer | time_residual | 11655 | 5376 | 0.0122 | 0.0172 | -0.0023 | 0.0048 | 0.0234 | 0.0274 | 0.5294 | 0.0164 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 11655 | 5376 | 0.0002 | 0.0002 | -0.0041 | 0.0043 | 0.0195 | 0.0244 | 0.4998 | -0.0001 |
| 3 | train_mean | time_residual | 11520 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 11520 | 5376 | -0.0016 | -0.0358 | -0.0001 | 0.0030 | 0.0175 | 0.0221 | 0.4816 | -0.0292 |
| 3 | ridge_last | time_residual | 11520 | 5376 | 0.0163 | 0.0423 | 0.0067 | 0.0050 | 0.0249 | 0.0278 | 0.5259 | 0.0410 |
| 3 | ridge_context_mean | time_residual | 11520 | 5376 | 0.0062 | 0.0174 | 0.0005 | 0.0026 | 0.0201 | 0.0239 | 0.5085 | 0.0163 |
| 3 | longctx_lowrank_transformer | time_residual | 11520 | 5376 | 0.0104 | 0.0138 | -0.0027 | 0.0048 | 0.0251 | 0.0276 | 0.5250 | 0.0137 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 11520 | 5376 | 0.0018 | 0.0027 | -0.0036 | 0.0039 | 0.0212 | 0.0247 | 0.5054 | 0.0023 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
