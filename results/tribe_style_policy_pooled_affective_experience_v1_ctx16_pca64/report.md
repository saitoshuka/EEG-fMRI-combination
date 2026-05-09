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
| 1 | train_mean | real | 16466 | 7283 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 16466 | 7283 | 0.0122 | 0.0254 | 0.0005 | 0.0088 | 0.0376 | 0.0397 | 0.5233 | 0.0596 |
| 1 | ridge_last | real | 16466 | 7283 | 0.0084 | 0.0282 | 0.0009 | 0.0063 | 0.0373 | 0.0367 | 0.5150 | 0.0216 |
| 1 | ridge_context_mean | real | 16466 | 7283 | 0.0085 | 0.0176 | 0.0002 | 0.0067 | 0.0350 | 0.0357 | 0.5041 | 0.0100 |
| 1 | longctx_lowrank_transformer | real | 16466 | 7283 | 0.0098 | 0.0079 | -0.0011 | 0.0085 | 0.0378 | 0.0377 | 0.5189 | 0.0064 |
| 1 | longctx_lowrank_transformer | shifted_null | 16466 | 7283 | 0.0010 | 0.0038 | -0.0015 | 0.0070 | 0.0305 | 0.0350 | 0.5048 | 0.0018 |
| 2 | train_mean | real | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 16490 | 7244 | 0.0149 | 0.0275 | 0.0005 | 0.0059 | 0.0385 | 0.0371 | 0.5185 | 0.0643 |
| 2 | ridge_last | real | 16490 | 7244 | 0.0129 | 0.0237 | 0.0012 | 0.0080 | 0.0359 | 0.0372 | 0.5156 | 0.0229 |
| 2 | ridge_context_mean | real | 16490 | 7244 | 0.0114 | 0.0151 | 0.0003 | 0.0064 | 0.0353 | 0.0354 | 0.5079 | 0.0140 |
| 2 | longctx_lowrank_transformer | real | 16490 | 7244 | 0.0100 | 0.0072 | -0.0010 | 0.0077 | 0.0327 | 0.0363 | 0.5166 | 0.0054 |
| 2 | longctx_lowrank_transformer | shifted_null | 16490 | 7244 | -0.0021 | 0.0028 | -0.0013 | 0.0054 | 0.0308 | 0.0334 | 0.4978 | -0.0005 |
| 3 | train_mean | real | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 16402 | 7272 | 0.0124 | 0.0274 | 0.0004 | 0.0076 | 0.0411 | 0.0391 | 0.5180 | 0.0627 |
| 3 | ridge_last | real | 16402 | 7272 | 0.0094 | 0.0337 | 0.0019 | 0.0066 | 0.0323 | 0.0365 | 0.5200 | 0.0250 |
| 3 | ridge_context_mean | real | 16402 | 7272 | 0.0081 | 0.0126 | 0.0000 | 0.0067 | 0.0345 | 0.0361 | 0.5061 | 0.0158 |
| 3 | longctx_lowrank_transformer | real | 16402 | 7272 | 0.0113 | 0.0049 | -0.0010 | 0.0056 | 0.0330 | 0.0354 | 0.5169 | 0.0043 |
| 3 | longctx_lowrank_transformer | shifted_null | 16402 | 7272 | -0.0027 | 0.0046 | -0.0020 | 0.0058 | 0.0312 | 0.0338 | 0.5010 | 0.0000 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
