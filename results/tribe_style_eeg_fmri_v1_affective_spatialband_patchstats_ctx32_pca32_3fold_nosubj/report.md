# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0119 | 0.0353 | 0.0000 | 0.0054 | 0.0249 | 0.0275 | 0.5243 | 0.0378 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0060 | 0.0131 | -0.2695 | 0.0065 | 0.0246 | 0.0275 | 0.5106 | 0.0145 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0168 | 0.0414 | 0.0032 | 0.0048 | 0.0227 | 0.0272 | 0.5271 | 0.0371 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | -0.0032 | -0.0000 | -0.0037 | 0.0047 | 0.0184 | 0.0239 | 0.4932 | -0.0035 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0043 | 0.0055 | -0.0001 | 0.0033 | 0.0184 | 0.0233 | 0.5056 | 0.0055 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0166 | 0.0429 | 0.0042 | 0.0056 | 0.0210 | 0.0266 | 0.5261 | 0.0403 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0054 | 0.0153 | -0.1907 | 0.0045 | 0.0199 | 0.0247 | 0.5095 | 0.0124 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0157 | 0.0417 | -0.0012 | 0.0047 | 0.0236 | 0.0271 | 0.5271 | 0.0383 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | -0.0008 | -0.0019 | -0.0032 | 0.0030 | 0.0173 | 0.0228 | 0.4973 | -0.0020 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0041 | 0.0106 | -0.0000 | 0.0050 | 0.0180 | 0.0243 | 0.5068 | 0.0111 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0104 | 0.0374 | 0.0021 | 0.0052 | 0.0259 | 0.0277 | 0.5253 | 0.0400 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0091 | 0.0097 | -0.2430 | 0.0037 | 0.0223 | 0.0249 | 0.5089 | 0.0143 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0171 | 0.0539 | 0.0091 | 0.0050 | 0.0270 | 0.0286 | 0.5373 | 0.0515 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | 0.0044 | 0.0051 | -0.0027 | 0.0060 | 0.0219 | 0.0263 | 0.5067 | 0.0036 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
