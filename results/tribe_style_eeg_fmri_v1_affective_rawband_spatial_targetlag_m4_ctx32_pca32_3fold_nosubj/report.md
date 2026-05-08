# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower_schaefer100_targetlag_m4.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9651 | 5369 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9651 | 5369 | 0.0049 | 0.0023 | -0.0000 | 0.0047 | 0.0188 | 0.0249 | 0.5036 | 0.0023 |
| 1 | ridge_last | real | 9651 | 5369 | 0.0365 | 0.0382 | -0.0067 | 0.0067 | 0.0242 | 0.0289 | 0.5251 | 0.0381 |
| 1 | ridge_context_mean | real | 9651 | 5369 | 0.0066 | 0.0003 | -0.0001 | 0.0043 | 0.0225 | 0.0250 | 0.4959 | -0.0011 |
| 1 | longctx_lowrank_transformer | real | 9651 | 5369 | 0.0415 | 0.0296 | 0.0013 | 0.0056 | 0.0296 | 0.0310 | 0.5452 | 0.0268 |
| 1 | longctx_lowrank_transformer | shifted_null | 9651 | 5369 | -0.0040 | 0.0029 | -0.0018 | 0.0056 | 0.0216 | 0.0254 | 0.5008 | -0.0003 |
| 2 | train_mean | real | 9722 | 5360 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9722 | 5360 | 0.0043 | 0.0004 | -0.0001 | 0.0035 | 0.0188 | 0.0236 | 0.5011 | -0.0003 |
| 2 | ridge_last | real | 9722 | 5360 | 0.0375 | 0.0425 | -0.0000 | 0.0071 | 0.0261 | 0.0300 | 0.5307 | 0.0434 |
| 2 | ridge_context_mean | real | 9722 | 5360 | 0.0061 | 0.0058 | -0.0001 | 0.0056 | 0.0243 | 0.0268 | 0.4999 | 0.0004 |
| 2 | longctx_lowrank_transformer | real | 9722 | 5360 | 0.0389 | 0.0359 | 0.0020 | 0.0060 | 0.0284 | 0.0314 | 0.5501 | 0.0342 |
| 2 | longctx_lowrank_transformer | shifted_null | 9722 | 5360 | -0.0084 | -0.0031 | -0.0020 | 0.0026 | 0.0201 | 0.0226 | 0.4973 | -0.0018 |
| 3 | train_mean | real | 9440 | 5363 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9440 | 5363 | 0.0037 | 0.0076 | -0.0001 | 0.0045 | 0.0196 | 0.0246 | 0.5058 | 0.0098 |
| 3 | ridge_last | real | 9440 | 5363 | 0.0332 | 0.0564 | -0.0011 | 0.0058 | 0.0267 | 0.0290 | 0.5342 | 0.0476 |
| 3 | ridge_context_mean | real | 9440 | 5363 | 0.0050 | 0.0116 | 0.0000 | 0.0043 | 0.0194 | 0.0246 | 0.5035 | 0.0059 |
| 3 | longctx_lowrank_transformer | real | 9440 | 5363 | 0.0371 | 0.0260 | 0.0015 | 0.0062 | 0.0272 | 0.0304 | 0.5448 | 0.0262 |
| 3 | longctx_lowrank_transformer | shifted_null | 9440 | 5363 | -0.0090 | 0.0014 | -0.0020 | 0.0037 | 0.0177 | 0.0234 | 0.4943 | -0.0023 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
