# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0136 | 0.0229 | -0.0076 | 0.0033 | 0.0229 | 0.0254 | 0.5105 | 0.0235 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0080 | 0.0038 | -0.0001 | 0.0050 | 0.0218 | 0.0256 | 0.5022 | 0.0035 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0174 | 0.0274 | 0.0019 | 0.0060 | 0.0270 | 0.0297 | 0.5280 | 0.0232 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | 0.0003 | 0.0001 | -0.0020 | 0.0052 | 0.0214 | 0.0249 | 0.5016 | 0.0006 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0043 | 0.0055 | -0.0001 | 0.0033 | 0.0184 | 0.0233 | 0.5056 | 0.0055 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0172 | 0.0416 | -0.0069 | 0.0043 | 0.0242 | 0.0265 | 0.5254 | 0.0372 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0065 | 0.0057 | -0.0000 | 0.0054 | 0.0207 | 0.0254 | 0.5028 | 0.0038 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0173 | 0.0213 | 0.0004 | 0.0054 | 0.0257 | 0.0282 | 0.5244 | 0.0182 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | -0.0116 | -0.0032 | -0.0048 | 0.0037 | 0.0197 | 0.0241 | 0.4936 | -0.0042 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0041 | 0.0106 | -0.0000 | 0.0050 | 0.0180 | 0.0243 | 0.5068 | 0.0111 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0127 | 0.0496 | 0.0052 | 0.0047 | 0.0231 | 0.0273 | 0.5274 | 0.0455 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0067 | 0.0050 | 0.0001 | 0.0037 | 0.0206 | 0.0247 | 0.5060 | 0.0087 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0185 | 0.0288 | 0.0022 | 0.0065 | 0.0238 | 0.0294 | 0.5355 | 0.0266 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | 0.0008 | 0.0004 | -0.0016 | 0.0050 | 0.0219 | 0.0253 | 0.4990 | -0.0001 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
