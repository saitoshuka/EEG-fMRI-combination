# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_labram_plus_spatialband_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0031 | 0.0299 | -0.0297 | 0.0054 | 0.0221 | 0.0256 | 0.5182 | 0.0294 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0063 | 0.0155 | -1.1638 | 0.0063 | 0.0223 | 0.0264 | 0.5063 | 0.0094 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0017 | 0.0384 | -0.0932 | 0.0043 | 0.0206 | 0.0254 | 0.5187 | 0.0370 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | 0.0045 | 0.0298 | -0.1087 | 0.0047 | 0.0219 | 0.0262 | 0.5154 | 0.0262 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0043 | 0.0055 | -0.0001 | 0.0033 | 0.0184 | 0.0233 | 0.5056 | 0.0055 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0125 | 0.0300 | -0.0220 | 0.0045 | 0.0220 | 0.0263 | 0.5148 | 0.0265 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0050 | 0.0098 | -1.2299 | 0.0052 | 0.0205 | 0.0255 | 0.5063 | 0.0082 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0170 | 0.0523 | -0.0637 | 0.0037 | 0.0255 | 0.0272 | 0.5284 | 0.0518 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | -0.0007 | -0.0037 | -0.1211 | 0.0033 | 0.0173 | 0.0232 | 0.4984 | -0.0045 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0041 | 0.0106 | -0.0000 | 0.0050 | 0.0180 | 0.0243 | 0.5068 | 0.0111 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0023 | 0.0273 | -0.0185 | 0.0069 | 0.0231 | 0.0275 | 0.5104 | 0.0246 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0046 | 0.0078 | -1.3434 | 0.0028 | 0.0164 | 0.0225 | 0.5009 | 0.0061 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0092 | 0.0328 | -0.0756 | 0.0069 | 0.0227 | 0.0277 | 0.5160 | 0.0335 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | 0.0031 | 0.0072 | -0.1034 | 0.0045 | 0.0173 | 0.0237 | 0.5012 | 0.0057 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
