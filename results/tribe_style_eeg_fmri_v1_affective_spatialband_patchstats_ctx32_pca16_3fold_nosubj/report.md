# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz`
- Context steps: 32
- Target space: pca16
- Target variance retained: 0.7551
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0062 | -0.0031 | -0.0000 | 0.0045 | 0.0199 | 0.0245 | 0.4985 | -0.0024 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0181 | 0.0371 | 0.0010 | 0.0052 | 0.0262 | 0.0276 | 0.5233 | 0.0407 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0072 | 0.0153 | -0.2737 | 0.0054 | 0.0229 | 0.0266 | 0.5102 | 0.0164 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0231 | 0.0494 | -0.0236 | 0.0048 | 0.0242 | 0.0271 | 0.5271 | 0.0477 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | -0.0062 | -0.0082 | -0.0050 | 0.0035 | 0.0210 | 0.0233 | 0.4958 | -0.0050 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0031 | 0.0052 | -0.0001 | 0.0030 | 0.0184 | 0.0228 | 0.5053 | 0.0058 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0271 | 0.0466 | 0.0054 | 0.0045 | 0.0234 | 0.0263 | 0.5260 | 0.0438 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0056 | 0.0179 | -0.1905 | 0.0039 | 0.0188 | 0.0243 | 0.5089 | 0.0131 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0240 | 0.0395 | -0.0286 | 0.0045 | 0.0227 | 0.0264 | 0.5186 | 0.0347 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | 0.0004 | 0.0021 | -0.0062 | 0.0039 | 0.0199 | 0.0241 | 0.4986 | -0.0045 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0050 | 0.0124 | -0.0000 | 0.0039 | 0.0162 | 0.0236 | 0.5072 | 0.0123 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0160 | 0.0406 | 0.0035 | 0.0045 | 0.0266 | 0.0274 | 0.5250 | 0.0429 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0087 | 0.0085 | -0.2461 | 0.0035 | 0.0216 | 0.0246 | 0.5086 | 0.0142 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0286 | 0.0558 | -0.0012 | 0.0039 | 0.0205 | 0.0259 | 0.5286 | 0.0568 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | 0.0053 | 0.0105 | -0.0026 | 0.0050 | 0.0218 | 0.0254 | 0.5086 | 0.0057 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
