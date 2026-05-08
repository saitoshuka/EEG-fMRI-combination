# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/labram_retrieval_schaefer100_affective/features.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0021 | 0.0266 | -0.0287 | 0.0056 | 0.0218 | 0.0262 | 0.5135 | 0.0261 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0048 | 0.0043 | -0.6466 | 0.0050 | 0.0208 | 0.0247 | 0.4997 | 0.0017 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0027 | 0.0329 | -0.0743 | 0.0050 | 0.0218 | 0.0265 | 0.5201 | 0.0349 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | 0.0040 | 0.0336 | -0.0706 | 0.0047 | 0.0216 | 0.0259 | 0.5200 | 0.0321 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0043 | 0.0055 | -0.0001 | 0.0033 | 0.0184 | 0.0233 | 0.5056 | 0.0055 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0110 | 0.0288 | -0.0183 | 0.0032 | 0.0179 | 0.0240 | 0.5138 | 0.0254 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0048 | 0.0071 | -0.6798 | 0.0048 | 0.0223 | 0.0254 | 0.5056 | 0.0074 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0109 | 0.0272 | -0.0767 | 0.0035 | 0.0184 | 0.0240 | 0.5162 | 0.0262 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | -0.0030 | -0.0008 | -0.1076 | 0.0045 | 0.0221 | 0.0251 | 0.4985 | -0.0031 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0041 | 0.0106 | -0.0000 | 0.0050 | 0.0180 | 0.0243 | 0.5068 | 0.0111 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0012 | 0.0252 | -0.0177 | 0.0050 | 0.0218 | 0.0260 | 0.5102 | 0.0232 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0036 | 0.0013 | -0.7517 | 0.0052 | 0.0199 | 0.0251 | 0.4986 | 0.0000 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0074 | 0.0275 | -0.0656 | 0.0048 | 0.0240 | 0.0259 | 0.5146 | 0.0260 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | -0.0031 | 0.0141 | -0.0967 | 0.0032 | 0.0212 | 0.0241 | 0.5040 | 0.0112 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
