# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_summary_bandpower_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0029 | 0.0204 | 0.0009 | 0.0048 | 0.0223 | 0.0255 | 0.5105 | 0.0223 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0038 | 0.0059 | 0.0001 | 0.0052 | 0.0233 | 0.0263 | 0.5022 | 0.0042 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0073 | 0.0061 | -0.0012 | 0.0045 | 0.0227 | 0.0260 | 0.5095 | 0.0044 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | 0.0015 | 0.0016 | -0.0013 | 0.0050 | 0.0216 | 0.0250 | 0.5017 | 0.0006 |
| 2 | train_mean | real | 9751 | 5374 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9751 | 5374 | 0.0043 | 0.0055 | -0.0001 | 0.0033 | 0.0184 | 0.0233 | 0.5056 | 0.0055 |
| 2 | ridge_last | real | 9751 | 5374 | 0.0059 | 0.0156 | 0.0005 | 0.0047 | 0.0201 | 0.0248 | 0.5055 | 0.0112 |
| 2 | ridge_context_mean | real | 9751 | 5374 | 0.0014 | 0.0035 | -0.0001 | 0.0052 | 0.0216 | 0.0256 | 0.5019 | 0.0011 |
| 2 | longctx_lowrank_transformer | real | 9751 | 5374 | 0.0087 | 0.0048 | -0.0013 | 0.0041 | 0.0190 | 0.0250 | 0.5124 | 0.0045 |
| 2 | longctx_lowrank_transformer | shifted_null | 9751 | 5374 | -0.0100 | -0.0002 | -0.0018 | 0.0041 | 0.0199 | 0.0242 | 0.4944 | -0.0033 |
| 3 | train_mean | real | 9470 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9470 | 5376 | 0.0041 | 0.0106 | -0.0000 | 0.0050 | 0.0180 | 0.0243 | 0.5068 | 0.0111 |
| 3 | ridge_last | real | 9470 | 5376 | 0.0053 | 0.0270 | 0.0018 | 0.0032 | 0.0206 | 0.0248 | 0.5128 | 0.0218 |
| 3 | ridge_context_mean | real | 9470 | 5376 | 0.0018 | -0.0024 | 0.0000 | 0.0032 | 0.0199 | 0.0232 | 0.5017 | 0.0040 |
| 3 | longctx_lowrank_transformer | real | 9470 | 5376 | 0.0127 | 0.0107 | -0.0008 | 0.0047 | 0.0199 | 0.0262 | 0.5193 | 0.0095 |
| 3 | longctx_lowrank_transformer | shifted_null | 9470 | 5376 | -0.0008 | 0.0003 | -0.0013 | 0.0041 | 0.0216 | 0.0245 | 0.5002 | -0.0005 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
