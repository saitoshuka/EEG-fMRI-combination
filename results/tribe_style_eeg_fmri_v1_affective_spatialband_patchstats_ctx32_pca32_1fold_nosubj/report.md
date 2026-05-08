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

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
