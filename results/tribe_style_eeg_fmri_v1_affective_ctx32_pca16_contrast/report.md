# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/labram_retrieval_schaefer100_affective/features.npz`
- Context steps: 32
- Target space: pca16
- Target variance retained: 0.7551
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0062 | -0.0031 | -0.0000 | 0.0045 | 0.0199 | 0.0245 | 0.4985 | -0.0024 |
| 1 | ridge_last | real | 9687 | 5376 | 0.0059 | 0.0299 | -0.0285 | 0.0054 | 0.0214 | 0.0261 | 0.5134 | 0.0290 |
| 1 | ridge_context_mean | real | 9687 | 5376 | 0.0053 | 0.0038 | -0.6533 | 0.0033 | 0.0193 | 0.0231 | 0.4994 | 0.0015 |
| 1 | longctx_lowrank_transformer | real | 9687 | 5376 | 0.0096 | 0.0378 | -0.0981 | 0.0041 | 0.0223 | 0.0254 | 0.5171 | 0.0362 |
| 1 | longctx_lowrank_transformer | shifted_null | 9687 | 5376 | 0.0022 | 0.0144 | -0.1341 | 0.0054 | 0.0238 | 0.0263 | 0.5062 | 0.0114 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
