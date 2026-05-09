# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/experience_spatial_bandpower.npz`
- Context steps: 16
- Target space: pca64
- Target variance retained: 0.9597
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 4902 | 1906 | nan | nan | -0.0006 | 0.0000 | 0.0000 | 0.0248 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 4902 | 1906 | -0.0475 | -0.0126 | -0.0006 | 0.0100 | 0.0540 | 0.0581 | 0.4894 | 0.0092 |
| 1 | ridge_last | time_residual | 4902 | 1906 | 0.0213 | 0.0521 | -0.0163 | 0.0184 | 0.0782 | 0.0724 | 0.5358 | 0.0589 |
| 1 | ridge_context_mean | time_residual | 4902 | 1906 | 0.0515 | 0.1029 | 0.0038 | 0.0215 | 0.1060 | 0.0855 | 0.5752 | 0.1167 |
| 1 | longctx_lowrank_transformer | time_residual | 4902 | 1906 | 0.0273 | 0.0405 | -0.0027 | 0.0205 | 0.0845 | 0.0785 | 0.5695 | 0.0449 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 4902 | 1906 | -0.0026 | -0.0079 | -0.0092 | 0.0089 | 0.0593 | 0.0583 | 0.4932 | -0.0094 |
| 2 | train_mean | time_residual | 4939 | 1824 | nan | nan | -0.0003 | 0.0000 | 0.0000 | 0.0259 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 4939 | 1824 | -0.0489 | -0.0345 | -0.0003 | 0.0137 | 0.0532 | 0.0586 | 0.4741 | -0.0206 |
| 2 | ridge_last | time_residual | 4939 | 1824 | 0.0210 | 0.0472 | -0.0093 | 0.0099 | 0.0762 | 0.0660 | 0.5318 | 0.0433 |
| 2 | ridge_context_mean | time_residual | 4939 | 1824 | 0.0444 | 0.0607 | -0.0253 | 0.0208 | 0.0888 | 0.0776 | 0.5379 | 0.0680 |
| 2 | longctx_lowrank_transformer | time_residual | 4939 | 1824 | 0.0220 | 0.0302 | -0.0037 | 0.0170 | 0.0806 | 0.0747 | 0.5451 | 0.0294 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 4939 | 1824 | 0.0070 | 0.0048 | -0.0078 | 0.0137 | 0.0685 | 0.0664 | 0.5051 | 0.0091 |
| 3 | train_mean | time_residual | 4896 | 1852 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0255 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 4896 | 1852 | -0.0535 | -0.0382 | -0.0001 | 0.0092 | 0.0529 | 0.0550 | 0.4700 | -0.0262 |
| 3 | ridge_last | time_residual | 4896 | 1852 | 0.0213 | 0.0576 | -0.0018 | 0.0151 | 0.0799 | 0.0719 | 0.5417 | 0.0604 |
| 3 | ridge_context_mean | time_residual | 4896 | 1852 | 0.0430 | 0.0759 | -0.0018 | 0.0211 | 0.0864 | 0.0785 | 0.5499 | 0.0822 |
| 3 | longctx_lowrank_transformer | time_residual | 4896 | 1852 | 0.0251 | 0.0312 | -0.0029 | 0.0205 | 0.0940 | 0.0804 | 0.5482 | 0.0363 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 4896 | 1852 | -0.0174 | -0.0166 | -0.0124 | 0.0103 | 0.0486 | 0.0535 | 0.4682 | -0.0188 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
