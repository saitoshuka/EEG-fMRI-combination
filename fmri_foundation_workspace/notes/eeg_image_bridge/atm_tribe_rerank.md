# ATM Retrieval Reranked With TRIBE

Train targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_n256.npz`
Test targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Components: `32`; explained variance: `0.9978`
Subjects: `10`; train images: `256`; test images: `200`

The frozen ATM embedding retrieves CLIP image candidates first. The TRIBE score only reranks the top-k candidate set.

| kind | top-k | TRIBE weight | top1 | top5 | top10 | rank pct | mean rank | null p(top1 >= real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.00 | 0.5200 | 0.8550 | 0.9250 | 0.9854 | 3.9050 |  |
| real_tribe_rerank | 10 | 0.50 | 0.5950 | 0.8750 | 0.9250 | 0.9863 | 3.7250 |  |
| shifted_tribe_rerank | 10 | 0.50 | 0.4100 | 0.7900 | 0.9250 | 0.9826 | 4.4550 |  |
| permutation_null | 10 | 0.50 | 0.4219 | 0.8006 | 0.9250 | 0.9828 | 4.4220 | 0.0 |
| real_tribe_rerank | 10 | 0.70 | 0.6000 | 0.8600 | 0.9250 | 0.9862 | 3.7550 |  |
| shifted_tribe_rerank | 10 | 0.70 | 0.3600 | 0.7850 | 0.9250 | 0.9814 | 4.7000 |  |
| permutation_null | 10 | 0.70 | 0.3571 | 0.7683 | 0.9250 | 0.9813 | 4.7247 | 0.0 |
| real_tribe_rerank | 20 | 0.50 | 0.5950 | 0.8750 | 0.9450 | 0.9871 | 3.5600 |  |
| shifted_tribe_rerank | 20 | 0.50 | 0.4100 | 0.7700 | 0.8900 | 0.9807 | 4.8400 |  |
| permutation_null | 20 | 0.50 | 0.4190 | 0.7696 | 0.8967 | 0.9811 | 4.7649 | 0.0 |
| real_tribe_rerank | 100 | 0.50 | 0.5950 | 0.8750 | 0.9450 | 0.9895 | 3.0900 |  |
| shifted_tribe_rerank | 100 | 0.50 | 0.4100 | 0.7450 | 0.8750 | 0.9778 | 5.4150 |  |
| permutation_null | 100 | 0.50 | 0.4188 | 0.7625 | 0.8721 | 0.9777 | 5.4300 | 0.0 |
| real_tribe_rerank | 100 | 0.70 | 0.6000 | 0.8650 | 0.9450 | 0.9895 | 3.0850 |  |
| shifted_tribe_rerank | 100 | 0.70 | 0.3500 | 0.6600 | 0.8100 | 0.9671 | 7.5400 |  |
| permutation_null | 100 | 0.70 | 0.3377 | 0.6700 | 0.8023 | 0.9683 | 7.3149 | 0.0 |

## Readout

- This does not change the ATM retrieval embedding.
- Real TRIBE reranking improves top1 over the frozen ATM baseline in the tested settings.
- Shifted and permutation-null TRIBE scores hurt or fail to match the real rerank result, which suggests the gain depends on image-specific EEG-to-TRIBE alignment.
- Because the rerank weights are still chosen from a small grid, this should be repeated with a stronger validation protocol before becoming a paper number.
