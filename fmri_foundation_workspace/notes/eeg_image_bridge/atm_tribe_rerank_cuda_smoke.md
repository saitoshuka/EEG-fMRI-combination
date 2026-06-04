# ATM Retrieval Reranked With TRIBE

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_n256.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Components: `32`; explained variance: `0.9978`
Device: `cuda`
Subjects: `10`; train images: `256`; test images: `200`

The frozen ATM embedding retrieves CLIP image candidates first. The TRIBE score only reranks the top-k candidate set.

| kind | top-k | TRIBE weight | top1 | top5 | top10 | rank pct | mean rank | null p(top1 >= real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.00 | 0.5200 | 0.8550 | 0.9250 | 0.9854 | 3.9050 |  |
| real_tribe_rerank | 10 | 0.30 | 0.5650 | 0.8800 | 0.9250 | 0.9861 | 3.7600 |  |
| shifted_tribe_rerank | 10 | 0.30 | 0.4900 | 0.8150 | 0.9250 | 0.9842 | 4.1450 |  |
| permutation_null | 10 | 0.30 | 0.4925 | 0.8225 | 0.9250 | 0.9843 | 4.1200 | 0.0 |

## Readout

- This does not change the ATM retrieval embedding.
- Real TRIBE reranking improves top1 over the frozen ATM baseline in the tested settings.
- Shifted and permutation-null TRIBE scores hurt or fail to match the real rerank result, which suggests the gain depends on image-specific EEG-to-TRIBE alignment.
- Because the rerank weights are still chosen from a small grid, this should be repeated with a stronger validation protocol before becoming a paper number.
