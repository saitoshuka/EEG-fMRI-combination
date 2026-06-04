# ATM Retrieval Reranked With TRIBE

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Components: `32`; explained variance: `0.9973`
Device: `cuda`
Subjects: `10`; train images: `16540`; test images: `200`

The frozen ATM embedding retrieves CLIP image candidates first. The TRIBE score only reranks the top-k candidate set.

| kind | top-k | TRIBE weight | top1 | top5 | top10 | rank pct | mean rank | null p(top1 >= real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.00 | 0.5200 | 0.8550 | 0.9250 | 0.9854 | 3.9050 |  |
| real_tribe_rerank | 10 | 0.30 | 0.5600 | 0.8800 | 0.9250 | 0.9863 | 3.7300 |  |
| shifted_tribe_rerank | 10 | 0.30 | 0.4650 | 0.8150 | 0.9250 | 0.9843 | 4.1200 |  |
| permutation_null | 10 | 0.30 | 0.4847 | 0.8351 | 0.9250 | 0.9844 | 4.1108 | 0.0 |
| real_tribe_rerank | 10 | 0.50 | 0.5650 | 0.8800 | 0.9250 | 0.9863 | 3.7350 |  |
| shifted_tribe_rerank | 10 | 0.50 | 0.4250 | 0.7950 | 0.9250 | 0.9830 | 4.3900 |  |
| permutation_null | 10 | 0.50 | 0.4269 | 0.8064 | 0.9250 | 0.9830 | 4.3928 | 0.0 |
| real_tribe_rerank | 20 | 0.30 | 0.5600 | 0.8900 | 0.9550 | 0.9874 | 3.5100 |  |
| shifted_tribe_rerank | 20 | 0.30 | 0.4650 | 0.8100 | 0.9200 | 0.9837 | 4.2450 |  |
| permutation_null | 20 | 0.30 | 0.4834 | 0.8296 | 0.9196 | 0.9839 | 4.1993 | 0.0 |
| real_tribe_rerank | 20 | 0.50 | 0.5650 | 0.8950 | 0.9550 | 0.9875 | 3.4950 |  |
| shifted_tribe_rerank | 20 | 0.50 | 0.4200 | 0.7850 | 0.8950 | 0.9814 | 4.7100 |  |
| permutation_null | 20 | 0.50 | 0.4259 | 0.7800 | 0.8976 | 0.9814 | 4.7107 | 0.0 |
| real_tribe_rerank | 100 | 0.30 | 0.5600 | 0.8900 | 0.9550 | 0.9892 | 3.1450 |  |
| shifted_tribe_rerank | 100 | 0.30 | 0.4650 | 0.8100 | 0.9100 | 0.9831 | 4.3550 |  |
| permutation_null | 100 | 0.30 | 0.4853 | 0.8285 | 0.9153 | 0.9835 | 4.2917 | 0.0 |
| real_tribe_rerank | 100 | 0.50 | 0.5650 | 0.8950 | 0.9600 | 0.9899 | 3.0100 |  |
| shifted_tribe_rerank | 100 | 0.50 | 0.4200 | 0.7600 | 0.8750 | 0.9787 | 5.2400 |  |
| permutation_null | 100 | 0.50 | 0.4234 | 0.7654 | 0.8766 | 0.9779 | 5.3932 | 0.0 |

## Readout

- This does not change the ATM retrieval embedding.
- Real TRIBE reranking improves top1 over the frozen ATM baseline in the tested settings.
- Shifted and permutation-null TRIBE scores hurt or fail to match the real rerank result, which suggests the gain depends on image-specific EEG-to-TRIBE alignment.
- Because the rerank weights are still chosen from a small grid, this should be repeated with a stronger validation protocol before becoming a paper number.
