# ATM Retrieval Reranked With TRIBE

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_budget4096_faststill_n4096.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Components: `32`; explained variance: `0.9974`
Subjects: `10`; train images: `4096`; test images: `200`

The frozen ATM embedding retrieves CLIP image candidates first. The TRIBE score only reranks the top-k candidate set.

| kind | top-k | TRIBE weight | top1 | top5 | top10 | rank pct | mean rank | null p(top1 >= real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.00 | 0.5200 | 0.8550 | 0.9250 | 0.9854 | 3.9050 |  |
| real_tribe_rerank | 10 | 0.30 | 0.5550 | 0.8750 | 0.9250 | 0.9862 | 3.7400 |  |
| shifted_tribe_rerank | 10 | 0.30 | 0.4750 | 0.8200 | 0.9250 | 0.9843 | 4.1200 |  |
| permutation_null | 10 | 0.30 | 0.4846 | 0.8346 | 0.9250 | 0.9844 | 4.1119 | 0.0 |
| real_tribe_rerank | 10 | 0.50 | 0.5650 | 0.8750 | 0.9250 | 0.9863 | 3.7350 |  |
| shifted_tribe_rerank | 10 | 0.50 | 0.4150 | 0.8000 | 0.9250 | 0.9830 | 4.3800 |  |
| permutation_null | 10 | 0.50 | 0.4265 | 0.8063 | 0.9250 | 0.9829 | 4.3959 | 0.0 |
| real_tribe_rerank | 20 | 0.30 | 0.5550 | 0.8850 | 0.9550 | 0.9873 | 3.5300 |  |
| shifted_tribe_rerank | 20 | 0.30 | 0.4750 | 0.8150 | 0.9200 | 0.9837 | 4.2400 |  |
| permutation_null | 20 | 0.30 | 0.4845 | 0.8289 | 0.9190 | 0.9839 | 4.2012 | 0.0 |
| real_tribe_rerank | 20 | 0.50 | 0.5650 | 0.8900 | 0.9550 | 0.9874 | 3.5050 |  |
| shifted_tribe_rerank | 20 | 0.50 | 0.4100 | 0.7800 | 0.8950 | 0.9814 | 4.6950 |  |
| permutation_null | 20 | 0.50 | 0.4254 | 0.7792 | 0.8977 | 0.9813 | 4.7185 | 0.0 |
| real_tribe_rerank | 100 | 0.30 | 0.5550 | 0.8850 | 0.9550 | 0.9891 | 3.1600 |  |
| shifted_tribe_rerank | 100 | 0.30 | 0.4750 | 0.8150 | 0.9150 | 0.9833 | 4.3200 |  |
| permutation_null | 100 | 0.30 | 0.4867 | 0.8278 | 0.9143 | 0.9834 | 4.2998 | 0.0 |
| real_tribe_rerank | 100 | 0.50 | 0.5650 | 0.8900 | 0.9600 | 0.9898 | 3.0200 |  |
| shifted_tribe_rerank | 100 | 0.50 | 0.4100 | 0.7600 | 0.8700 | 0.9789 | 5.1950 |  |
| permutation_null | 100 | 0.50 | 0.4235 | 0.7655 | 0.8753 | 0.9779 | 5.3959 | 0.0 |

## Readout

- This does not change the ATM retrieval embedding.
- Real TRIBE reranking improves top1 over the frozen ATM baseline in the tested settings.
- Shifted and permutation-null TRIBE scores hurt or fail to match the real rerank result, which suggests the gain depends on image-specific EEG-to-TRIBE alignment.
- Because the rerank weights are still chosen from a small grid, this should be repeated with a stronger validation protocol before becoming a paper number.
