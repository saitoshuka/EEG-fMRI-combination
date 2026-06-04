# Test-Split CV ATM TRIBE Rerank

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz`
Splits: `20`; validation images per split: `100`; device: `cuda`

For each split, top-k/weight is selected on one half of the 200 test images and evaluated on the held-out half.

| metric | baseline | selected rerank | gain |
|---|---:|---:|---:|
| top1 | 0.6280 ± 0.0372 | 0.6665 ± 0.0357 | 0.0385 ± 0.0262 |
| top5 | 0.9230 ± 0.0166 | 0.9440 ± 0.0188 | 0.0210 ± 0.0168 |
| rank pct | 0.9855 ± 0.0033 | 0.9883 ± 0.0036 | 0.0029 ± 0.0012 |

Selection counts: `{'100:0.3': 9, '100:0.5': 6, '20:0.5': 3, '20:0.3': 1, '10:0.3': 1}`

Readout: this is a small-sample diagnostic, not a final test-set number.
