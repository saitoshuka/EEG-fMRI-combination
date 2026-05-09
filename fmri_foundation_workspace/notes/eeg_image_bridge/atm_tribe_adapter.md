# ATM TRIBE Post-Training Adapter

Train targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_n256.npz`
Test targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Device: `cuda`
Components: `32`; explained variance: `0.9978`
Subjects: `10`; train images: `256`; test images: `200`

The ATM EEG embeddings are frozen. This trains only a small residual adapter plus latent prediction head.

| model | selection | val TRIBE rank pct | test CLIP image top1 | top5 | rank pct | test TRIBE latent rank pct | shifted |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge_latent_all256 | closed_form | 0.9643 | 0.5200 | 0.8550 | 0.9854 | 0.9561 | 0.4821 |
| frozen_atm_embedding | no_training | 0.0000 | 0.5200 | 0.8550 | 0.9854 | 0.9561 | 0.4821 |
| linear_tribe_head | train_split_val | 0.9133 | 0.5200 | 0.8550 | 0.9854 | 0.9055 | 0.4848 |
| adapter_mse_b32 | train_split_val | 0.9051 | 0.3250 | 0.6650 | 0.9676 | 0.9099 | 0.4891 |
| adapter_contrast_b32 | train_split_val | 0.8976 | 0.3850 | 0.7250 | 0.9736 | 0.9079 | 0.4848 |
| adapter_contrast_b64 | train_split_val | 0.9141 | 0.3900 | 0.7500 | 0.9723 | 0.9162 | 0.4839 |
| adapter_clip_only_b32 | train_split_val | 0.9133 | 0.4800 | 0.7600 | 0.9789 | 0.9154 | 0.4865 |
| adapter_clip_tribe_b32 | train_split_val | 0.8965 | 0.5000 | 0.7700 | 0.9780 | 0.9109 | 0.4817 |
| adapter_clip_tribe_b64 | train_split_val | 0.8988 | 0.4750 | 0.7550 | 0.9783 | 0.9084 | 0.4863 |
| linear_tribe_head_retrain_all256 | best_by_train_val_clip:linear_tribe_head | 0.9698 | 0.5200 | 0.8550 | 0.9854 | 0.9186 | 0.4876 |

## Readout

- `frozen_atm_embedding` is the no-training retrieval baseline: mean-subject ATM EEG embedding directly retrieves CLIP image/text features.
- `ridge_latent_all256` is the closed-form baseline for TRIBE latent prediction; its CLIP retrieval is still the frozen ATM embedding.
- Adapter rows are trained with frozen ATM embeddings and evaluated on fixed, unseen test200 images.
- The key retrieval question is whether `test_clip_image_top1/top5/rank_percentile` improves over `frozen_atm_embedding`.
- The TRIBE columns check whether any retrieval gain still preserves brain-space alignment.

## Conclusion

The post-training TRIBE adapter did not improve CLIP image retrieval on unseen
test200 images in this run.

The frozen ATM baseline is already strong:

```text
top1 = 0.5200
top5 = 0.8550
rank percentile = 0.9854
```

The only configuration selected by train-validation retrieval was
`linear_tribe_head`, which does not alter the embedding and therefore keeps
retrieval exactly unchanged. Every configuration that actually changes the EEG
embedding reduces image retrieval on test200, even when CLIP preservation is
included.

This does not mean TRIBE is useless. It means the first safe use of TRIBE should
probably be an auxiliary brain-space head or adapter branch, not a direct
modification of the retrieval embedding. In other words:

```text
Keep ATM/CLIP retrieval embedding stable.
Add TRIBE latent head as spatial regularizer / auxiliary output.
Only let TRIBE affect the shared representation after stronger controls and
more TRIBE-labeled train images.
```

An exploratory residual CLIP calibration over all THINGS train images found a
tiny possible test-grid improvement (`top1 0.52 -> 0.54`), but validation was
too saturated to select that setting honestly. Treat this as a hint, not a
claim.
