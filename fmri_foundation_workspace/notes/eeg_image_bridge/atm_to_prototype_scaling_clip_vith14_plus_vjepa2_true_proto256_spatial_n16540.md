# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_plus_vjepa2_true_proto256_spatial_n16540/visual_roi_targets_clip_vith14_plus_vjepa2_true_proto256_spatial_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_plus_vjepa2_true_proto256_spatial_n16540/visual_roi_targets_clip_vith14_plus_vjepa2_true_proto256_spatial_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5277 | 0.5035 | 0.0243 | 0.5477 | 0.4993 | 0.0484 |
| clip_image_ceiling | 1024 | 0.5359 | 0.4626 | 0.0733 | 0.5307 | 0.4621 | 0.0686 |
| atm_eeg_mean_subject | 4096 | 0.5589 | 0.5179 | 0.0410 | 0.5770 | 0.5110 | 0.0660 |
| clip_image_ceiling | 4096 | 0.5231 | 0.5207 | 0.0024 | 0.5293 | 0.5309 | -0.0016 |
| atm_eeg_mean_subject | 8192 | 0.5607 | 0.5387 | 0.0220 | 0.5764 | 0.5276 | 0.0488 |
| clip_image_ceiling | 8192 | 0.5107 | 0.4940 | 0.0167 | 0.4983 | 0.4948 | 0.0035 |
| atm_eeg_mean_subject | 16540 | 0.5693 | 0.5466 | 0.0228 | 0.5895 | 0.5369 | 0.0526 |
| clip_image_ceiling | 16540 | 0.5231 | 0.4937 | 0.0294 | 0.5115 | 0.4818 | 0.0297 |
