# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_pca_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_pca_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_pca_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_pca_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.7834 | 0.5030 | 0.2804 | 0.6338 | 0.5128 | 0.1210 |
| clip_image_ceiling | 1024 | 0.5351 | 0.4960 | 0.0391 | 0.5104 | 0.4875 | 0.0228 |
| atm_eeg_mean_subject | 4096 | 0.8723 | 0.5160 | 0.3563 | 0.7208 | 0.5322 | 0.1886 |
| clip_image_ceiling | 4096 | 0.5663 | 0.5324 | 0.0339 | 0.5380 | 0.5376 | 0.0005 |
| atm_eeg_mean_subject | 8192 | 0.9148 | 0.5088 | 0.4060 | 0.7348 | 0.5123 | 0.2225 |
| clip_image_ceiling | 8192 | 0.5930 | 0.5383 | 0.0547 | 0.5342 | 0.5254 | 0.0088 |
| atm_eeg_mean_subject | 16540 | 0.9322 | 0.5146 | 0.4176 | 0.7545 | 0.5114 | 0.2430 |
| clip_image_ceiling | 16540 | 0.7680 | 0.4755 | 0.2926 | 0.6430 | 0.4840 | 0.1589 |
