# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_spatial_shuffle_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_spatial_shuffle_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_spatial_shuffle_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_spatial_shuffle_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.6106 | 0.4971 | 0.1135 | 0.6677 | 0.4736 | 0.1941 |
| clip_image_ceiling | 1024 | 0.5237 | 0.5066 | 0.0171 | 0.5410 | 0.4969 | 0.0441 |
| atm_eeg_mean_subject | 4096 | 0.6156 | 0.4965 | 0.1192 | 0.6676 | 0.4793 | 0.1883 |
| clip_image_ceiling | 4096 | 0.5339 | 0.5441 | -0.0102 | 0.5614 | 0.5310 | 0.0304 |
| atm_eeg_mean_subject | 8192 | 0.6125 | 0.4871 | 0.1254 | 0.6731 | 0.4780 | 0.1951 |
| clip_image_ceiling | 8192 | 0.4920 | 0.5241 | -0.0321 | 0.5023 | 0.5019 | 0.0004 |
| atm_eeg_mean_subject | 16540 | 0.6231 | 0.4870 | 0.1361 | 0.6822 | 0.4778 | 0.2043 |
| clip_image_ceiling | 16540 | 0.5273 | 0.5141 | 0.0132 | 0.5713 | 0.5110 | 0.0603 |
