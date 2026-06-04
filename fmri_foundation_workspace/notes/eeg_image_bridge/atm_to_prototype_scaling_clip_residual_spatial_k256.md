# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_spatial_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_spatial_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_spatial_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_spatial_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.6398 | 0.4821 | 0.1577 | 0.6739 | 0.4845 | 0.1894 |
| clip_image_ceiling | 1024 | 0.5481 | 0.4846 | 0.0635 | 0.5523 | 0.4843 | 0.0680 |
| atm_eeg_mean_subject | 4096 | 0.6358 | 0.4858 | 0.1499 | 0.6887 | 0.4848 | 0.2039 |
| clip_image_ceiling | 4096 | 0.5339 | 0.5312 | 0.0028 | 0.5518 | 0.5149 | 0.0368 |
| atm_eeg_mean_subject | 8192 | 0.6375 | 0.4836 | 0.1538 | 0.6920 | 0.4856 | 0.2064 |
| clip_image_ceiling | 8192 | 0.4868 | 0.4921 | -0.0053 | 0.5109 | 0.4886 | 0.0223 |
| atm_eeg_mean_subject | 16540 | 0.6489 | 0.4826 | 0.1662 | 0.7024 | 0.4819 | 0.2205 |
| clip_image_ceiling | 16540 | 0.5485 | 0.5053 | 0.0433 | 0.5751 | 0.5029 | 0.0722 |
