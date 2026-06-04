# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_random_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_random_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_alpha_search_proto256_random_n16540/visual_roi_targets_clip_vith14_alpha_search_proto256_random_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.6104 | 0.4956 | 0.1147 | 0.6702 | 0.4708 | 0.1993 |
| clip_image_ceiling | 1024 | 0.5202 | 0.5028 | 0.0174 | 0.5390 | 0.5007 | 0.0382 |
| atm_eeg_mean_subject | 4096 | 0.6157 | 0.4988 | 0.1169 | 0.6684 | 0.4789 | 0.1895 |
| clip_image_ceiling | 4096 | 0.5374 | 0.5445 | -0.0071 | 0.5662 | 0.5315 | 0.0347 |
| atm_eeg_mean_subject | 8192 | 0.6111 | 0.4876 | 0.1235 | 0.6734 | 0.4769 | 0.1965 |
| clip_image_ceiling | 8192 | 0.4930 | 0.5244 | -0.0315 | 0.5032 | 0.5021 | 0.0011 |
| atm_eeg_mean_subject | 16540 | 0.6221 | 0.4871 | 0.1350 | 0.6844 | 0.4750 | 0.2094 |
| clip_image_ceiling | 16540 | 0.5282 | 0.5138 | 0.0144 | 0.5723 | 0.5115 | 0.0609 |
