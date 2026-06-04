# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_plus_vjepa2_proto256_spatial_n1024probe/visual_roi_targets_clip_vith14_plus_vjepa2_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_plus_vjepa2_proto256_spatial_n1024probe/visual_roi_targets_clip_vith14_plus_vjepa2_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.6348 | 0.4917 | 0.1430 | 0.6865 | 0.4869 | 0.1996 |
| clip_image_ceiling | 1024 | 0.4444 | 0.4465 | -0.0020 | 0.4130 | 0.4638 | -0.0509 |
