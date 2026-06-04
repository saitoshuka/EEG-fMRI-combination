# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames16_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames16_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames16_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames16_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5542 | 0.4967 | 0.0576 | 0.5938 | 0.4973 | 0.0964 |
| clip_image_ceiling | 1024 | 0.5258 | 0.4725 | 0.0533 | 0.5089 | 0.4917 | 0.0172 |
