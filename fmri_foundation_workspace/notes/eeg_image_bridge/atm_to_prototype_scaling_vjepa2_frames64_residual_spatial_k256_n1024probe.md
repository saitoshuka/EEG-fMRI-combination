# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames64_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames64_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames64_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames64_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5572 | 0.4890 | 0.0682 | 0.5930 | 0.4926 | 0.1004 |
| clip_image_ceiling | 1024 | 0.5071 | 0.4685 | 0.0386 | 0.4978 | 0.4863 | 0.0116 |
