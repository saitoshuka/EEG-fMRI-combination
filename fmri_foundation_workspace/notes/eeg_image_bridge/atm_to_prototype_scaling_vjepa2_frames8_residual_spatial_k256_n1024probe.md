# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames8_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames8_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames8_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames8_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5598 | 0.5033 | 0.0565 | 0.5969 | 0.5060 | 0.0909 |
| clip_image_ceiling | 1024 | 0.5245 | 0.4584 | 0.0662 | 0.5138 | 0.4743 | 0.0394 |
