# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames32_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames32_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_frames32_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_frames32_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5594 | 0.4964 | 0.0630 | 0.5908 | 0.5002 | 0.0905 |
| clip_image_ceiling | 1024 | 0.5201 | 0.4669 | 0.0532 | 0.5059 | 0.4856 | 0.0203 |
