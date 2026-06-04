# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_proto256_spatial_n1024probe/visual_roi_targets_vjepa2_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5671 | 0.4923 | 0.0748 | 0.6003 | 0.4971 | 0.1032 |
| clip_image_ceiling | 1024 | 0.5069 | 0.4650 | 0.0419 | 0.5032 | 0.4906 | 0.0126 |
