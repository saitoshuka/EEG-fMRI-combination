# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_proto256_spatial_n1024probe/visual_roi_targets_clip_vith14_proto256_spatial_n1024probe_residual_train_n1024.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/clip_vith14_proto256_spatial_n1024probe/visual_roi_targets_clip_vith14_proto256_spatial_n1024probe_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.7305 | 0.4990 | 0.2314 | 0.7912 | 0.5072 | 0.2840 |
| clip_image_ceiling | 1024 | 0.4842 | 0.4472 | 0.0370 | 0.4527 | 0.4432 | 0.0095 |
