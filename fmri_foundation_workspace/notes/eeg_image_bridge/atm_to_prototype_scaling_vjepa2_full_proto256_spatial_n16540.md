# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_full_proto256_spatial_n16540/visual_roi_targets_vjepa2_full_proto256_spatial_n16540_residual_train_n16540.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/vjepa2_full_proto256_spatial_n16540/visual_roi_targets_vjepa2_full_proto256_spatial_n16540_residual_test_n200.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.5140 | 0.5091 | 0.0049 | 0.5435 | 0.4951 | 0.0484 |
| clip_image_ceiling | 1024 | 0.5584 | 0.4693 | 0.0891 | 0.5538 | 0.4690 | 0.0848 |
| atm_eeg_mean_subject | 4096 | 0.5545 | 0.5179 | 0.0366 | 0.5779 | 0.5106 | 0.0673 |
| clip_image_ceiling | 4096 | 0.5828 | 0.4997 | 0.0831 | 0.5733 | 0.5102 | 0.0631 |
| atm_eeg_mean_subject | 8192 | 0.5712 | 0.5341 | 0.0370 | 0.5927 | 0.5316 | 0.0612 |
| clip_image_ceiling | 8192 | 0.5835 | 0.4741 | 0.1094 | 0.5759 | 0.4773 | 0.0986 |
| atm_eeg_mean_subject | 16540 | 0.5790 | 0.5420 | 0.0370 | 0.6011 | 0.5413 | 0.0598 |
| clip_image_ceiling | 16540 | 0.5930 | 0.4902 | 0.1029 | 0.5959 | 0.4964 | 0.0995 |
