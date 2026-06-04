# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_targets_train_n16540_k256.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_targets_test_n200_k256.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.8817 | 0.4854 | 0.3963 | 0.9458 | 0.4864 | 0.4594 |
| clip_image_ceiling | 1024 | 0.8481 | 0.5038 | 0.3443 | 0.9088 | 0.4879 | 0.4209 |
| atm_eeg_mean_subject | 4096 | 0.8825 | 0.4894 | 0.3931 | 0.9438 | 0.4883 | 0.4555 |
| clip_image_ceiling | 4096 | 0.8759 | 0.5054 | 0.3706 | 0.9460 | 0.5038 | 0.4422 |
| atm_eeg_mean_subject | 8192 | 0.8834 | 0.4891 | 0.3943 | 0.9442 | 0.4874 | 0.4568 |
| clip_image_ceiling | 8192 | 0.8851 | 0.4992 | 0.3858 | 0.9525 | 0.5000 | 0.4525 |
| atm_eeg_mean_subject | 16540 | 0.8816 | 0.4877 | 0.3939 | 0.9443 | 0.4857 | 0.4587 |
| clip_image_ceiling | 16540 | 0.8905 | 0.4943 | 0.3961 | 0.9576 | 0.4951 | 0.4625 |
