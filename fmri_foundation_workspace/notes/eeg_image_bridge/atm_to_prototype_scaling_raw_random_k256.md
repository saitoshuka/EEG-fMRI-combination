# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_random_targets_train_n16540_k256.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_random_targets_test_n200_k256.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.7979 | 0.5162 | 0.2817 | 0.9175 | 0.4939 | 0.4236 |
| clip_image_ceiling | 1024 | 0.7414 | 0.5187 | 0.2227 | 0.8761 | 0.5014 | 0.3747 |
| atm_eeg_mean_subject | 4096 | 0.7968 | 0.5195 | 0.2773 | 0.9132 | 0.4991 | 0.4141 |
| clip_image_ceiling | 4096 | 0.7854 | 0.5179 | 0.2675 | 0.9201 | 0.5064 | 0.4137 |
| atm_eeg_mean_subject | 8192 | 0.8006 | 0.5206 | 0.2799 | 0.9138 | 0.4976 | 0.4162 |
| clip_image_ceiling | 8192 | 0.8030 | 0.5078 | 0.2952 | 0.9287 | 0.5007 | 0.4280 |
| atm_eeg_mean_subject | 16540 | 0.7983 | 0.5192 | 0.2791 | 0.9135 | 0.4972 | 0.4162 |
| clip_image_ceiling | 16540 | 0.8160 | 0.5159 | 0.3001 | 0.9345 | 0.4985 | 0.4361 |
