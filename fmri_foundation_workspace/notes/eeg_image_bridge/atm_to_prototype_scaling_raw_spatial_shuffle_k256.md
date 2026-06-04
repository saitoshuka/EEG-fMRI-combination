# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_shuffle_targets_train_n16540_k256.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_shuffle_targets_test_n200_k256.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.7969 | 0.5167 | 0.2803 | 0.9175 | 0.4937 | 0.4238 |
| clip_image_ceiling | 1024 | 0.7413 | 0.5196 | 0.2217 | 0.8777 | 0.5010 | 0.3767 |
| atm_eeg_mean_subject | 4096 | 0.7956 | 0.5204 | 0.2752 | 0.9132 | 0.4983 | 0.4149 |
| clip_image_ceiling | 4096 | 0.7837 | 0.5184 | 0.2653 | 0.9213 | 0.5073 | 0.4140 |
| atm_eeg_mean_subject | 8192 | 0.7989 | 0.5222 | 0.2767 | 0.9137 | 0.4975 | 0.4162 |
| clip_image_ceiling | 8192 | 0.8005 | 0.5091 | 0.2914 | 0.9295 | 0.5012 | 0.4283 |
| atm_eeg_mean_subject | 16540 | 0.7968 | 0.5201 | 0.2767 | 0.9139 | 0.4966 | 0.4172 |
| clip_image_ceiling | 16540 | 0.8141 | 0.5159 | 0.2983 | 0.9356 | 0.4981 | 0.4374 |
