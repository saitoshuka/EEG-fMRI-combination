# ATM to TRIBE Train-Size Scaling

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_pca_targets_train_n16540_k256.npz`
Test targets: `fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_pca_targets_test_n200_k256.npz`
Components: `32`; alpha: `100.0`
Latent basis: PCA fit once on all extracted train targets, then reused for every train size.

| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 1024 | 0.9551 | 0.4864 | 0.4687 | 0.9522 | 0.4861 | 0.4661 |
| clip_image_ceiling | 1024 | 0.9322 | 0.4892 | 0.4430 | 0.9216 | 0.4902 | 0.4314 |
| atm_eeg_mean_subject | 4096 | 0.9528 | 0.4890 | 0.4638 | 0.9501 | 0.4884 | 0.4616 |
| clip_image_ceiling | 4096 | 0.9626 | 0.5052 | 0.4574 | 0.9545 | 0.5044 | 0.4501 |
| atm_eeg_mean_subject | 8192 | 0.9534 | 0.4877 | 0.4657 | 0.9504 | 0.4872 | 0.4633 |
| clip_image_ceiling | 8192 | 0.9682 | 0.5003 | 0.4679 | 0.9602 | 0.4999 | 0.4603 |
| atm_eeg_mean_subject | 16540 | 0.9539 | 0.4870 | 0.4669 | 0.9507 | 0.4866 | 0.4641 |
| clip_image_ceiling | 16540 | 0.9719 | 0.4954 | 0.4764 | 0.9646 | 0.4951 | 0.4695 |
