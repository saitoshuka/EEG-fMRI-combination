# Repeated ATM to TRIBE Splits

Targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n64.npz`
Images: `64`; train per split: `48`; val per split: `16`
Repeats: `20`; seed: `33`; ridge alpha: `100.0`

| model | rank pct | shifted rank pct | rank gap | diag-offdiag | shifted diag-offdiag | spatial r | shifted spatial r | spatial gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 0.8788 +/- 0.0356 | 0.4800 +/- 0.0823 | 0.3988 | 0.2015 | 0.0042 | 0.2345 | 0.0463 | 0.1882 |
| clip_image_ceiling | 0.7167 +/- 0.0556 | 0.4752 +/- 0.0535 | 0.2415 | 0.1237 | -0.0011 | 0.1283 | -0.0091 | 0.1374 |
