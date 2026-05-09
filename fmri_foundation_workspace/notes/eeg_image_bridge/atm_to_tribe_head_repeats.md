# Repeated ATM to TRIBE Splits

Targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Images: `200`; train per split: `150`; val per split: `50`
Repeats: `30`; seed: `33`; ridge alpha: `100.0`

| model | rank pct | shifted rank pct | rank gap | diag-offdiag | shifted diag-offdiag | spatial r | shifted spatial r | spatial gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 0.9088 +/- 0.0152 | 0.4884 +/- 0.0461 | 0.4204 | 0.2432 | -0.0030 | 0.2692 | 0.0358 | 0.2335 |
| clip_image_ceiling | 0.8073 +/- 0.0202 | 0.5030 +/- 0.0444 | 0.3043 | 0.1670 | 0.0048 | 0.1914 | 0.0317 | 0.1596 |
