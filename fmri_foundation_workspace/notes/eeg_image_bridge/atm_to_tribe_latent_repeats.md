# Repeated ATM to Low-Rank TRIBE Latent Splits

Targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Images: `200`; train per split: `150`; val per split: `50`
Components: `32`; repeats: `30`; alpha: `100.0`

| model | latent rank pct | latent shifted | latent gap | surface rank pct | surface shifted | surface gap | surface r | shifted surface r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 0.9616 +/- 0.0093 | 0.4905 +/- 0.0401 | 0.4711 | 0.8203 +/- 0.0291 | 0.4915 +/- 0.0258 | 0.3288 | 0.9272 | 0.8696 |
| clip_image_ceiling | 0.8491 +/- 0.0224 | 0.5094 +/- 0.0457 | 0.3397 | 0.7253 +/- 0.0255 | 0.4981 +/- 0.0270 | 0.2271 | 0.8817 | 0.8400 |
