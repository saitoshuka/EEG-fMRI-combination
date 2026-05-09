# ATM to TRIBE Head

Targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n64.npz`
Images: `64` (`48` train / `16` val)
Subjects: `10`
Ridge alpha: `100.0`

| model | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag | spatial r | shifted spatial r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atm_eeg_mean_subject | 0.5000 | 0.8750 | 0.9083 | 0.6583 | 0.2484 | 0.0867 | 0.3047 | 0.1396 |
| clip_image_ceiling | 0.1875 | 0.7500 | 0.7833 | 0.5125 | 0.1575 | 0.0415 | 0.1544 | 0.0278 |

## Readout

- This is a small sanity alignment on held-out images from the first 64 THINGS test images.
- Splitting is by image, so subject repeats of the same image do not leak from train to validation.
- `clip_image_ceiling` is not an EEG model; it tests whether a visual embedding can map to the TRIBE cortical target under the same split.
