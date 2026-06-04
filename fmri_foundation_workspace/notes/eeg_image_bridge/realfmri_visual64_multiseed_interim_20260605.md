# Real-fMRI Visual64 Multiseed Interim

Date: 2026-06-05

## Status

The direct real THINGS-fMRI visual64 experiment is being repeated across seeds.
This is an interim checkpoint while the queued pooled and seed77 runs are still
running.

Completed best-checkpoint runs so far:

| head | seed | visual64 rank | shifted | delta | top1 | top5 | image corr | ROI corr | query identity diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query | 11 | 0.7244 | 0.4833 | 0.2411 | 0.0519 | 0.2338 | 0.0505 | 0.1506 | 0.1278 |
| query | 33 | 0.7384 | 0.4838 | 0.2546 | 0.0779 | 0.1818 | 0.0525 | 0.1488 | 0.1315 |
| pooled | 33 | 0.7739 | 0.4836 | 0.2903 | 0.0649 | 0.2468 | 0.0619 | 0.0932 | 0.0896 |

Interim query aggregate:

| head | seeds | rank mean +/- sd | shifted mean | delta mean | image corr mean | ROI corr mean | identity diag-offdiag mean |
|---|---|---:|---:|---:|---:|---:|---:|
| query | 11,33 | 0.7314 +/- 0.0099 | 0.4835 | 0.2479 | 0.0515 | 0.1497 | 0.1296 |

## Interpretation

The ordered-query real visual64 signal is not a one-seed accident: seed11 and
seed33 give very similar rank, shifted-null gap, ROI-wise correlation, and
query-target identity. This strengthens the real-fMRI grounding claim for
visual cortex.

This is not yet a final query-vs-pooled conclusion because pooled has only
completed seed33 so far. The current scalar-rank comparison remains:

- query is robust across two seeds;
- pooled is stronger on seed33 scalar rank;
- query remains stronger on ROI identity/interpretability.

The queued `atm_real_fmri_visual64_multiseed` run is continuing with pooled
seed11, then query/pooled seed77.

## Artifacts

- Multiseed summarizer:
  `fmri_foundation_workspace/scripts/summarize_atm_real_fmri_visual64_multiseed.py`
- Interim summary JSON:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed.json`
- Interim aggregate CSV:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed_aggregate.csv`
- Running tmux session:
  `atm_real_fmri_visual64_multiseed`
