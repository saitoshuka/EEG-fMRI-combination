# Real-fMRI Visual64 Multiseed Interim

Date: 2026-06-05

## Status

The direct real THINGS-fMRI visual64 experiment is being repeated across seeds.
This is an interim checkpoint after seed11 and seed33 have completed for both
ordered-query and pooled heads. Seed77 is still running.

Completed best-checkpoint runs so far:

| head | seed | visual64 rank | shifted | delta | top1 | top5 | image corr | ROI corr | query identity diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query | 11 | 0.7244 | 0.4833 | 0.2411 | 0.0519 | 0.2338 | 0.0505 | 0.1506 | 0.1278 |
| query | 33 | 0.7384 | 0.4838 | 0.2546 | 0.0779 | 0.1818 | 0.0525 | 0.1488 | 0.1315 |
| pooled | 11 | 0.7669 | 0.4779 | 0.2890 | 0.0649 | 0.2208 | 0.0585 | 0.0846 | 0.0816 |
| pooled | 33 | 0.7739 | 0.4836 | 0.2903 | 0.0649 | 0.2468 | 0.0619 | 0.0932 | 0.0896 |

Interim aggregate:

| head | seeds | rank mean +/- sd | shifted mean | delta mean | image corr mean | ROI corr mean | identity diag-offdiag mean |
|---|---|---:|---:|---:|---:|---:|---:|
| query | 11,33 | 0.7314 +/- 0.0099 | 0.4835 | 0.2479 | 0.0515 | 0.1497 | 0.1296 |
| pooled | 11,33 | 0.7704 +/- 0.0050 | 0.4808 | 0.2896 | 0.0602 | 0.0889 | 0.0856 |

## Interpretation

The direct real visual64 signal is not a one-seed accident for either head.
Both ordered-query and pooled models beat shifted-null with very similar
rank gaps across seed11 and seed33. This strengthens the real-fMRI grounding
claim for visual cortex.

The query-vs-pooled split is now clearer:

- pooled is the stronger scalar ROI-retrieval head:
  0.7704 +/- 0.0050 vs query 0.7314 +/- 0.0099;
- query is the stronger ROI-wise / identity-binding head:
  ROI corr 0.1497 vs pooled 0.0889, identity diag-offdiag 0.1296 vs 0.0856;
- therefore the current claim should be framed as performance head vs
  interpretable spatial head, not as "ordered query beats pooled on every
  metric."

The queued `atm_real_fmri_visual64_multiseed` run is continuing with
query/pooled seed77.

## Artifacts

- Multiseed summarizer:
  `fmri_foundation_workspace/scripts/summarize_atm_real_fmri_visual64_multiseed.py`
- Interim summary JSON:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed.json`
- Interim aggregate CSV:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed_aggregate.csv`
- Running tmux session:
  `atm_real_fmri_visual64_multiseed`
