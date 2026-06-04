# Real-fMRI Visual64 Multiseed Final

Date: 2026-06-05

## Status

The direct real THINGS-fMRI visual64 experiment has now completed for
seed11/33/77 with both ordered-query and pooled heads.

This is a robustness gate for the measured-fMRI result:

```text
THINGS-EEG -> ATM backbone -> visual64 ROI head -> real THINGS-fMRI visual64
```

The target is real measured THINGS-fMRI visual ROI beta activity, not a TRIBE
pseudo-target.

## Per-Seed Results

Best ROI-checkpoint results:

| head | seed | visual64 rank | shifted | delta | top1 | top5 | image corr | ROI corr | query identity diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query | 11 | 0.7244 | 0.4833 | 0.2411 | 0.0519 | 0.2338 | 0.0505 | 0.1506 | 0.1278 |
| query | 33 | 0.7384 | 0.4838 | 0.2546 | 0.0779 | 0.1818 | 0.0525 | 0.1488 | 0.1315 |
| query | 77 | 0.6931 | 0.4757 | 0.2174 | 0.0260 | 0.1299 | 0.0476 | 0.1346 | 0.1190 |
| pooled | 11 | 0.7669 | 0.4780 | 0.2890 | 0.0649 | 0.2208 | 0.0585 | 0.0846 | 0.0816 |
| pooled | 33 | 0.7739 | 0.4836 | 0.2903 | 0.0649 | 0.2468 | 0.0619 | 0.0932 | 0.0896 |
| pooled | 77 | 0.7592 | 0.4788 | 0.2804 | 0.0779 | 0.1818 | 0.0551 | 0.0812 | 0.0774 |

## Aggregate Results

| head | seeds | rank mean +/- sd | shifted mean | delta mean | image corr mean | ROI corr mean | identity diag-offdiag mean |
|---|---|---:|---:|---:|---:|---:|---:|
| query | 11,33,77 | 0.7186 +/- 0.0232 | 0.4809 | 0.2377 | 0.0502 | 0.1447 | 0.1261 |
| pooled | 11,33,77 | 0.7667 +/- 0.0074 | 0.4801 | 0.2866 | 0.0585 | 0.0863 | 0.0829 |

## Interpretation

The direct real visual64 signal is robust across seeds for both heads. Both
ordered-query and pooled models beat shifted-null by a large margin on the
exact 77-image THINGS-EEG/THINGS-fMRI overlap.

The query-vs-pooled split is now clean:

- pooled is the stronger scalar ROI-retrieval / performance head:
  0.7667 +/- 0.0074 vs query 0.7186 +/- 0.0232;
- query is the stronger ROI-wise and identity-binding head:
  ROI corr 0.1447 vs pooled 0.0863, identity diag-offdiag 0.1261 vs 0.0829;
- therefore the claim should be framed as:

> EEG contains stimulus-driven visual-fMRI structure. A pooled ROI head is best
> for scalar ROI-pattern retrieval, while ordered ROI queries provide stronger
> spatial identity binding and interpretable cortical readouts.

This is stronger than the previous single-seed result and should replace the
earlier interim note in presentation / paper planning.

## Claim Boundary

Supported:

- direct real visual-fMRI ROI patterns are predictable from EEG above shifted
  null;
- the result is stable across three random seeds;
- the ordered-query branch provides stronger fixed-ROI identity and
  interpretability than pooled prediction.

Not supported:

- ordered query is not the scalar-rank winner;
- this is not a whole-brain fMRI decoding claim;
- this does not prove TRIBE pseudo-targets are real fMRI, because this result
  uses measured THINGS-fMRI directly.

## Artifacts

- Multiseed summarizer:
  `fmri_foundation_workspace/scripts/summarize_atm_real_fmri_visual64_multiseed.py`
- Final summary JSON:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed.json`
- Final aggregate CSV:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64_multiseed_aggregate.csv`
