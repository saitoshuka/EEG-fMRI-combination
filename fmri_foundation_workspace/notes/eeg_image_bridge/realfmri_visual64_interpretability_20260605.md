# Real-fMRI Visual64 Interpretability Check

## Question

Does the ordered ROI-query branch show interpretable time/channel structure
when the target is real THINGS-fMRI visual ROI activity, rather than TRIBE
pseudo-cortical output?

## Setup

- Target: real THINGS-fMRI visual64 ROI activity.
- Split: 6,330 train images, 77 exact-overlap THINGS-EEG/THINGS-fMRI test
  images.
- Model: ATM EEG backbone, `d_model=256`, no subject token, shallow semantic
  head.
- Checkpoint: `model_best_roi_rank.pt`.
- Runs compared:
  - ordered ROI-query head
  - pooled no-query head

The time-window metric is:

```text
corr(predicted ROI_i, real ROI_i) - corr(predicted ROI_i, shifted real ROI_i)
```

computed over the 77 exact-overlap test images. The channel-attention map is
post-hoc query cross-attention over ATM EEG channel tokens, averaged over
subjects and test images.

## Results

### Retrieval and identity

| head | visual64 rank | shifted | image corr | ROI corr | query-target diag-offdiag |
|---|---:|---:|---:|---:|---:|
| ordered query | 0.7384 | 0.4838 | 0.0525 | 0.1488 | 0.1332 |
| pooled no-query | 0.7739 | 0.4836 | 0.0619 | 0.0932 | 0.0908 |

The pooled head remains better for scalar image-level ROI retrieval, but the
ordered query head has stronger ROI identity binding.

### Time-window dependency

| head | ROI family | full corr-signal | best keep | most important drop |
|---|---|---:|---|---|
| ordered query | early visual | 0.0612 | 100-200 ms | 100-200 ms |
| ordered query | mid visual | 0.1677 | 300-400 ms | 300-400 ms |
| ordered query | ventral/category high | 0.1345 | 300-400 ms | 300-400 ms |
| pooled no-query | early visual | 0.0861 | 700-800 ms | 100-200 ms |
| pooled no-query | mid visual | 0.0923 | 300-400 ms | 300-400 ms |
| pooled no-query | ventral/category high | 0.1164 | 300-400 ms | 100-200 ms |

The ordered query branch gives a cleaner staged pattern: early visual ROIs are
most dependent on 100-200 ms EEG, while mid and ventral/category visual ROIs are
most dependent on 300-400 ms EEG.

### Channel attention

Top channels by mean query attention are posterior/parieto-occipital:

| channel | mean attention |
|---|---:|
| P8 | 0.0868 |
| Oz | 0.0739 |
| P6 | 0.0691 |
| O1 | 0.0591 |
| O2 | 0.0516 |
| PO7 | 0.0507 |
| PO4 | 0.0445 |
| PO3 | 0.0408 |
| PO8 | 0.0375 |
| POz | 0.0354 |

Channel-family attention share:

| channel family | share |
|---|---:|
| parietal P | 0.2886 |
| parieto-occipital PO | 0.2089 |
| occipital O | 0.1845 |
| temporo-parietal TP | 0.0562 |

## Interpretation

This supports a cautious neuroscience-facing claim:

> On real THINGS-fMRI visual targets, the ordered query branch does not beat
> pooled prediction in image-level ROI retrieval, but it gives stronger
> ROI-specific identity and more interpretable temporal/channel structure:
> early visual targets rely on earlier EEG windows, later visual-category
> targets rely on later windows, and attention is concentrated over posterior
> visual electrodes.

Do not claim precise cortical source localization from EEG attention. The safer
claim is model attribution and structured cortical-target binding.

## Artifacts

- Time-window outputs:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/realfmri_visual64_bestroi_query_vs_pooled`
- Query-channel attention:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/realfmri_visual64_bestroi_query_attention`
- Query-target identity heatmaps:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/realfmri_visual64_bestroi_query_vs_pooled_identity`
- Summary:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/realfmri_visual64_interpretability/summary.md`
