# ATM ROI Temporal Hierarchy Probe, 4096 Images

Date: 2026-05-10

## Question

Can the TRIBE-supervised ATM ROI branch show a temporal hierarchy, where early
visual pseudo-ROIs depend on earlier EEG windows and high-level temporal/semantic
pseudo-ROIs depend on later EEG windows?

## Setup

- Models:
  - `atm_spatial_group_train_seed33_budget4096_faststill_n4096_d256_none`
  - `atm_spatial_parcel_train_seed33_budget4096_faststill_n4096_d256_none`
- Test data:
  - 200 THING-EEG test images, averaged over 10 subjects.
- EEG time axis:
  - Preprocessing used `tmin=-0.2, tmax=1.0`, baseline correction to 0, then saved
    `[:, :, :, 50:]`.
  - Current model input is therefore 250 samples = 0-1000 ms post-stimulus at
    250 Hz.
- Metrics:
  - `stim_corr_signal = corr(EEG_pred_ROI, TRIBE_ROI) - corr(EEG_pred_ROI, shifted_TRIBE_ROI)`
  - `keep-window`: only one EEG time window is retained.
  - `drop-window`: one EEG time window is zeroed out.

## Main Result

The 100 ms equal-window analysis supports a coarse temporal hierarchy.

| model | ROI group | full signal corr | best keep window | most important drop window |
|---|---:|---:|---:|---:|
| group12 | early_calcarine | 0.732 | 100-200 ms | 100-200 ms |
| group12 | early_visual_combined | 0.539 | 100-200 ms | 100-200 ms |
| group12 | high_level_combined | 0.100 | 600-700 ms | 300-400 ms |
| parcel38 | early_calcarine | 0.637 | 100-200 ms | 100-200 ms |
| parcel38 | early_visual_combined | 0.501 | 200-300 ms | 200-300 ms |
| parcel38 | high_level_combined | 0.214 | 600-700 ms | 400-500 ms |

The strongest and cleanest signal is:

- Early calcarine pseudo-ROI:
  - best keep window = `100-200 ms`
  - most important drop window = `100-200 ms`
  - replicated in both group12 and parcel38.
- High-level/semantic temporal pseudo-ROI:
  - best keep window = `600-700 ms`
  - most important drop window = `300-500 ms`
  - replicated directionally in both models.

## Interpretation

This is better evidence for brain-structured learning than the current CLIP
retrieval top1 result. It suggests the ROI branch is not only predicting a
generic image identity target: different visual ROI groups are most supported by
different EEG time windows.

Conservative claim:

> TRIBE-derived ROI supervision induces a temporally organized EEG-to-cortical
> pseudo-representation: early visual pseudo-ROIs are most sensitive to early
> post-stimulus EEG, while high-level temporal/semantic pseudo-ROIs depend more
> on later EEG activity.

Do not overclaim this as measured fMRI decoding. The ROI targets are still
TRIBE pseudo-labels derived from the stimulus image.

## Caveats

- `keep-window` is partly out-of-distribution because the model was trained on
  full EEG windows.
- `drop-window` is the more causal ablation, but windows can interact.
- The high-level full signal is weaker than early visual full signal, although
  the late-window preference is consistent.
- This should be repeated with 8192 targets and preferably multiple seeds.

## Artifacts

- Coarse windows:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256/`
- Equal 100 ms windows:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/`
- Metrics CSV:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/temporal_hierarchy_metrics.csv`
- Key heatmaps:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/atm_spatial_group_train_seed33_budget4096_faststill_n4096_d256_none/keep_window_stim_corr_signal.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/atm_spatial_group_train_seed33_budget4096_faststill_n4096_d256_none/drop_window_stim_corr_signal.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/atm_spatial_parcel_train_seed33_budget4096_faststill_n4096_d256_none/keep_window_stim_corr_signal.png`
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n4096_clean256_fine100/atm_spatial_parcel_train_seed33_budget4096_faststill_n4096_d256_none/drop_window_stim_corr_signal.png`

## Next Step

Use the same temporal hierarchy evaluator on the 8192 model after training. If
the early-to-late ROI timing pattern strengthens with data scale, it becomes a
much stronger story than raw CLIP top1.
