# Real-fMRI Visual64 Dual-Head Pilot

Date: 2026-06-05

## Question

Top-conference reviewers will usually want a primary metric, not only an
interpretability story. The current blocker is that:

- pooled readout wins scalar real-fMRI visual64 ROI retrieval;
- ordered query readout wins ROI identity and interpretability.

This pilot tests whether a dual-head model can keep both:

```text
ATM EEG tokens
  -> pooled ROI head       -> scalar ROI-pattern retrieval
  -> ordered-query ROI head -> ROI identity / interpretability
```

The heads are not mixed into one output. They are evaluated separately.

## Setup

- Target: measured THINGS-fMRI visual64 ROI beta targets, not TRIBE.
- Train: 6330 exact THINGS-EEG/THINGS-fMRI overlap images.
- Test: 77 exact overlap images.
- Seed: 33.
- Checkpoint: `model_best_roi_rank.pt`.
- Main dual output `roi_pred`: pooled head.
- Auxiliary output `roi_pred_query`: ordered-query head.

## Results

| model/output | rank | shifted | delta | top1 | top5 | image corr | ROI corr | identity diag-offdiag | within-between |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query / roi_pred | 0.7384 | 0.4838 | 0.2546 | 0.0779 | 0.1818 | 0.0525 | 0.1488 | 0.1315 | 0.0163 |
| pooled / roi_pred | 0.7739 | 0.4836 | 0.2903 | 0.0649 | 0.2468 | 0.0619 | 0.0932 | 0.0896 | 0.0080 |
| dual / pooled output | 0.7652 | 0.4928 | 0.2724 | 0.0649 | 0.2468 | 0.0583 | 0.0814 | 0.0792 | 0.0064 |
| dual / query output | 0.7054 | 0.4880 | 0.2174 | 0.0390 | 0.1688 | 0.0423 | 0.1346 | 0.1198 | 0.0104 |

## Interpretation

The dual-head direction is better aligned with the story than direct mixing:

- the pooled output preserves most scalar rank/top5: 0.7652 / 0.2468, close to
  pooled baseline 0.7739 / 0.2468;
- the query output preserves a stronger identity signal than pooled:
  0.1198 vs 0.0896;
- but it still does not beat the best single-purpose heads:
  pooled remains best for scalar rank and original query remains best for
  identity.

Therefore this is a partial positive architecture result, not a main-conference
metric win.

## Decision

For a top-conference paper, this cannot be sold as "we improve retrieval/SOTA"
yet. The stronger framing is:

> the cortical branch is a metric-preserving spatial grounding/readout module:
> pooled/semantic heads preserve retrieval performance, while ordered queries
> provide real-fMRI-aligned, ROI-specific interpretability.

To become a main-metric paper, the next experiments need either:

- a standard THINGS-EEG retrieval/generation gain over the ATM baseline;
- a finer real/pseudo cortical target where query/locality has a measurable
  advantage over pooled readout;
- a stronger dual-objective design that improves the query output without
  weakening pooled performance.

## Artifacts

- Code:
  `fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py`
  `fmri_foundation_workspace/scripts/evaluate_atm_real_fmri_roi_runs.py`
- Run:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_real_fmri_shared_roi207/atm_dual_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005/`
- Evaluation:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64.csv`
