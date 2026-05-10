# ROI-Query Temporal Hierarchy vs Semantic-Only Probe, 8192 Images

Date: 2026-05-10

## Question

Does the early-to-late ROI timing pattern come specifically from the ordered
ROI-query spatial branch, or is it already present in the semantic-only
EEG-CLIP representation?

## Setup

- Budget: 8192 THING-EEG training images.
- Test: 200 THING-EEG test images, averaged across 10 subjects.
- Time windows: 100 ms windows over 0-1000 ms post-stimulus EEG.
- ROI targets: TRIBE-derived visual pseudo-ROI targets.
- ROI-query model:
  - ATM backbone plus ordered ROI-query branch.
  - Directly predicts group12 or parcel38 ROI targets.
- Semantic-only probe baseline:
  - Load the trained semantic-only ATM model.
  - Average its train EEG-CLIP embeddings per image.
  - Fit a ridge probe from semantic embedding to the same TRIBE ROI targets.
  - Run the same test-time keep/drop EEG window ablation.

## Key Comparison

Best keep-window and most important drop-window:

| model | target | ROI group | full signal corr | best keep | most important drop |
|---|---|---|---:|---|---|
| ROI-query | group12 | early_calcarine | 0.733 | 100-200 ms | 100-200 ms |
| ROI-query | group12 | high_level_combined | 0.112 | 600-700 ms | 300-400 ms |
| semantic probe | group12 | early_calcarine | 0.701 | 100-200 ms | 100-200 ms |
| semantic probe | group12 | high_level_combined | 0.678 | 200-300 ms | 100-200 ms |
| ROI-query | parcel38 | early_calcarine | 0.628 | 100-200 ms | 100-200 ms |
| ROI-query | parcel38 | high_level_combined | 0.172 | 600-700 ms | 200-300 ms |
| semantic probe | parcel38 | early_calcarine | 0.738 | 100-200 ms | 100-200 ms |
| semantic probe | parcel38 | high_level_combined | 0.698 | 200-300 ms | 200-300 ms |

## Interpretation

The semantic-only probe is very strong at predicting TRIBE pseudo-ROI targets
from global EEG-CLIP embeddings. This means TRIBE visual ROI targets contain a
large amount of stimulus/semantic information, and a semantic representation can
recover much of them.

However, the semantic-only probe does not reproduce the same early-to-late ROI
timing separation:

- For semantic-only probe, high-level ROI predictions mostly peak at 200-300 ms
  and are often harmed most by dropping 100-200 ms or 200-300 ms.
- For the ROI-query branch, early_calcarine consistently depends on 100-200 ms,
  while high_level_combined shifts later, with best keep at 600-700 ms in both
  group12 and parcel38.

So the claim should not be:

> The ROI-query branch is simply better at predicting ROI targets.

The stronger and more defensible claim is:

> Semantic-only embeddings already carry strong stimulus information that can
> linearly predict TRIBE pseudo-ROI targets, but ordered ROI-query supervision
> induces a more ROI-specific temporal organization: early visual ROI outputs
> depend on early EEG windows, while high-level visual/semantic ROI outputs
> depend more on later EEG windows.

## Caveats

- The high-level ROI-query full signal is weaker than the semantic probe full
  signal, especially for group12. This is not a pure performance win.
- The ROI targets are TRIBE pseudo-labels, not measured fMRI.
- This comparison uses a post-hoc semantic probe. It is a useful baseline for
  representational content, but it is not the same architecture as the
  end-to-end ROI-query branch.

## Artifacts

- ROI-query temporal hierarchy:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_temporal_hierarchy/n8192_reuse4096_clean256_shallow_fine100/`
- Semantic-only probe temporal hierarchy:
  - `fmri_foundation_workspace/results/eeg_image_bridge/atm_semantic_probe_temporal_hierarchy/n8192_reuse4096_clean256_shallow_fine100/`
- New semantic-probe evaluator:
  - `fmri_foundation_workspace/scripts/evaluate_semantic_probe_temporal_hierarchy.py`

## Next Controls

The next controls should test whether the ROI-query temporal separation is
specific to ordered spatial supervision:

- Train ROI-query with shuffled train ROI targets.
- Train no-query global ROI head with the same ROI losses.
- Permute ROI query-to-target order after training and rerun ablation.
- Repeat with another random seed.
