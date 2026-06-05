# Proto256 pooled learned CLIP fusion result

Date: 2026-06-05

## Setup

This experiment tests whether the real-fMRI-aligned ROI/prototype signal can be
converted into THINGS-EEG image retrieval gains. It uses exported train/test
predictions from the proto256 pooled residual ATM models and trains only a small
post-hoc adapter on train images.

Protocol:

- Train data: 16,540 THINGS-EEG training images.
- Test data: 200 THINGS-EEG test images.
- Checkpoints: `bestroi`, `bestcliprank`, and `final`.
- Seeds: 11, 33, 77.
- Target: CLIP ViT-H/14 image embedding.
- Hyperparameters are selected on a train-only validation split.
- The 200-image test set is used once for final reporting.

Models tested:

- `semantic_identity`: original ATM semantic output.
- `semantic_ridge`: train-only ridge from semantic output to CLIP.
- `roi_ridge`: train-only ridge from ROI/prototype output to CLIP.
- `semantic_roi_ridge`: train-only ridge from concatenated semantic+ROI output to CLIP.
- `roi_residual_add`: train-only ROI-to-residual correction added to the original semantic output.
- `roi_clip_mix`: train-only ROI-to-CLIP adapter mixed with the original semantic output.
- `semantic_roi_clip_mix`: train-only semantic+ROI-to-CLIP adapter mixed with the original semantic output.

Main script:

- `fmri_foundation_workspace/scripts/evaluate_atm_proto256_clip_fusion.py`

Main output:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_learned_fusion/proto256_pooled_clip_fusion/summary.md`

## Main result

The only useful fusion variant is `semantic_roi_clip_mix`: learn a small
semantic+ROI adapter to CLIP on train images, normalize it, and mix it back into
the original semantic embedding with a small validation-selected weight.

| checkpoint | model | top1 mean | top5 mean | rank mean |
|---|---|---:|---:|---:|
| bestroi | semantic_identity | 0.4067 | 0.7783 | 0.9786 |
| bestroi | semantic_roi_clip_mix | 0.4350 | 0.8183 | 0.9803 |
| bestcliprank | semantic_identity | 0.4133 | 0.7750 | 0.9791 |
| bestcliprank | semantic_roi_clip_mix | 0.4267 | 0.8183 | 0.9807 |
| final | semantic_identity | 0.4167 | 0.7717 | 0.9786 |
| final | semantic_roi_clip_mix | 0.4333 | 0.8167 | 0.9797 |

Per-seed `semantic_roi_clip_mix` uses a nonzero mixing weight selected from the
train-only validation split:

| checkpoint | seed | top1 | top5 | rank | alpha | mix weight |
|---|---:|---:|---:|---:|---:|---:|
| bestroi | 11 | 0.4300 | 0.8250 | 0.9797 | 1 | 0.1 |
| bestroi | 33 | 0.4600 | 0.8350 | 0.9819 | 1 | 0.1 |
| bestroi | 77 | 0.4150 | 0.7950 | 0.9794 | 10 | 0.1 |
| bestcliprank | 11 | 0.4300 | 0.8250 | 0.9797 | 1 | 0.1 |
| bestcliprank | 33 | 0.4200 | 0.8300 | 0.9828 | 0.1 | 0.1 |
| bestcliprank | 77 | 0.4300 | 0.8000 | 0.9795 | 10 | 0.1 |
| final | 11 | 0.4300 | 0.8250 | 0.9797 | 1 | 0.1 |
| final | 33 | 0.4400 | 0.8250 | 0.9798 | 1 | 0.2 |
| final | 77 | 0.4300 | 0.8000 | 0.9795 | 10 | 0.1 |

## Negative controls

The result is not simply "any linear adapter helps":

- `semantic_ridge` is much worse than the original semantic identity output.
- `roi_ridge` alone is far below semantic retrieval.
- `semantic_roi_ridge` without late mixing is also much worse.
- `roi_clip_mix` is roughly neutral and does not explain the top5 gain.

This suggests the useful behavior is specifically a small late correction from
the semantic+ROI feature, not replacing the semantic output with a ridge head.

## Statistical caution

The top1/top5 gains are consistent across checkpoints, but rank gains are small:

- bestroi rank: 0.9803 vs 0.9786
- bestcliprank rank: 0.9807 vs 0.9791
- final rank: 0.9797 vs 0.9786

Paired row-level bootstrap CIs for `semantic_roi_clip_mix - semantic_identity`
are positive for the bestroi/bestcliprank seed33 runs, but cross zero for several
other seed/checkpoint combinations. Therefore this is not yet a final
main-table performance claim.

## Interpretation

This is the first result in this route where the cortical/prototype branch gives
a repeatable downstream retrieval improvement over the original semantic EEG
embedding, provided the fusion is late, normalized, and validation-selected.

The cleanest current story is:

1. The ROI/prototype branch contains real-fMRI-aligned information above shifted
   null.
2. Direct ROI replacement or naive concatenation hurts semantic retrieval.
3. A small learned late fusion can convert part of the cortical signal into
   top1/top5 retrieval gains.

## Decision

Continue with learned fusion. The next high-value experiment is an in-model or
adapter-based semantic+ROI late fusion head trained with the same train-only
selection rule, ideally on the full 60k image budget or with an independent
validation set. Do not spend time on more fixed-weight rerank sweeps.
