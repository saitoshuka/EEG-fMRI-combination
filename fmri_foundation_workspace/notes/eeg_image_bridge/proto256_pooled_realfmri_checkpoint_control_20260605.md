# Proto256 pooled real-fMRI checkpoint-control result

Date: 2026-06-05

## Setup

This follow-up addresses the checkpoint-selection warning from the integrity
audit. The earlier positive result used `model_best_roi_rank.pt`, which could be
criticized as selecting a checkpoint by pseudo-ROI behavior. I therefore ran the
same measured THINGS-fMRI external validation on two additional checkpoint
choices:

- `bestroi`: `model_best_roi_rank.pt`
- `bestcliprank`: `model_best_clip_rank.pt`
- `final`: `model_final.pt`

All three use the same train-only ridge probe from exported EEG features to
measured THINGS-fMRI targets, with 6330 train overlap images and 77 exact-overlap
test images.

New scripts:

- `fmri_foundation_workspace/scripts/run_atm_proto256_pooled_final_realfmri_probe.sh`
- `fmri_foundation_workspace/scripts/bootstrap_atm_proto256_realfmri_probe_ci.py`

Main output:

- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/proto256_pooled_realfmri_checkpoint_ci.md`

## Aggregate rank result

| checkpoint | target | semantic rank | ROI rank | ROI shifted | ROI - shifted | ROI - semantic |
|---|---|---:|---:|---:|---:|---:|
| bestroi | visual64 | 0.6005 | 0.6202 | 0.4827 | 0.1375 | 0.0198 |
| bestcliprank | visual64 | 0.5971 | 0.6164 | 0.4769 | 0.1394 | 0.0193 |
| final | visual64 | 0.5984 | 0.6101 | 0.4843 | 0.1258 | 0.0117 |
| bestroi | shared207 | 0.5587 | 0.5995 | 0.5110 | 0.0885 | 0.0408 |
| bestcliprank | shared207 | 0.5567 | 0.5738 | 0.4979 | 0.0759 | 0.0171 |
| final | shared207 | 0.5575 | 0.5767 | 0.5031 | 0.0736 | 0.0193 |

## Paired uncertainty result

The row-level bootstrap/sign-flip check samples the 77 exact-overlap test
images, not independent subjects. It is a small-test-set uncertainty check, not
a population-level inference.

Key pattern:

- `ROI - shifted` is positive for every seed, target, and checkpoint.
- The 95% bootstrap CI for `ROI - shifted` is positive for every final and
  bestcliprank seed.
- `ROI - semantic` is weaker. It is positive on the aggregate for all targets
  and checkpoints, but most per-seed CIs cross zero and seed 77 is negative for
  ROI-vs-semantic.

Representative final-checkpoint rows:

| target | seed | ROI - shifted mean | 95% CI | ROI - semantic mean | 95% CI |
|---|---:|---:|---|---:|---|
| visual64 | 11 | 0.1488 | [0.0748, 0.2271] | 0.0489 | [-0.0205, 0.1203] |
| visual64 | 33 | 0.1476 | [0.0586, 0.2430] | 0.0167 | [-0.0625, 0.0971] |
| visual64 | 77 | 0.0810 | [0.0311, 0.1323] | -0.0306 | [-0.0593, -0.0017] |
| shared207 | 11 | 0.0815 | [0.0111, 0.1555] | 0.0468 | [-0.0217, 0.1112] |
| shared207 | 33 | 0.0959 | [0.0176, 0.1748] | 0.0321 | [-0.0443, 0.1104] |
| shared207 | 77 | 0.0434 | [0.0079, 0.0810] | -0.0212 | [-0.0571, 0.0104] |

## Interpretation

The positive real-fMRI transfer signal is not only a pseudo-ROI-best-checkpoint
artifact. It remains above shifted null when using the final checkpoint and a
CLIP-rank-selected checkpoint.

However, the stronger claim that the ROI feature robustly beats semantic-only
is not yet stable enough for a main table. Best-ROI selection makes the effect
look stronger, especially on shared207, but clean checkpoint controls reduce the
ROI-vs-semantic margin and expose seed77 as a failure case.

## Claim update

Supported:

> The cortical-prototype branch learns EEG features with measurable
> real-fMRI-aligned signal above row-shifted null across checkpoint choices.

Tentative:

> ROI features can improve fMRI pattern-rank transfer over semantic-only
> features on average, especially on shared207.

Not yet supported:

> The current ROI branch is a robust performance improvement over semantic-only
> across seeds and checkpoint-selection rules.

## Decision

Continue the direction, but do not frame the next AAAI draft as "we beat SOTA by
adding ROI loss." The stronger story is:

1. semantic EEG visual decoding is already strong;
2. cortical teacher supervision injects a real-fMRI-aligned signal that survives
   clean checkpoint controls;
3. the remaining bottleneck is fusion: the ROI signal is measurable, but the
   current model does not reliably convert it into semantic retrieval gains.

The next decision-value experiment should therefore be a learned fusion or
reranking module evaluated against the semantic-only baseline, not another
target-space sweep.
