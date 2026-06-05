# ATM Feature To Real-fMRI Probe

Date: 2026-06-05

## Question

Does the cortical branch contain measured-fMRI information beyond the ATM
semantic embedding, or is the real-fMRI visual64 signal already explained by
semantic-only EEG features?

This check uses exported ATM predictions and fits a train-only ridge map to
measured THINGS-fMRI targets:

```text
exported ATM feature -> ridge on 6330 train-overlap images -> real fMRI ROI
```

The ridge alpha is selected on a heldout training split only, then refit on all
6330 train-overlap images and evaluated on the 77 exact THINGS-EEG/THINGS-fMRI
test images.

## Compared Features

- `semantic`: ATM semantic EEG embedding.
- `roi`: exported residual proto256 cortical prediction.
- `semantic_roi`: concatenated semantic + ROI features.
- `image_clip`: true image CLIP oracle control.

Two exported proto256 residual models were compared:

- `pooled`: pooled no-query residual proto256 head.
- `query`: ordered query residual proto256 head.

## Visual64 Results

Target: measured real THINGS-fMRI visual64.

| exported model | feature | rank | shifted | top1 | top5 | image corr | ROI corr |
|---|---|---:|---:|---:|---:|---:|---:|
| pooled | semantic | 0.6003 | 0.4790 | 0.0519 | 0.1818 | 0.2316 | 0.2362 |
| pooled | ROI proto256 | 0.6413 | 0.4797 | 0.0649 | 0.1948 | 0.1897 | 0.1845 |
| pooled | semantic + ROI | 0.5964 | 0.4797 | 0.0519 | 0.1688 | 0.2329 | 0.2294 |
| query | semantic | 0.6150 | 0.4814 | 0.0909 | 0.1948 | 0.2470 | 0.2179 |
| query | ROI proto256 | 0.5506 | 0.4836 | 0.0130 | 0.1039 | 0.0802 | 0.1737 |
| query | semantic + ROI | 0.6109 | 0.4848 | 0.0649 | 0.1558 | 0.2439 | 0.2172 |
| image CLIP oracle | image CLIP | 0.6765 | 0.4892 | 0.0649 | 0.2078 | 0.2576 | 0.2210 |

## Shared207 Results

Target: all 207 shared THINGS-fMRI metadata ROI columns.

| exported model | feature | rank | shifted | top1 | top5 | image corr | ROI corr |
|---|---|---:|---:|---:|---:|---:|---:|
| pooled | semantic | 0.5572 | 0.4971 | 0.0519 | 0.1429 | 0.1136 | 0.0820 |
| pooled | ROI proto256 | 0.6090 | 0.5099 | 0.0130 | 0.0909 | 0.0863 | 0.0808 |
| pooled | semantic + ROI | 0.5542 | 0.4979 | 0.0390 | 0.1169 | 0.1129 | 0.0694 |
| query | semantic | 0.5603 | 0.5041 | 0.0390 | 0.0909 | 0.1269 | 0.0461 |
| query | ROI proto256 | 0.5284 | 0.5068 | 0.0000 | 0.1299 | 0.0540 | 0.0851 |
| query | semantic + ROI | 0.5473 | 0.5113 | 0.0260 | 0.0909 | 0.1210 | 0.0463 |
| image CLIP oracle | image CLIP | 0.6224 | 0.4954 | 0.0909 | 0.2208 | 0.1207 | 0.1418 |

## Interpretation

This is a useful control result, not a final model result.

What it supports:

- The pooled residual proto256 ROI prediction transfers to measured fMRI better
  than the semantic EEG embedding on rank for both visual64 and shared207.
- The effect is stronger and more interpretable on visual64, which matches the
  intended visual-cortex story.
- The image CLIP oracle remains the strongest post-hoc feature, showing that
  the measured fMRI target is stimulus-driven and visual-model-readable.

What it does not support:

- Simple semantic+ROI concatenation does not improve the probe.  Fusion is not
  solved by post-hoc feature concatenation.
- The ordered query ROI output is not the strongest real-fMRI transfer feature.
  Query remains an identity/interpretability mechanism, not the current
  performance head.
- The post-hoc probe is weaker than directly training ATM on real visual64
  targets, where pooled reaches rank 0.7667 and ordered query reaches 0.7186
  across seeds 11/33/77.

## Claim Impact

This improves the story slightly:

> The cortical/prototype branch is not just a visualization artifact; the pooled
> proto256 residual prediction carries measurable visual-fMRI transfer signal.

But it also sharpens the limitation:

> The present ordered query constraint gives better spatial identity and
> interpretability, but it is not yet the best performance or real-fMRI transfer
> mechanism.

The next architecture should separate roles:

```text
pooled/local performance head -> scalar retrieval and real-fMRI transfer
ordered query head            -> ROI identity, time/channel maps, cortical viz
shared backbone + consistency -> make the two heads communicate
```

Artifacts:

- Script:
  `fmri_foundation_workspace/scripts/evaluate_atm_feature_to_realfmri_probe.py`
- Pooled visual64 metrics:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_pooled_residual_seed33_to_realfmri_visual64/test_metrics.csv`
- Pooled shared207 metrics:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_pooled_residual_seed33_to_realfmri_shared207/test_metrics.csv`
- Query visual64 metrics:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_query_residual_seed33_to_realfmri_visual64/test_metrics.csv`
- Query shared207 metrics:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/proto256_query_residual_seed33_to_realfmri_shared207/test_metrics.csv`
