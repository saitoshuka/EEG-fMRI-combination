# Real-fMRI Visual64 Multiseed Interpretability

Date: 2026-06-05

## Status

The ordered-query real THINGS-fMRI visual64 interpretability check has now been
replicated across seeds 11/33/77 using `model_best_roi_rank.pt`.

This analysis uses measured THINGS-fMRI visual ROI beta targets, not TRIBE
pseudo-targets:

```text
THINGS-EEG -> ATM ordered ROI queries -> real THINGS-fMRI visual64
```

## Time-Window Results

Scores are `corr(predicted ROI_i, real ROI_i) - corr(predicted ROI_i,
shifted real ROI_i)`. The table is seed-balanced before aggregation.

| ROI family | full signal mean | best keep | seed best keeps | most important drop | seed drops |
|---|---:|---|---|---|---|
| all_visual64 | 0.1430 +/- 0.0056 | 300-400 ms | 300-400, 300-400, 300-400 | 300-400 ms | 300-400, 300-400, 300-400 |
| early_visual | 0.0812 +/- 0.0204 | 100-200 ms | 900-1000, 100-200, 100-200 | 100-200 ms | 100-200, 100-200, 100-200 |
| mid_visual | 0.1674 +/- 0.0003 | 300-400 ms | 300-400, 300-400, 300-400 | 300-400 ms | 300-400, 300-400, 300-400 |
| ventral_category_high | 0.1368 +/- 0.0113 | 300-400 ms | 300-400, 300-400, 300-400 | 300-400 ms | 300-400, 300-400, 300-400 |

Interpretation:

- the mid/ventral/all-visual 300-400 ms dependency is stable across all three
  seeds;
- the early visual drop-window dependency is also stable at 100-200 ms, but one
  seed's best-keep window lands at 900-1000 ms, so the early latency claim
  should be phrased cautiously;
- this supports a neuro-plausible temporal profile, but it remains an
  ablation-derived dependency pattern, not direct neural activation timing.

## Channel-Attention Results

Post-hoc cross-attention over ATM EEG channel tokens remains posterior weighted
across seeds.

| channel family | attention share mean +/- sd |
|---|---:|
| parietal_P | 0.2643 +/- 0.0240 |
| other | 0.2608 +/- 0.0789 |
| occipital_O | 0.2528 +/- 0.0924 |
| parieto_occipital_PO | 0.1697 +/- 0.0445 |
| temporo_parietal_TP | 0.0524 +/- 0.0099 |

Posterior `O/PO/P` share is 0.6867. Top channels are:

| rank | channel | family | mean attention |
|---:|---|---|---:|
| 1 | Oz | occipital_O | 0.1244 |
| 2 | P8 | parietal_P | 0.0813 |
| 3 | O1 | occipital_O | 0.0644 |
| 4 | O2 | occipital_O | 0.0640 |
| 5 | P6 | parietal_P | 0.0606 |
| 6 | PO7 | parieto_occipital_PO | 0.0470 |

Interpretation:

- the query branch consistently uses posterior visual channels, matching visual
  stimulus decoding expectations;
- the attention map is not itself proof of ROI specificity. It should be paired
  with the query-target identity advantage and the time-window ablation.

## Claim Boundary

Supported:

- ordered queries provide stable real-fMRI visual64 identity binding and
  seed-stable posterior channel weighting;
- time-window ablation gives a plausible early-vs-mid/ventral split:
  early visual is most harmed by removing 100-200 ms, while mid/ventral visual
  is most harmed by removing 300-400 ms.

Not supported:

- this is not a direct measurement of cortical activation latency;
- this does not prove whole-brain spatial knowledge;
- Oz/posterior attention should not be over-read as query-specific anatomy on
  its own.

## Artifacts

- Time-window output:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/realfmri_visual64_bestroi_query_multiseed/`
- Channel-attention output:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/realfmri_visual64_bestroi_query_multiseed/`
- Summary output:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/realfmri_visual64_multiseed_interpretability/summary.md`
- Summary script:
  `fmri_foundation_workspace/scripts/summarize_realfmri_visual64_multiseed_interpretability.py`
