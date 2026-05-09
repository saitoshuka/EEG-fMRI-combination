# Subject Adaptation Decision - 2026-05-09

This report consolidates the subject-adaptation canary runs:

- `results/subject_adaptation_canary_v1_tf_bestlag/adaptation_summary.csv`
- `results/subject_adaptation_canary_v2_tf_threshold/adaptation_summary.csv`

Setup: heldout subjects receive an early calibration segment from each run, while evaluation uses later heldout blocks with a 20-step gap and the same residual/null controls used in the strict canary funnel.

Decision threshold used here: continue only when `time_residual_eeg_ridge` has rank percentile above `0.53` and remains clearly above the shifted-null residual control.

## Verdict

There is still a real but narrow signal, and it becomes visible mainly after subject-specific calibration. The direction should continue only as a personalized/subject-adapted EEG-to-fMRI alignment problem. It is not strong enough yet to justify blind zero-shot cross-subject LaBraM-to-NeuroSTORM spatial distillation across all datasets.

Practical consequence: do not scale the current pooled zero-shot setup further. The next model should include subject adaptation, for example per-subject adapters or FiLM/low-rank subject heads, and should treat XP2 as a negative/stress-control rather than as a primary training source.

## Best Rows

| dataset | best calib frac | status | resid rank | shifted | resid-shifted | diag-off | time rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| affective | 0.40 | pass | 0.5374 | 0.4926 | 0.0449 | 0.0494 | 0.4998 |
| experience | 0.40 | pass | 0.5349 | 0.5092 | 0.0256 | 0.0457 | 0.5880 |
| xp2 | 0.40 | fail | 0.5193 | 0.5076 | 0.0118 | 0.0229 | 0.5268 |

## Interpretation

- Affective is the cleanest positive result. It crosses the stricter `0.53` line at 30 percent calibration and improves further at 35-40 percent.
- Experience is weaker but not dead. It becomes convincing only at 40 percent calibration; the strong time-only rank means its non-residual score is heavily confounded, so the residual metric is the one to trust.
- XP2 does not pass even with 40 percent calibration. This is evidence that some datasets/runs are not currently usable for the main distillation objective under this preprocessing.
- Overall, EEG is probably not pure noise, but subject/run idiosyncrasy is dominating. The paired-data bottleneck is now adaptation and preprocessing stability, not simply model size.

## Key Threshold Rows

| dataset | calib frac | resid rank | shifted | resid-shifted | diag-off | pass meaning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| affective | 0.30 | 0.5316 | 0.4946 | 0.0370 | 0.0428 | Affective first clear continuation point |
| affective | 0.40 | 0.5374 | 0.4926 | 0.0449 | 0.0494 | Affective stronger calibrated point |
| experience | 0.40 | 0.5349 | 0.5092 | 0.0256 | 0.0457 | Experience large-calibration continuation point |
| xp2 | 0.40 | 0.5193 | 0.5076 | 0.0118 | 0.0229 | XP2 best large-calibration failure |

## Recommended Next Experiment

Train a small subject-adaptive model on Affective first, optionally adding Experience second: frozen/mostly frozen EEG encoder features plus subject adapter or FiLM conditioning, residual objective, shifted-null control, and XP2 as a heldout stress test. If this cannot beat the calibrated ridge canary, deeper LaBraM/NeuroSTORM coupling is premature.

Full consolidated CSV: `results/subject_adaptation_decision_20260509.csv`
