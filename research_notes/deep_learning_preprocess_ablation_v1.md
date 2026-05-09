# Deep Learning Preprocess Ablation - 2026-05-09

## Question

The pooled transformer was weak even after selecting the cleaner Schaefer100 datasets. The working hypothesis was:

- maybe the model is too small or data is insufficient;
- maybe concatenating datasets creates domain conflict;
- maybe EEG/fMRI preprocessing is not strict enough, so the model learns nuisance structure instead of EEG-to-fMRI correspondence.

This ablation tested the third explanation first.

## What Changed

`scripts/tribe_style_eeg_fmri.py` now supports:

- `--x-run-zscore`: z-score EEG features within each run before model training.
- `--x-detrend-degree`: optional EEG feature detrending.
- `--target-residualize-time`: subtract time-basis prediction from the fMRI target.
- `--time-residual-group {global,dataset,run}`: choose how strictly to remove time/block phase.
- `--dataset-metrics`: report pooled metrics and per-dataset metrics.
- `--contrastive-scope {batch,dataset}`: restrict contrastive negatives to the same dataset if desired.
- `--linear-skip`: optional direct linear skip path from mean+last EEG features.

## Main Finding

Preprocessing was a major bottleneck.

Adding EEG run-wise z-score raised the pooled Affective+Experience residual transformer from:

- rank percentile `0.5133`
- diag-minus-offdiag `0.0034`

to:

- rank percentile `0.5283`
- diag-minus-offdiag `0.0281`

while shifted-null stayed near chance:

- shifted rank `0.5015`
- shifted diag-minus-offdiag `0.0007`

So at least part of the previous failure was cross-run / cross-dataset EEG feature scale mismatch.

## Target Space

For pooled Affective+Experience with EEG run-zscore and global time residual:

| target | transformer rank | shifted rank | diag-off | comment |
| --- | ---: | ---: | ---: | --- |
| PCA64 | 0.5283 | 0.5015 | 0.0281 | solid improvement |
| PCA32 | 0.5282 | 0.5008 | 0.0355 | similar rank, higher diag |
| identity Schaefer100 | 0.5310 | 0.5006 | 0.0269 | best rank |

This suggests Schaefer100 itself is not necessarily the problem. Compressing the target can slightly hurt retrieval rank, probably because pooled PCA mixes dataset-specific variance.

## Dataset-Wise Diagnosis

With pooled Affective+Experience, identity Schaefer100, EEG run-zscore, and global time residual:

| subset | transformer rank | shifted rank | time ridge rank | note |
| --- | ---: | ---: | ---: | --- |
| overall | 0.5310 | 0.5006 | 0.4809 | looks good globally |
| Affective | 0.5233 | 0.5001 | 0.4581 | weak but non-null |
| Experience | 0.5526 | 0.5018 | 0.5459 | suspiciously time/phase-like |

The Experience subset still had high time-ridge performance after global residualization. That means global time residual was not strict enough for pooled data.

## Strict Run-Wise Time Residual

After fitting/removing time basis separately within each run:

| experiment | transformer rank | shifted rank | time ridge rank | interpretation |
| --- | ---: | ---: | ---: | --- |
| pooled overall | 0.5183 | 0.4983 | 0.5016 | much weaker |
| pooled Affective subset | 0.5225 | 0.5008 | 0.5038 | survives |
| pooled Experience subset | 0.5065 | 0.4912 | 0.4954 | mostly gone |
| Affective-only | 0.5284 | 0.4949 | 0.5059 | strongest credible result |

Interpretation:

- Experience's attractive score was mostly residual run/block phase.
- Affective retains a small but repeatable EEG-linked signal even after stricter deconfounding.
- Pooling Affective with Experience hurts because Experience contributes heterogeneous and partly confounded structure.

## Architecture Attempts

Linear skip path:

- rank dropped from `0.5310` to `0.5211`;
- diag-offdiag increased, but R2 became strongly negative.

This was not adopted as the best path. It seems to distort retrieval geometry rather than produce a cleaner EEG-to-fMRI mapping.

Dataset-scoped contrastive:

- almost identical to batch-wide contrastive;
- not the main bottleneck.

## Current Best Deep-Learning Claim

The best currently credible deep-learning result is:

- Affective-only
- EEG run-wise zscore
- identity Schaefer100 target
- run-wise time residual
- within-run block split
- transformer rank `0.5284`
- shifted-null rank `0.4949`
- time-ridge rank `0.5059`

This is still a small effect, but it is finally in the direction we want under a stricter control.

## What This Means

The poor deep-learning result is not simply "not enough data".

It is more specifically:

1. EEG feature normalization was insufficient.
2. Pooled target residualization was too global and left dataset/run phase structure.
3. Concatenating datasets can hurt when a dataset contributes confounded temporal structure.
4. The current bandpower token is useful for diagnosis, but probably too weak for the final spatial distillation model.

Next high-value move:

- build the same strict residual framework around better EEG inputs: official-style LaBraM preprocessing or raw waveform/time-frequency tokens;
- keep Affective as the clean positive anchor;
- only add a dataset to pooled training after it passes run-wise residual canary;
- use Experience cautiously, likely only after event/task-specific residualization.
