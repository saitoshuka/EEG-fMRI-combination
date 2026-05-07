# CATD Reproduction Audit

## What Was Tested

This folder contains an audit-style reproduction of the CATD paper's public-data claims. It does not rerun the full A800 diffusion model. Instead, it reproduces the data split and alignment assumptions that matter most for checking whether the reported results could be inflated by task schedule, slow BOLD structure, or split leakage.

Implemented checks:

- XP1 motor imagery dataset with the paper's subject split:
  - train: `xp101`, `xp102`, `xp103`, `xp107`, `xp108`, `xp109`, `xp110`
  - test: `xp104`, `xp105`, `xp106`
- NODDI resting EEG-fMRI with the paper's test subjects:
  - test: `48`, `49`
  - train: remaining available subjects
- 6 s EEG-to-BOLD delay.
- EEG condition features from STFT-style bandpower over delta/theta/alpha/beta/low-gamma.
- fMRI target as a coarse `8 x 8 x 4` whole-volume grid, z-scored within run, then compressed to train-fold PCA16.
- Strict controls:
  - `train_mean`
  - `schedule_or_time_ridge`: no EEG input
  - `eeg_shifted_null`: EEG input but circularly shifted fMRI training target

## Main Results

| Dataset | Model | Grid r mean | Spatial r mean | RMSE | R2 weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| XP1 all 53 runs | schedule/time only | 0.4460 | 0.2726 | 0.8478 | 0.2813 |
| XP1 all 53 runs | EEG ridge | 0.0285 | 0.0270 | 3.1799 | -9.1118 |
| XP1 all 53 runs | EEG shifted null | -0.0202 | -0.0275 | 1.4203 | -1.0172 |
| NODDI rest | time only | 0.3210 | 0.4871 | 0.8756 | 0.2334 |
| NODDI rest | EEG ridge | 0.1234 | 0.0690 | 1.1101 | -0.2323 |
| NODDI rest | EEG shifted null | -0.1258 | -0.0716 | 1.1244 | -0.2642 |

Full table:

```text
catd_reproduction/audit_summary.csv
```

Detailed outputs:

```text
catd_reproduction/xp1_all/results/
catd_reproduction/xp1/results/
catd_reproduction/noddi/results/
```

## Residual Prediction

I also ran residual fMRI prediction. The residual target is:

```text
fMRI_residual = fMRI_true - fMRI_predicted_from_schedule_or_time
```

This asks whether EEG explains anything after removing the part already explained by task/block phase or run time.

| Dataset | Residual model | Residual grid r mean | Residual spatial r mean | R2 weighted |
| --- | --- | ---: | ---: | ---: |
| XP1 all 53 runs | residual EEG ridge | -0.0039 | -0.0041 | -3.2100 |
| XP1 all 53 runs | residual EEG shifted null | -0.0031 | -0.0028 | -1.2734 |
| NODDI rest | residual EEG ridge | -0.0187 | 0.0385 | -0.1880 |
| NODDI rest | residual EEG shifted null | -0.0006 | 0.0116 | -0.1106 |

Residual outputs:

```text
catd_reproduction/xp1_all/results_residual/
catd_reproduction/noddi/results_residual/
```

The residual result is the strongest negative evidence so far: after removing schedule/time, EEG prediction falls to null or worse. Under this audit setup, the apparent EEG-fMRI signal is largely explained by nuisance timing structure rather than EEG condition strength.

## Classification Red Flag

On XP1 all tasks, a classifier trained on real training fMRI gets:

| Test input | Accuracy | F1 |
| --- | ---: | ---: |
| real fMRI | 0.6153 | 0.6291 |
| schedule-generated fMRI | 0.6141 | 0.6168 |
| EEG-generated fMRI | 0.5077 | 0.4509 |
| shifted-null EEG-generated fMRI | 0.4889 | 0.2872 |
| direct schedule classifier | 1.0000 | 1.0000 |

This means the block schedule alone can produce synthetic fMRI that performs about as well as real fMRI on rest-vs-task classification. Therefore, downstream classification is not strong evidence that EEG faithfully generated fMRI.

## Interpretation

This audit does not prove fabrication. It does show that the paper's headline-style evidence is not convincing without much stronger controls.

The main concern is XP1's block design. Rest and task alternate on a fixed 20 s schedule, and the same schedule exists across train/test subjects. A model can learn task/timing templates and achieve strong BOLD reconstruction/classification metrics without using EEG-to-fMRI neural correspondence. In this audit, the no-EEG schedule/time baseline is far stronger than EEG-only prediction.

NODDI is less obviously label-confounded because it is resting-state, but a no-EEG time/run-position baseline still beats EEG-only prediction. EEG is above shifted-null, so there may be a real weak signal, but it is not strong enough here to justify very large claims without additional validation.

After residualizing out schedule/time, EEG no longer predicts held-out residual fMRI in either XP1 or NODDI. That does not prove the paper fabricated results, but it does strongly support the concern that CATD-style scores can be dominated by timing priors and fMRI self-structure rather than EEG conditioning.

## Practical Verdict

The CATD architecture is still useful as inspiration, but the reported empirical results should be treated cautiously. Before trusting the paper, I would want to see:

1. Official code and exact preprocessing.
2. Subject-heldout EEG-shifted and schedule-only null baselines.
3. Metrics where synthetic BOLD must match held-out real BOLD temporally and spatially, not only improve classification.
4. Repeated subject-level splits, not hand-picked test subjects.
5. External validation on another paired EEG-fMRI dataset.

For our project, this supports a conservative approach: keep ridge/null/time baselines as mandatory controls, and do not report downstream classification as proof of EEG-to-fMRI generation.
