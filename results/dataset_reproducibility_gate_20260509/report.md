# Dataset-Level Reproducibility Gate - 2026-05-09

This is the stricter gate for deciding whether the EEG-to-fMRI distillation story can be told as a cross-dataset foundation-model direction.

## Decision

The current evidence does not clear the minimum multi-dataset gate for a big-model EEG-to-fMRI distillation claim.

- Download inventory: 21 top-level datasets.
- Strict gate evaluated: 7 datasets with shared Schaefer-style targets.
- Cross-subject residual pass: 0 datasets.
- Within-run residual pass: 3 datasets.
- Waveform-token evaluated: 4 datasets.
- Waveform nested strict within-run pass: 3 datasets.
- Waveform nested soft-or-strict within-run pass: 4 datasets.
- Early strict subject-adaptation pass, calibration <= 0.20: 0 datasets.
- Late strict subject-adaptation pass, calibration <= 0.40: 2 datasets.

Paper-level rule used here: at least 2-3 independent datasets should pass cross-subject residual controls or pass strict subject-adaptation with no more than 10-20 percent calibration. Current count is below that bar.

## Gate Matrix

| dataset | final gate | subj resid | subj gap | within resid | within gap | waveform best | waveform nested | first strict calib | best <=20% | target dim | ROI mask |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gradcpt | exclude_or_reprocess | 0.5059 | 0.0131 | 0.4892 | -0.0226 |  |  |  |  | 100 | 1.000 |
| natview | exclude_or_reprocess | 0.4942 | -0.0031 | 0.4952 | -0.0169 |  |  |  |  | 100 | 1.000 |
| speeded | exclude_or_reprocess | 0.5020 | 0.0018 | 0.5050 | 0.0072 |  |  |  |  | 100 | 1.000 |
| affective | weak_late_calibration_only | 0.5072 | 0.0046 | 0.5243 | 0.0242 | 0.5632 | 0.5632 | 0.30 | 0.5234 | 100 | 1.000 |
| experience | weak_late_calibration_only | 0.5046 | 0.0078 | 0.5419 | 0.0429 | 0.5380 | 0.5380 | 0.40 | 0.5275 | 100 | 1.000 |
| sleep | within_run_only_or_stress_control | 0.5058 | 0.0069 | 0.5166 | 0.0183 | 0.5325 | 0.5233 |  | 0.5183 | 100 | 1.000 |
| xp2 | within_run_only_or_stress_control | 0.5054 | 0.0040 | 0.5492 | 0.0349 | 0.5461 | 0.5408 |  | 0.5125 | 76 | 0.868 |

## Interpretation

- Affective is the best current dataset, but it only becomes strict-positive after about 30 percent subject calibration. That is useful as a canary, not enough as a general foundation-model claim.
- Experience is weaker and time-dominated in the non-residual metrics. It only crosses the strict threshold at 40 percent calibration, so it is a late-calibration support point rather than an independent strong dataset.
- XP2 passes within-run tests but fails subject adaptation and has partial Schaefer coverage. Treat it as a stress/negative-control dataset, not as positive evidence.
- Sleep now has a soft waveform within-run signal, but nested confirmation is weak and subject adaptation only reaches a soft pass at 40 percent calibration. It is a promising reprocessing/modeling target, not a current positive.
- NatView, speeded, and gradCPT fail the current strict gates. NatView remains scientifically important, but this target/cache does not currently provide positive EEG-to-fMRI evidence.

## Route Recommendation

Do not continue as blind pooled LaBraM/NeuroSTORM distillation. The evidence is below the multi-dataset reproducibility bar.

The paper-capable route is narrower: reliability-gated paired EEG-fMRI distillation. The story should be that public paired datasets are valuable, but naive cross-dataset distillation fails; with reliability gates and subject adaptation, only specific datasets show conditional signal. To become a method paper, the next experiment must add at least one more independent dataset that passes early calibration, most likely Sleep after better EEG preprocessing/conditioning.

## Artifacts

- Gate matrix: `results/dataset_reproducibility_gate_20260509/gate_matrix.csv`
- Inventory gate: `results/dataset_reproducibility_gate_20260509/inventory_gate.csv`
- Bandpower subject canary: `results/dataset_gate_bandpower_subject_residual_20260509`
- Bandpower within-run canary: `results/dataset_gate_bandpower_withinrun_residual_20260509`
