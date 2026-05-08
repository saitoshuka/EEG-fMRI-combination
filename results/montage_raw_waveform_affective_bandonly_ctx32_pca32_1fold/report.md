# Montage Raw Waveform EEG-to-fMRI Diagnostic

Small montage-aware encoder using raw EEG waveform patches, per-channel bandpower, and approximate 10-20 coordinates.

- Raw cache: `data/montage_waveform_affective_v1/affective_waveform_cache.npz`
- Input mode: `band_only`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9687 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9687 | 5376 | 0.0046 | -0.0028 | -0.0000 | 0.0047 | 0.0186 | 0.0243 | 0.4989 | -0.0019 |
| 1 | band_ridge_last | real | 9687 | 5376 | 0.0136 | 0.0229 | -0.0076 | 0.0033 | 0.0229 | 0.0254 | 0.5105 | 0.0235 |
| 1 | montage_band_only | real | 9687 | 5376 | 0.0088 | 0.0033 | -0.0008 | 0.0032 | 0.0210 | 0.0241 | 0.5077 | 0.0025 |
| 1 | montage_band_only | shifted_null | 9687 | 5376 | -0.0024 | 0.0003 | -0.0011 | 0.0043 | 0.0205 | 0.0239 | 0.4997 | -0.0001 |

Guardrail:

- The neural model should beat `band_ridge_last` and shifted-null.
- If subject-heldout returns to chance, the claim remains within-subject/within-run only.
