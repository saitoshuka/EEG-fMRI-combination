# EEG Positive Controls v1

These diagnostics use frozen LaBraM EEG features only. They do not use fMRI targets.

Interpretation:

- High subject/run decoding means the representation contains stable fingerprints, which may be subject/session artifacts rather than task-relevant neural signal.
- High within-run time decoding means the representation contains slow temporal structure or drift.
- Cross-subject time decoding is the stricter proxy for shared stimulus-locked EEG signal; near-chance scores here support the EEG-bottleneck hypothesis.
- Temporal autocorrelation above random means the features are smooth, but smoothness alone can be artifact/drift.

## affective

| control | split | score | chance/random | aux | n folds |
| --- | --- | ---: | ---: | ---: | ---: |
| run_id_from_eeg | within_run_block | 0.9687 +/- 0.0022 | 0.0476 | 0.9681 +/- 0.0022 | 3 |
| subject_id_from_eeg | within_run_block | 0.9687 +/- 0.0022 | 0.0476 | 0.9681 +/- 0.0022 | 3 |
| temporal_autocorr_lag1 | no_train | 0.8573 +/- 0.0000 | 0.4220 | 0.4353 +/- 0.0000 | 1 |
| temporal_autocorr_lag16 | no_train | 0.5161 +/- 0.0000 | 0.4210 | 0.0951 +/- 0.0000 | 1 |
| temporal_autocorr_lag4 | no_train | 0.5534 +/- 0.0000 | 0.4223 | 0.1311 +/- 0.0000 | 1 |
| temporal_autocorr_lag64 | no_train | 0.4860 +/- 0.0000 | 0.4222 | 0.0638 +/- 0.0000 | 1 |
| time_bin_from_eeg | subject_heldout | 0.1675 +/- 0.0048 | 0.1250 | 0.1676 +/- 0.0048 | 3 |
| time_bin_from_eeg | within_run_block | 0.1262 +/- 0.0079 | 0.1250 | 0.1293 +/- 0.0110 | 3 |
| time_frac_from_eeg | subject_heldout | 0.1812 +/- 0.0570 | 0.0000 | -0.1995 +/- 0.0397 | 3 |
| time_frac_from_eeg | within_run_block | 0.2913 +/- 0.0287 | 0.0000 | 0.0527 +/- 0.0290 | 3 |

## natview_neurostorm

| control | split | score | chance/random | aux | n folds |
| --- | --- | ---: | ---: | ---: | ---: |
| run_id_from_eeg | within_run_block | 0.9555 +/- 0.0053 | 0.0455 | 0.9521 +/- 0.0046 | 3 |
| subject_id_from_eeg | within_run_block | 0.9564 +/- 0.0058 | 0.0455 | 0.9534 +/- 0.0058 | 3 |
| temporal_autocorr_lag1 | no_train | 0.6684 +/- 0.0000 | 0.4444 | 0.2240 +/- 0.0000 | 1 |
| temporal_autocorr_lag16 | no_train | 0.4891 +/- 0.0000 | 0.4436 | 0.0454 +/- 0.0000 | 1 |
| temporal_autocorr_lag4 | no_train | 0.5267 +/- 0.0000 | 0.4415 | 0.0852 +/- 0.0000 | 1 |
| temporal_autocorr_lag64 | no_train | 0.4471 +/- 0.0000 | 0.4404 | 0.0067 +/- 0.0000 | 1 |
| time_bin_from_eeg | subject_heldout | 0.1315 +/- 0.0051 | 0.1250 | 0.1318 +/- 0.0049 | 3 |
| time_bin_from_eeg | within_run_block | 0.0487 +/- 0.0092 | 0.1250 | 0.0496 +/- 0.0112 | 3 |
| time_frac_from_eeg | subject_heldout | 0.0225 +/- 0.0293 | 0.0000 | -0.1039 +/- 0.0465 | 3 |
| time_frac_from_eeg | within_run_block | -0.4229 +/- 0.0849 | 0.0000 | -0.5533 +/- 0.0473 | 3 |

## sleep

| control | split | score | chance/random | aux | n folds |
| --- | --- | ---: | ---: | ---: | ---: |
| run_id_from_eeg | within_run_block | 0.9429 +/- 0.0338 | 0.0303 | 0.9348 +/- 0.0394 | 3 |
| subject_id_from_eeg | within_run_block | 0.9429 +/- 0.0338 | 0.0303 | 0.9348 +/- 0.0394 | 3 |
| temporal_autocorr_lag1 | no_train | 0.9526 +/- 0.0000 | 0.7971 | 0.1555 +/- 0.0000 | 1 |
| temporal_autocorr_lag16 | no_train | 0.8939 +/- 0.0000 | 0.7864 | 0.1075 +/- 0.0000 | 1 |
| temporal_autocorr_lag4 | no_train | 0.9347 +/- 0.0000 | 0.7876 | 0.1471 +/- 0.0000 | 1 |
| temporal_autocorr_lag64 | no_train | 0.8379 +/- 0.0000 | 0.7977 | 0.0401 +/- 0.0000 | 1 |
| time_bin_from_eeg | subject_heldout | 0.1680 +/- 0.0076 | 0.1250 | 0.1692 +/- 0.0077 | 3 |
| time_bin_from_eeg | within_run_block | 0.0630 +/- 0.0077 | 0.1250 | 0.0750 +/- 0.0252 | 3 |
| time_frac_from_eeg | subject_heldout | 0.1382 +/- 0.0815 | 0.0000 | -1.5493 +/- 0.7523 | 3 |
| time_frac_from_eeg | within_run_block | 0.1483 +/- 0.0731 | 0.0000 | -0.1578 +/- 0.1032 | 3 |

## Bottom Line

The key pass/fail criterion for useful EEG is not subject/run decoding; that can be driven by stable artifacts. The important criterion is whether EEG features predict shared time/stimulus structure across held-out subjects, and whether that signal is stronger than within-run drift/fingerprints.
