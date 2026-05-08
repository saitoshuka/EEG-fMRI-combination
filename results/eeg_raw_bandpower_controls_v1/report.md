# Raw EEG Bandpower Controls v1

This diagnostic extracts simple relative bandpower features directly from raw EEG caches and reruns EEG-only positive controls.

Feature modes:

- `summary`: mean/std/quantiles of relative bandpower across channels.
- `spatial`: fixed montage channel x band relative bandpower.

## affective

| control | split | score | chance/random | aux | n |
| --- | --- | ---: | ---: | ---: | ---: |
| spatial:run_id_from_eeg | within_run_block | 0.9935 +/- 0.0028 | 0.0476 | 0.9933 +/- 0.0029 | 3 |
| spatial:subject_id_from_eeg | within_run_block | 0.9935 +/- 0.0028 | 0.0476 | 0.9933 +/- 0.0029 | 3 |
| spatial:temporal_autocorr_lag1 | no_train | 0.9689 +/- 0.0000 | 0.6865 | 0.2823 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag16 | no_train | 0.8464 +/- 0.0000 | 0.6887 | 0.1577 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag4 | no_train | 0.8786 +/- 0.0000 | 0.6845 | 0.1941 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag64 | no_train | 0.7952 +/- 0.0000 | 0.6894 | 0.1058 +/- 0.0000 | 1 |
| spatial:time_bin_from_eeg | subject_heldout | 0.1744 +/- 0.0111 | 0.1250 | 0.1745 +/- 0.0112 | 3 |
| spatial:time_bin_from_eeg | within_run_block | 0.1648 +/- 0.0219 | 0.1250 | 0.1692 +/- 0.0266 | 3 |
| spatial:time_frac_from_eeg | subject_heldout | 0.0980 +/- 0.1508 | 0.0000 | -1.0027 +/- 0.6573 | 3 |
| spatial:time_frac_from_eeg | within_run_block | 0.4405 +/- 0.0464 | 0.0000 | 0.1467 +/- 0.0597 | 3 |
| summary:run_id_from_eeg | within_run_block | 0.8256 +/- 0.0144 | 0.0476 | 0.8240 +/- 0.0143 | 3 |
| summary:subject_id_from_eeg | within_run_block | 0.8256 +/- 0.0144 | 0.0476 | 0.8240 +/- 0.0143 | 3 |
| summary:temporal_autocorr_lag1 | no_train | 0.9482 +/- 0.0000 | 0.6387 | 0.3095 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag16 | no_train | 0.7793 +/- 0.0000 | 0.6436 | 0.1357 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag4 | no_train | 0.8191 +/- 0.0000 | 0.6361 | 0.1830 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag64 | no_train | 0.7267 +/- 0.0000 | 0.6446 | 0.0821 +/- 0.0000 | 1 |
| summary:time_bin_from_eeg | subject_heldout | 0.1746 +/- 0.0144 | 0.1250 | 0.1747 +/- 0.0146 | 3 |
| summary:time_bin_from_eeg | within_run_block | 0.1268 +/- 0.0080 | 0.1250 | 0.1304 +/- 0.0106 | 3 |
| summary:time_frac_from_eeg | subject_heldout | 0.2859 +/- 0.0585 | 0.0000 | 0.0138 +/- 0.0618 | 3 |
| summary:time_frac_from_eeg | within_run_block | 0.3683 +/- 0.0144 | 0.0000 | 0.1319 +/- 0.0088 | 3 |

QC means:

- line60_ratio: 0.744
- alpha_rel_power: 0.857
- alpha_peak_hz: 6.607
- flat_channel_fraction: 0.000
- high_rms_channel_fraction: 0.000

## natview

| control | split | score | chance/random | aux | n |
| --- | --- | ---: | ---: | ---: | ---: |
| spatial:run_id_from_eeg | within_run_block | 1.0000 +/- 0.0000 | 0.0455 | 1.0000 +/- 0.0000 | 3 |
| spatial:subject_id_from_eeg | within_run_block | 1.0000 +/- 0.0000 | 0.0455 | 1.0000 +/- 0.0000 | 3 |
| spatial:temporal_autocorr_lag1 | no_train | 0.9632 +/- 0.0000 | 0.8018 | 0.1614 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag16 | no_train | 0.8279 +/- 0.0000 | 0.7958 | 0.0321 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag4 | no_train | 0.8584 +/- 0.0000 | 0.7962 | 0.0622 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag64 | no_train | 0.8031 +/- 0.0000 | 0.7989 | 0.0043 +/- 0.0000 | 1 |
| spatial:time_bin_from_eeg | subject_heldout | 0.1313 +/- 0.0106 | 0.1250 | 0.1314 +/- 0.0108 | 3 |
| spatial:time_bin_from_eeg | within_run_block | 0.0169 +/- 0.0096 | 0.1250 | 0.0160 +/- 0.0092 | 3 |
| spatial:time_frac_from_eeg | subject_heldout | -0.0292 +/- 0.0109 | 0.0000 | -0.4034 +/- 0.2160 | 3 |
| spatial:time_frac_from_eeg | within_run_block | -0.4687 +/- 0.0722 | 0.0000 | -0.9423 +/- 0.3170 | 3 |
| summary:run_id_from_eeg | within_run_block | 0.7634 +/- 0.0166 | 0.0455 | 0.7658 +/- 0.0135 | 3 |
| summary:subject_id_from_eeg | within_run_block | 0.7634 +/- 0.0166 | 0.0455 | 0.7658 +/- 0.0135 | 3 |
| summary:temporal_autocorr_lag1 | no_train | 0.9298 +/- 0.0000 | 0.6653 | 0.2645 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag16 | no_train | 0.7082 +/- 0.0000 | 0.6537 | 0.0545 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag4 | no_train | 0.7566 +/- 0.0000 | 0.6557 | 0.1009 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag64 | no_train | 0.6712 +/- 0.0000 | 0.6569 | 0.0142 +/- 0.0000 | 1 |
| summary:time_bin_from_eeg | subject_heldout | 0.1431 +/- 0.0051 | 0.1250 | 0.1432 +/- 0.0050 | 3 |
| summary:time_bin_from_eeg | within_run_block | 0.0510 +/- 0.0071 | 0.1250 | 0.0522 +/- 0.0071 | 3 |
| summary:time_frac_from_eeg | subject_heldout | 0.0447 +/- 0.0305 | 0.0000 | -0.0366 +/- 0.0031 | 3 |
| summary:time_frac_from_eeg | within_run_block | -0.2669 +/- 0.0673 | 0.0000 | -0.4295 +/- 0.1106 | 3 |

QC means:

- line60_ratio: 0.676
- alpha_rel_power: 1.673
- alpha_peak_hz: 7.261
- flat_channel_fraction: 0.000
- high_rms_channel_fraction: 0.000

## sleep

| control | split | score | chance/random | aux | n |
| --- | --- | ---: | ---: | ---: | ---: |
| spatial:run_id_from_eeg | within_run_block | 0.9501 +/- 0.0316 | 0.0303 | 0.9448 +/- 0.0361 | 3 |
| spatial:subject_id_from_eeg | within_run_block | 0.9501 +/- 0.0316 | 0.0303 | 0.9448 +/- 0.0361 | 3 |
| spatial:temporal_autocorr_lag1 | no_train | 0.9762 +/- 0.0000 | 0.7823 | 0.1939 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag16 | no_train | 0.8607 +/- 0.0000 | 0.7785 | 0.0822 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag4 | no_train | 0.9018 +/- 0.0000 | 0.7795 | 0.1222 +/- 0.0000 | 1 |
| spatial:temporal_autocorr_lag64 | no_train | 0.8079 +/- 0.0000 | 0.7831 | 0.0248 +/- 0.0000 | 1 |
| spatial:time_bin_from_eeg | subject_heldout | 0.1553 +/- 0.0021 | 0.1250 | 0.1553 +/- 0.0018 | 3 |
| spatial:time_bin_from_eeg | within_run_block | 0.0596 +/- 0.0113 | 0.1250 | 0.0728 +/- 0.0331 | 3 |
| spatial:time_frac_from_eeg | subject_heldout | 0.1758 +/- 0.0630 | 0.0000 | -0.6382 +/- 0.6007 | 3 |
| spatial:time_frac_from_eeg | within_run_block | 0.0177 +/- 0.1073 | 0.0000 | -0.2921 +/- 0.1083 | 3 |
| summary:run_id_from_eeg | within_run_block | 0.8980 +/- 0.0254 | 0.0303 | 0.8898 +/- 0.0317 | 3 |
| summary:subject_id_from_eeg | within_run_block | 0.8980 +/- 0.0254 | 0.0303 | 0.8898 +/- 0.0317 | 3 |
| summary:temporal_autocorr_lag1 | no_train | 0.9579 +/- 0.0000 | 0.7081 | 0.2497 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag16 | no_train | 0.7928 +/- 0.0000 | 0.7039 | 0.0889 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag4 | no_train | 0.8409 +/- 0.0000 | 0.7075 | 0.1335 +/- 0.0000 | 1 |
| summary:temporal_autocorr_lag64 | no_train | 0.7352 +/- 0.0000 | 0.7081 | 0.0271 +/- 0.0000 | 1 |
| summary:time_bin_from_eeg | subject_heldout | 0.1606 +/- 0.0167 | 0.1250 | 0.1606 +/- 0.0167 | 3 |
| summary:time_bin_from_eeg | within_run_block | 0.0720 +/- 0.0188 | 0.1250 | 0.0918 +/- 0.0445 | 3 |
| summary:time_frac_from_eeg | subject_heldout | 0.2624 +/- 0.0662 | 0.0000 | 0.0106 +/- 0.0585 | 3 |
| summary:time_frac_from_eeg | within_run_block | 0.1399 +/- 0.0780 | 0.0000 | -0.0507 +/- 0.0345 | 3 |

QC means:

- line60_ratio: 0.004
- alpha_rel_power: 0.185
- alpha_peak_hz: 6.788
- flat_channel_fraction: 0.004
- high_rms_channel_fraction: 0.000

## Interpretation

If raw bandpower has stronger cross-subject time/stimulus signal than LaBraM features, the bottleneck is likely the frozen LaBraM representation. If both are near chance, the bottleneck is likely raw MR-EEG quality or preprocessing.
