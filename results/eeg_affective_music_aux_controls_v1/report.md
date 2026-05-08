# Affective Music Auxiliary Controls

EEG-only bandpower features are used to predict Affective auxiliary channels, especially the `MUSIC` channel envelope/RMS.

| target | model | split | corr | R2 | n |
| --- | --- | --- | ---: | ---: | ---: |
| ft_arousal_mean | spatial_bandpower | subject_heldout | 0.0127 +/- 0.0088 | -0.0197 +/- 0.0100 | 3 |
| ft_arousal_mean | spatial_bandpower | within_run_block | 0.0288 +/- 0.0079 | -0.0009 +/- 0.0016 | 3 |
| ft_arousal_mean | summary_bandpower | subject_heldout | 0.0349 +/- 0.0031 | -0.0013 +/- 0.0019 | 3 |
| ft_arousal_mean | summary_bandpower | within_run_block | 0.0323 +/- 0.0066 | 0.0001 +/- 0.0013 | 3 |
| ft_arousal_mean | time_ridge | subject_heldout | -0.0072 +/- 0.0106 | -0.0023 +/- 0.0012 | 3 |
| ft_arousal_mean | time_ridge | within_run_block | 0.0185 +/- 0.0143 | -0.0000 +/- 0.0010 | 3 |
| ft_valence_mean | spatial_bandpower | subject_heldout | 0.0057 +/- 0.0014 | -0.0085 +/- 0.0006 | 3 |
| ft_valence_mean | spatial_bandpower | within_run_block | 0.0254 +/- 0.0057 | -0.0017 +/- 0.0004 | 3 |
| ft_valence_mean | summary_bandpower | subject_heldout | 0.0214 +/- 0.0104 | -0.0005 +/- 0.0014 | 3 |
| ft_valence_mean | summary_bandpower | within_run_block | 0.0190 +/- 0.0133 | -0.0009 +/- 0.0009 | 3 |
| ft_valence_mean | time_ridge | subject_heldout | 0.0220 +/- 0.0048 | 0.0001 +/- 0.0003 | 3 |
| ft_valence_mean | time_ridge | within_run_block | 0.0197 +/- 0.0046 | -0.0001 +/- 0.0003 | 3 |
| music_absmean | spatial_bandpower | subject_heldout | 0.0611 +/- 0.0234 | -0.2087 +/- 0.1472 | 3 |
| music_absmean | spatial_bandpower | within_run_block | 0.1401 +/- 0.0117 | 0.0156 +/- 0.0044 | 3 |
| music_absmean | summary_bandpower | subject_heldout | 0.1069 +/- 0.0153 | 0.0012 +/- 0.0085 | 3 |
| music_absmean | summary_bandpower | within_run_block | 0.1351 +/- 0.0268 | 0.0170 +/- 0.0086 | 3 |
| music_absmean | time_ridge | subject_heldout | 0.2323 +/- 0.0254 | 0.0532 +/- 0.0127 | 3 |
| music_absmean | time_ridge | within_run_block | 0.2341 +/- 0.0081 | 0.0526 +/- 0.0031 | 3 |
| music_rms | spatial_bandpower | subject_heldout | 0.0668 +/- 0.0266 | -0.2183 +/- 0.1581 | 3 |
| music_rms | spatial_bandpower | within_run_block | 0.1542 +/- 0.0134 | 0.0203 +/- 0.0053 | 3 |
| music_rms | summary_bandpower | subject_heldout | 0.1108 +/- 0.0135 | -0.0002 +/- 0.0075 | 3 |
| music_rms | summary_bandpower | within_run_block | 0.1461 +/- 0.0278 | 0.0201 +/- 0.0094 | 3 |
| music_rms | time_ridge | subject_heldout | 0.2449 +/- 0.0266 | 0.0592 +/- 0.0140 | 3 |
| music_rms | time_ridge | within_run_block | 0.2466 +/- 0.0080 | 0.0585 +/- 0.0031 | 3 |
| music_std | spatial_bandpower | subject_heldout | 0.0669 +/- 0.0267 | -0.2183 +/- 0.1586 | 3 |
| music_std | spatial_bandpower | within_run_block | 0.1543 +/- 0.0132 | 0.0204 +/- 0.0052 | 3 |
| music_std | summary_bandpower | subject_heldout | 0.1106 +/- 0.0133 | -0.0003 +/- 0.0073 | 3 |
| music_std | summary_bandpower | within_run_block | 0.1458 +/- 0.0277 | 0.0200 +/- 0.0094 | 3 |
| music_std | time_ridge | subject_heldout | 0.2446 +/- 0.0265 | 0.0591 +/- 0.0139 | 3 |
| music_std | time_ridge | within_run_block | 0.2463 +/- 0.0080 | 0.0584 +/- 0.0031 | 3 |
| trialtype_rms | spatial_bandpower | subject_heldout | 0.0918 +/- 0.0260 | -0.1814 +/- 0.1604 | 3 |
| trialtype_rms | spatial_bandpower | within_run_block | 0.1823 +/- 0.0171 | 0.0304 +/- 0.0061 | 3 |
| trialtype_rms | summary_bandpower | subject_heldout | 0.1266 +/- 0.0167 | 0.0040 +/- 0.0091 | 3 |
| trialtype_rms | summary_bandpower | within_run_block | 0.1652 +/- 0.0322 | 0.0263 +/- 0.0116 | 3 |
| trialtype_rms | time_ridge | subject_heldout | 0.2436 +/- 0.0274 | 0.0581 +/- 0.0140 | 3 |
| trialtype_rms | time_ridge | within_run_block | 0.2493 +/- 0.0077 | 0.0596 +/- 0.0026 | 3 |

Interpretation:

- `time_ridge` is an upper-bound nuisance baseline: if the auxiliary target is mostly deterministic over the shared run timeline, time can predict it.
- The EEG models matter only if bandpower predicts auxiliary targets across held-out subjects with positive R2 or robust correlation.
