# NatView LaBraM-Style EEG Preprocessing

This rebuild keeps the official-style NeuroSTORM fMRI latents and paired targets unchanged, but replaces EEG with raw NatView EEGLAB data processed according to the LaBraM preprocessing contract.

- Source cache: `data/pooled_raw_schaefer100_neurostorm_official_natview/run_cache`
- Output cache: `data/pooled_raw_schaefer100_neurostorm_official_natview_labram_eeg/run_cache`
- Preprocess kind: `natview_eeglab_bandpass0p1-75_powerline_notch_resample200_uV_v1`
- Runs processed: 22 / 22
- Bandpass: 0.1-75.0 Hz
- Resample: 200.0 Hz
- Notch policy: BIDS PowerLineFrequency

| subject | session | windows | channels | samples | notch | std uV | max abs uV |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| sub-01 | ses-01 | 282 | 59 | 120304 | 60.0 | 11.301 | 172.1 |
| sub-02 | ses-01 | 283 | 60 | 121588 | 60.0 | 6.125 | 102.5 |
| sub-03 | ses-01 | 280 | 56 | 119103 | 60.0 | 10.215 | 151.5 |
| sub-04 | ses-01 | 283 | 58 | 127648 | 60.0 | 8.724 | 140.6 |
| sub-05 | ses-01 | 256 | 58 | 109250 | 60.0 | 14.895 | 239.1 |
| sub-06 | ses-01 | 280 | 51 | 119099 | 60.0 | 8.381 | 109.8 |
| sub-07 | ses-01 | 283 | 56 | 172084 | 60.0 | 7.200 | 104.7 |
| sub-08 | ses-01 | 265 | 60 | 112882 | 60.0 | 8.712 | 111.0 |
| sub-09 | ses-01 | 234 | 45 | 100050 | 60.0 | 7.687 | 159.2 |
| sub-10 | ses-01 | 250 | 61 | 106610 | 60.0 | 9.253 | 142.9 |
| sub-11 | ses-01 | 272 | 60 | 115822 | 60.0 | 11.294 | 144.1 |
| sub-12 | ses-01 | 271 | 57 | 115503 | 60.0 | 6.910 | 110.4 |
| sub-13 | ses-01 | 280 | 50 | 119470 | 60.0 | 8.093 | 115.3 |
| sub-14 | ses-01 | 272 | 61 | 115798 | 60.0 | 11.890 | 200.2 |
| sub-15 | ses-01 | 261 | 60 | 111188 | 60.0 | 9.344 | 178.7 |
| sub-16 | ses-01 | 247 | 54 | 105358 | 60.0 | 6.357 | 136.7 |
| sub-17 | ses-01 | 237 | 55 | 101310 | 60.0 | 9.889 | 150.7 |
| sub-18 | ses-01 | 281 | 56 | 119812 | 60.0 | 13.540 | 193.1 |
| sub-19 | ses-01 | 193 | 46 | 82945 | 60.0 | 24.103 | 345.3 |
| sub-20 | ses-01 | 283 | 58 | 121695 | 60.0 | 7.489 | 117.9 |
| sub-21 | ses-01 | 235 | 60 | 100282 | 60.0 | 9.209 | 126.2 |
| sub-22 | ses-01 | 283 | 59 | 121114 | 60.0 | 15.105 | 233.3 |
