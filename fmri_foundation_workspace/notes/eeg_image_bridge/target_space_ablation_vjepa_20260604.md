# Target-Space Ablation: CLIP vs V-JEPA

This lightweight readout uses existing ATM EEG embeddings averaged over subjects/repeats.
It is a reviewer-risk test, not the final trainable-backbone result.

| train images | model | target/eval | top1 | top5 | rank | shifted | blend | alpha |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | frozen_atm_clip_direct | clip_test | 0.5200 | 0.8550 | 0.9854 | 0.4945 | 0.00 | 0.0 |
| 1024 | frozen_plus_atm_to_clip_ridge | clip_test | 0.5350 | 0.8550 | 0.9855 | 0.4945 | 0.05 | 10.0 |
| 1024 | atm_to_vjepa_ridge | vjepa_test | 0.3050 | 0.5350 | 0.9115 | 0.4883 | 1.00 | 100.0 |
| 1024 | atm_to_vjepa_to_clip | clip_test | 0.1050 | 0.3800 | 0.8588 | 0.4952 | 1.00 | 100.0 |
| 1024 | frozen_plus_atm_to_vjepa_to_clip | clip_test | 0.5350 | 0.8600 | 0.9855 | 0.4944 | 0.05 | 100.0 |
| 1024 | atm_to_clip_plus_vjepa_concat | combo_test | 0.3400 | 0.6350 | 0.9505 | 0.4872 | 1.00 | 10.0 |
| 4096 | frozen_atm_clip_direct | clip_test | 0.5200 | 0.8550 | 0.9854 | 0.4945 | 0.00 | 0.0 |
| 4096 | frozen_plus_atm_to_clip_ridge | clip_test | 0.4750 | 0.7700 | 0.9706 | 0.5011 | 0.50 | 1.0 |
| 4096 | atm_to_vjepa_ridge | vjepa_test | 0.3500 | 0.6050 | 0.9133 | 0.4855 | 1.00 | 100.0 |
| 4096 | atm_to_vjepa_to_clip | clip_test | 0.1200 | 0.4000 | 0.8537 | 0.5005 | 1.00 | 100.0 |
| 4096 | frozen_plus_atm_to_vjepa_to_clip | clip_test | 0.5500 | 0.8400 | 0.9824 | 0.4969 | 0.30 | 100.0 |
| 4096 | atm_to_clip_plus_vjepa_concat | combo_test | 0.3050 | 0.6450 | 0.9396 | 0.4943 | 1.00 | 10.0 |
| 8192 | frozen_atm_clip_direct | clip_test | 0.5200 | 0.8550 | 0.9854 | 0.4945 | 0.00 | 0.0 |
| 8192 | frozen_plus_atm_to_clip_ridge | clip_test | 0.4800 | 0.7900 | 0.9767 | 0.4977 | 0.50 | 10.0 |
| 8192 | atm_to_vjepa_ridge | vjepa_test | 0.3650 | 0.6150 | 0.9184 | 0.4833 | 1.00 | 1.0 |
| 8192 | atm_to_vjepa_to_clip | clip_test | 0.1350 | 0.4150 | 0.8622 | 0.4964 | 1.00 | 1.0 |
| 8192 | frozen_plus_atm_to_vjepa_to_clip | clip_test | 0.5550 | 0.8550 | 0.9829 | 0.4952 | 0.30 | 1.0 |
| 8192 | atm_to_clip_plus_vjepa_concat | combo_test | 0.3550 | 0.7050 | 0.9505 | 0.4911 | 1.00 | 1.0 |
| 16540 | frozen_atm_clip_direct | clip_test | 0.5200 | 0.8550 | 0.9854 | 0.4945 | 0.00 | 0.0 |
| 16540 | frozen_plus_atm_to_clip_ridge | clip_test | 0.4950 | 0.8050 | 0.9776 | 0.4979 | 0.50 | 1.0 |
| 16540 | atm_to_vjepa_ridge | vjepa_test | 0.3750 | 0.6600 | 0.9243 | 0.4879 | 1.00 | 1.0 |
| 16540 | atm_to_vjepa_to_clip | clip_test | 0.1450 | 0.4300 | 0.8631 | 0.4946 | 1.00 | 1.0 |
| 16540 | frozen_plus_atm_to_vjepa_to_clip | clip_test | 0.5350 | 0.8550 | 0.9853 | 0.4951 | 0.10 | 1.0 |
| 16540 | atm_to_clip_plus_vjepa_concat | combo_test | 0.4000 | 0.7250 | 0.9566 | 0.4936 | 1.00 | 1.0 |

Interpretation guide:

- If `atm_to_vjepa_ridge` has strong V-JEPA-space retrieval but `atm_to_vjepa_to_clip` does not improve CLIP retrieval, V-JEPA alone is not a drop-in replacement for the CLIP reconstruction interface.
- If `frozen_plus_atm_to_vjepa_to_clip` beats `frozen_atm_clip_direct`, the gain can be explained as a better visual target/control unless cortical-query structure adds separate evidence.
- If cortical-query models beat these target-space controls, the cortical-supervision claim becomes much stronger.
