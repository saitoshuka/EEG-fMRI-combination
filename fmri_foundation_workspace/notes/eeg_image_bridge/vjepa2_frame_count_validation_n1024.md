# V-JEPA2 Repeated-Frame Count Validation

Setup: still images are repeated to N frames and encoded by frozen V-JEPA2; a ridge probe predicts raw TRIBE k256 cortical prototype targets; EEG residual predictability is tested with the fast ATM embedding probe.

| frames | test cosine vs 64 | train cosine vs 64 | V-JEPA2->raw corr | V-JEPA2->raw rank | EEG residual rank | shifted | delta | latent delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.9231 | 0.9220 | 0.8150 | 0.9317 | 0.5598 | 0.5033 | 0.0565 | 0.0909 |
| 16 | 0.9708 | 0.9703 | 0.8178 | 0.9327 | 0.5542 | 0.4967 | 0.0576 | 0.0964 |
| 32 | 0.9934 | 0.9933 | 0.8226 | 0.9355 | 0.5594 | 0.4964 | 0.0630 | 0.0905 |
| 64 | 1.0000 | 1.0000 | 0.8234 | 0.9349 | 0.5572 | 0.4890 | 0.0682 | 0.1004 |

Decision guide:

- If 8/16/32-frame features have high cosine to 64-frame features and similar raw-prediction/residual metrics, use the lowest stable frame count for future still-image feature extraction.
- If fewer frames reduce V-JEPA2->raw corr or change residual EEG rank materially, keep 64 frames for the main result and only mention lower-frame runs as speed controls.
