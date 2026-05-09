# ROI-Query TRIBE Branch

Cache: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/cache/eeg_image_bridge/thing_eeg_token_cache_train256_test200.pt`
Train targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_n256.npz`
Test targets: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz`
Device: `cuda`
Components/queries: `32`; explained variance: `0.9978`

This is a minimal token-level test of the proposed spatial branch. The queries are data-driven TRIBE cortical PCA component queries, not anatomical V1/V2/V4/IT labels yet.

Frozen ATM baseline retrieval on test200:

- top1 `0.5200`, top5 `0.8550`, rank percentile `0.9854`

| model | val latent rank | test latent rank | shifted | top10/w0.5 top1 | top10/w0.5 top5 | top100/w0.7 top1 | top100/w0.7 top5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 0.5557 | 0.4963 | 0.4831 | 0.4300 | 0.8300 | 0.3550 | 0.6850 |
| query | 0.5173 | 0.4974 | 0.5046 | 0.3650 | 0.8100 | 0.3200 | 0.6800 |

## Readout

- `global` uses the same EEG token encoder but predicts the TRIBE latent from mean-pooled tokens.
- `query` uses learnable cortical-component queries that cross-attend EEG tokens and predict one latent component per query.
- The rerank columns keep the frozen ATM retrieval embedding unchanged and use the predicted TRIBE latent only as a second-stage brain-space score.

## Conclusion

This minimal ROI-query branch did not improve retrieval or TRIBE latent
prediction.

The raw EEG token models are near chance on unseen test200 TRIBE latent
retrieval:

```text
global token pooling: rank percentile 0.4963
ROI-query attention:  rank percentile 0.4974
shifted null:         around 0.48-0.50
```

Reranking with these raw-token predictions hurts the frozen ATM retrieval
baseline rather than improving it. The best tested query rerank drops top1 from
`0.5200` to `0.3650`.

I also ran a quick ATM-token sanity check where the queries attended to the
already-trained ATM subject/repeat embeddings instead of raw EEG patches. That
was better than raw EEG but still not good enough:

```text
ATM-token global MLP: test TRIBE latent rank percentile about 0.9100
ATM-token query head: test TRIBE latent rank percentile about 0.5822
closed-form ATM ridge baseline from prior run: 0.9561
```

So the negative result is informative: with only `256` TRIBE-labeled train
images, training a new token-level spatial branch is underpowered and less
stable than the simple closed-form head on frozen ATM embeddings. The ROI-query
idea is still architecturally reasonable, but it needs either many more TRIBE
targets or access to a stronger pretrained ATM intermediate feature stream.
