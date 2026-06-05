# Real-fMRI Visual64 Query vs Pooled ATM Check

Date: 2026-06-05

## Setup

- Target: THINGS-fMRI external validation visual64 real-fMRI ROI target.
- Training images: 6330 overlapping THINGS images.
- EEG input: THINGS-EEG, 10 subjects, subject_mode none.
- Backbone: ATM, d_model 256, heads 4, layers 1.
- Loss: CLIP + visual64 ROI, lambda_roi 0.05, lambda_spatial 0.05.
- Compared heads:
  - `query`: ordered ROI-query supervision.
  - `pooled`: pooled spatial head.

## Multiseed Metrics

| seed | head | final CLIP top1 | final CLIP top5 | final ROI rank | shifted | delta | best ROI epoch | best ROI rank | best shifted | best delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | pooled | 0.2987 | 0.7792 | 0.7669 | 0.5159 | 0.2510 | 25 | 0.7669 | 0.5159 | 0.2510 |
| 11 | query | 0.2468 | 0.7792 | 0.7244 | 0.4911 | 0.2333 | 25 | 0.7244 | 0.4911 | 0.2333 |
| 33 | pooled | 0.3117 | 0.7403 | 0.7739 | 0.5072 | 0.2667 | 25 | 0.7739 | 0.5072 | 0.2667 |
| 33 | query | 0.3247 | 0.7532 | 0.7264 | 0.4928 | 0.2336 | 24 | 0.7384 | 0.4723 | 0.2661 |
| 77 | pooled | 0.2727 | 0.7662 | 0.7546 | 0.5149 | 0.2397 | 21 | 0.7592 | 0.5200 | 0.2392 |
| 77 | query | 0.2987 | 0.7403 | 0.6931 | 0.4691 | 0.2240 | 25 | 0.6931 | 0.4691 | 0.2240 |

Aggregate means:

| head | n | final ROI rank | final delta vs shifted | best ROI rank | best delta vs shifted | final CLIP top1 | final CLIP top5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | 3 | 0.7652 | 0.2525 | 0.7667 | 0.2523 | 0.2944 | 0.7619 |
| query | 3 | 0.7146 | 0.2303 | 0.7186 | 0.2411 | 0.2900 | 0.7576 |

## Interpretation

The real-fMRI visual64 signal is clearly above shifted-null for both heads, so the
stimulus-mediated EEG model is not random on this external fMRI target. However,
the ordered ROI-query head does not improve the real-fMRI visual64 rank over the
pooled head in this setting. Pooled is higher in all three seeds for final ROI
rank and has the better aggregate best ROI rank.

This means the current query constraint is useful for interpretability and
visualization, but it should not be claimed as the best-performing real-fMRI
alignment head yet. For metric-driven claims, pooled or pooled-plus-fusion should
remain the primary head. Query-specific analyses should be framed as mechanism
evidence rather than the main performance driver unless a later finer-grained
target or architecture variant reverses this trend.

## Decision

Do not launch 60k solely to scale the current ordered query head. The next
decision-value experiment should first test in-model late fusion with the pooled
proto256/real-fMRI signal at 16k, because existing post-hoc fusion is the only
route that has shown direct retrieval gains over semantic-only ATM.
