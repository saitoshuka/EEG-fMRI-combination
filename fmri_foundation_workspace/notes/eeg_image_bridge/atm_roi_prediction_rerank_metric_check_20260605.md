# ATM ROI Prediction Rerank Metric Check

Question: can the trained spatial branch improve standard 200-image image
retrieval when used as a second-stage cortical reranker?

Protocol:

- Export image-level test predictions from existing ATM spatial checkpoints.
- Baseline score: semantic branch EEG->CLIP prediction against CLIP image
  targets.
- Cortical score: EEG->ROI prediction against the corresponding image-derived
  ROI/prototype target.
- Rerank only the semantic top-k candidates using row-zscored semantic score
  plus a weighted ROI score.
- Controls: shifted ROI prediction, permutation-null ROI prediction, and
  20-split test-set CV as a diagnostic for hyperparameter stability.

## Results

| exported model | target | baseline top1 | best rerank top1 | full-grid top1 gain | CV top1 gain | CV top5 gain | CV rank gain | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| proto256 query | residual k256 | 0.425 | 0.420 | -0.005 | -0.0025 | -0.0050 | -0.00036 | no metric gain |
| proto256 pooled | residual k256 | 0.405 | 0.410 | +0.005 | +0.0015 | +0.0065 | -0.00001 | negligible |
| parcel38 query | residual 38 | 0.430 | 0.450 | +0.020 | -0.0020 | -0.0005 | -0.00013 | full-grid gain not CV-stable |
| parcel38 pooled | raw strong 38 | 0.300 | 0.300 | +0.000 | -0.0040 | +0.0065 | -0.00016 | no rank/top1 gain |

## Interpretation

The ROI branches contain real target signal, as shown by ROI-only rank above
chance and by prior shifted-null controls, but the signal is not yet strong or
well-calibrated enough to act as a reliable retrieval reranker.  The strongest
full-grid top1 gain is residual parcel38 (+0.020), but split-CV selects no
stable improvement.  The fine proto256 residual target is learnable as an ROI
target, yet it does not improve retrieval in this simple rerank form.

Decision: stop spending compute on small rerank-grid variants of the current
ROI predictions.  For main-conference metrics, the next high-value direction is
not more post-hoc reranking of these heads, but a stronger architecture or
target:

- dual-head training where the pooled head optimizes retrieval and the query
  head is explicitly regularized for cortical identity;
- finer visual surface/prototype targets with locality/hierarchy constraints;
- validation-selected checkpointing and an independent validation route;
- generation-side comparison if retrieval remains saturated.

Artifacts:

- `fmri_foundation_workspace/scripts/rerank_atm_roi_predictions.py`
- `fmri_foundation_workspace/notes/eeg_image_bridge/proto256_query_residual_rerank_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/proto256_pooled_residual_rerank_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/residual_parcel38_rerank_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/raw_strong_parcel38_rerank_20260605.md`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/`
