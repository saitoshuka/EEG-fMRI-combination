# ATM CLIP Retrieval Baseline

Asset root: `/mnt/c/Users/xinji/Desktop/Image Reconstruction`
Subjects evaluated: `10`
Test images: `200`

## Subject Mean

| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|
| CLIP image | 0.2875 | 0.6170 | 0.9498 | 0.4962 | 0.1249 | -0.0013 |
| CLIP text | 0.0665 | 0.2215 | 0.7857 | 0.4890 | 0.0369 | -0.0015 |

## Mean-Subject Embedding

| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|
| clip_image | 0.5200 | 0.8550 | 0.9854 | 0.4968 | 0.1741 | -0.0015 |
| clip_text | 0.1400 | 0.3400 | 0.8587 | 0.4929 | 0.0516 | -0.0019 |

## Readout

- This evaluates the already trained ATM embeddings, not a new TRIBE-conditioned model.
- The shifted target is a circular mismatch control over the same 200 test images.
- If TRIBE surface predictions are useful as an added teacher, the next experiment should preserve or improve this image-retrieval signal while adding a brain-space alignment head.
