# EEG CLIP Retrieval with Cortical Reranking

## Protocol

- Same image-heldout THINGS-fMRI overlap splits as the real-fMRI validation.
- Semantic baseline: ridge from image-averaged posterior EEG to CLIP ViT-H/14 image features.
- Cortical branch: trainable factorized-query EEG -> real-fMRI visual ROI prediction.
- Fusion: row-zscored semantic similarity plus cortical similarity; fusion weight selected on validation split only.

## Results

| seed | semantic_rank | semantic_top5 | cortical_rank | fusion_weight | fusion_rank | fusion_top5 | fusion_minus_semantic_rank | oracle_weight | oracle_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.9592 | 0.3460 | 0.6604 | 0.0500 | 0.9598 | 0.3500 | 0.0006 | 0.1000 | 0.9599 |
| 11 | 0.9586 | 0.3330 | 0.6370 | 0.1000 | 0.9589 | 0.3340 | 0.0003 | 0.0500 | 0.9591 |
| 77 | 0.9590 | 0.3310 | 0.6495 | 0.0500 | 0.9597 | 0.3360 | 0.0007 | 0.1000 | 0.9598 |
| 101 | 0.9598 | 0.3280 | 0.6374 | 0.1000 | 0.9603 | 0.3410 | 0.0005 | 0.0500 | 0.9604 |
| mean | 0.9592 | 0.3345 | 0.6461 | 0.0750 | 0.9597 | 0.3402 | 0.0005 | 0.0750 | 0.9598 |
| std | 0.0005 | 0.0079 | 0.0112 | 0.0289 | 0.0006 | 0.0071 | 0.0002 | 0.0289 | 0.0005 |

## Interpretation

- Semantic EEG->CLIP retrieval is already strong on the overlap splits.
- The cortical branch alone retrieves images above chance because it predicts real visual-fMRI patterns.
- Validation-selected fusion currently gives mean rank 0.9597 vs semantic-only 0.9592. This is an improvement under the current fusion protocol.
- The oracle column shows whether a useful fusion weight exists on heldout; it should not be used for claims, only for diagnosing whether validation selection is the bottleneck.

Artifacts:

- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_vjepa_retrieval_cortical_rerank/summary.csv`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_vjepa_retrieval_cortical_rerank/summary.json`
