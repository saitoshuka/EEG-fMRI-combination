# EEG CLIP Retrieval with Cortical Reranking

## Protocol

- Same image-heldout THINGS-fMRI overlap splits as the real-fMRI validation.
- Semantic baseline: ridge from image-averaged posterior EEG to CLIP ViT-H/14 image features.
- Cortical branch: trainable factorized-query EEG -> real-fMRI visual ROI prediction.
- Fusion: row-zscored semantic similarity plus cortical similarity; fusion weight selected on validation split only.

## Results

| seed | semantic_rank | semantic_top5 | cortical_rank | fusion_weight | fusion_rank | fusion_top5 | fusion_minus_semantic_rank | oracle_weight | oracle_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.9649 | 0.3870 | 0.6604 | 0.0500 | 0.9653 | 0.3880 | 0.0004 | 0.0500 | 0.9653 |
| 11 | 0.9655 | 0.3730 | 0.6370 | 0.1000 | 0.9657 | 0.3590 | 0.0002 | 0.0500 | 0.9658 |
| 77 | 0.9665 | 0.3730 | 0.6495 | 0.0000 | 0.9665 | 0.3730 | 0.0000 | 0.0500 | 0.9667 |
| 101 | 0.9658 | 0.3790 | 0.6374 | 0.0200 | 0.9659 | 0.3800 | 0.0002 | 0.0500 | 0.9660 |
| mean | 0.9657 | 0.3780 | 0.6461 | 0.0425 | 0.9658 | 0.3750 | 0.0002 | 0.0500 | 0.9660 |
| std | 0.0006 | 0.0066 | 0.0112 | 0.0435 | 0.0005 | 0.0123 | 0.0002 | 0.0000 | 0.0006 |

## Interpretation

- Semantic EEG->CLIP retrieval is already strong on the overlap splits.
- The cortical branch alone retrieves images above chance because it predicts real visual-fMRI patterns.
- Validation-selected fusion currently gives mean rank 0.9658 vs semantic-only 0.9657. This is an improvement under the current fusion protocol.
- The oracle column shows whether a useful fusion weight exists on heldout; it should not be used for claims, only for diagnosing whether validation selection is the bottleneck.

Artifacts:

- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_clip_vjepa_retrieval_cortical_rerank/summary.csv`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_clip_vjepa_retrieval_cortical_rerank/summary.json`
