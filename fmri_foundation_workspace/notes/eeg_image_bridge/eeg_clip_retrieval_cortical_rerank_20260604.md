# EEG CLIP Retrieval with Cortical Reranking

## Protocol

- Same image-heldout THINGS-fMRI overlap splits as the real-fMRI validation.
- Semantic baseline: ridge from image-averaged posterior EEG to CLIP ViT-H/14 image features.
- Cortical branch: trainable factorized-query EEG -> real-fMRI visual ROI prediction.
- Fusion: row-zscored semantic similarity plus cortical similarity; fusion weight selected on validation split only.

## Results

| seed | semantic_rank | semantic_top5 | cortical_rank | fusion_weight | fusion_rank | fusion_top5 | fusion_minus_semantic_rank | oracle_weight | oracle_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.9152 | 0.1820 | 0.6604 | 0.1500 | 0.9180 | 0.1910 | 0.0028 | 0.1500 | 0.9180 |
| 11 | 0.9137 | 0.1860 | 0.6370 | 0.1500 | 0.9158 | 0.1760 | 0.0021 | 0.1000 | 0.9159 |
| 77 | 0.9137 | 0.1870 | 0.6495 | 0.1000 | 0.9156 | 0.1930 | 0.0019 | 0.1000 | 0.9156 |
| 101 | 0.9092 | 0.1840 | 0.6374 | 0.1000 | 0.9116 | 0.1950 | 0.0024 | 0.1000 | 0.9116 |
| mean | 0.9129 | 0.1847 | 0.6461 | 0.1250 | 0.9152 | 0.1888 | 0.0023 | 0.1125 | 0.9153 |
| std | 0.0026 | 0.0022 | 0.0112 | 0.0289 | 0.0027 | 0.0087 | 0.0004 | 0.0250 | 0.0027 |

## Interpretation

- Semantic EEG->CLIP retrieval is already strong on the overlap splits.
- The cortical branch alone retrieves images above chance because it predicts real visual-fMRI patterns.
- Validation-selected fusion currently gives mean rank 0.9152 vs semantic-only 0.9129. This is an improvement under the current fusion protocol.
- The oracle column shows whether a useful fusion weight exists on heldout; it should not be used for claims, only for diagnosing whether validation selection is the bottleneck.

Artifacts:

- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_clip_retrieval_cortical_rerank/summary.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/eeg_clip_retrieval_cortical_rerank/summary.json`
