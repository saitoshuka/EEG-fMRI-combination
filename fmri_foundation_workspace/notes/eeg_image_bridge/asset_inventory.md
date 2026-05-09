# EEG Image Bridge Asset Inventory

Asset root: `/mnt/c/Users/xinji/Desktop/Image Reconstruction`

## Subjects

sub-01, sub-02, sub-03, sub-04, sub-05, sub-06, sub-07, sub-08, sub-09, sub-10

## Core Files

- `repo`: 0.0 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/EEG_Image_decode`
- `image_metadata`: 0.6 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/images_set/image_metadata.npy`
- `clip_train`: 71.1 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/ViT-H-14_features_train.pt`
- `clip_test`: 1.6 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/ViT-H-14_features_test.pt`
- `train_image_latent_512`: 1035.3 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/train_image_latent_512.pt`
- `test_image_latent_512`: 12.5 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/test_image_latent_512.pt`
- `generated_imgs_tar`: 7750.2 MB at `/mnt/c/Users/xinji/Desktop/Image Reconstruction/generated_imgs.tar.gz`

## Key Shapes

- `image_metadata`: `{"test_img_concepts": 200, "test_img_concepts_THINGS": 200, "test_img_files": 200, "train_img_files": 16540, "train_img_concepts": 16540, "train_img_concepts_THINGS": 16540}`
- `clip_train`: `{"text_features": {"shape": [1654, 1024], "dtype": "torch.float32"}, "img_features": {"shape": [16540, 1024], "dtype": "torch.float32"}}`
- `clip_test`: `{"text_features": {"shape": [200, 1024], "dtype": "torch.float32"}, "img_features": {"shape": [200, 1024], "dtype": "torch.float32"}}`
- `eeg_train_embedding`: `{"shape": [66160, 1024], "dtype": "torch.float32"}`
- `eeg_test_embedding`: `{"shape": [200, 1024], "dtype": "torch.float32"}`

## Interpretation

- The local assets already contain ATM EEG embeddings for all detected subjects, so the first bridge experiment can start from embeddings rather than raw-waveform retraining.
- Train split mapping is image-level: 16,540 images with 4 EEG repeats per image in the ATM embedding files.
- Test split mapping is one ATM embedding per held-out image in the provided embedding files; raw test EEG keeps 80 repeats per image.
- TRIBE outputs should be attached at the image/stimulus level first, then expanded to EEG repeats when training an EEG-to-brain mapper.
