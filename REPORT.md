# Flood Damage Segmentation on FloodNet: Technical Report

**Jeevan Raj M** · B.E. Artificial Intelligence & Data Science, University of Mysore School of Engineering · 2025

## Abstract

After a flood, responders need to know quickly which roads and buildings are under water. This project trains a U-Net with an ImageNet-pretrained ResNet34 encoder to label every pixel of FloodNet drone images with one of 10 classes. Training used 1,445 images at 256 × 256 with a 50/50 cross-entropy and Dice loss, AdamW and cosine annealing, with early stopping on validation IoU. On the 448-image held-out test set the model reaches 70.7% mean IoU over the 9 non-background classes (66.9% over all 10) and 89.3% pixel accuracy. Large classes such as grass, trees and dry roads score above 0.80 IoU. Vehicles (0.51) and pools (0.60), each under 0.2% of pixels, are the main weakness, consistent with the original FloodNet paper. A shorter training run with less patience scored 67.6%, and the report explains why training-time validation IoU and test mIoU are not directly comparable.

## 1. Problem

Manual review of aerial imagery after a disaster is slow and does not scale. Semantic segmentation gives a per-pixel map that separates, for example, a flooded road from a dry one, which is what routing and rescue decisions depend on.

## 2. Data

| Item | Value |
| --- | --- |
| Dataset | FloodNet, Track 1 (supervised) |
| Split | 1,445 train / 450 validation / 448 test (official) |
| Input size | 256 × 256 (resized from full resolution) |
| Classes | Background, building flooded, building non-flooded, road flooded, road non-flooded, water, tree, vehicle, pool, grass |

The classes are very imbalanced. In the test set grass covers 55.7% of pixels, trees 17.9% and water 10.7%, while vehicles cover 0.16% and pools 0.19%.

## 3. Method

| Component | Choice |
| --- | --- |
| Model | U-Net decoder (upsampling with skip connections) on a ResNet34 encoder pretrained on ImageNet; 24.4 M parameters; `segmentation-models-pytorch` |
| Loss | 0.5 × weighted cross-entropy + 0.5 × Dice. Class weights: square root of inverse pixel frequency, scaled so the largest weight is 1, minimum 0.1 |
| Optimisation | AdamW (lr 3e-4, weight decay 1e-4), gradient clipping at 1.0, cosine annealing to 1e-6, batch size 8 |
| Early stopping | Patience 12 on validation IoU; best checkpoint kept |
| Augmentation | Horizontal/vertical flips, 90° rotations, shift-scale-rotate, brightness/contrast/hue/gamma, Gaussian noise or blur |
| Hardware | Single laptop GPU (RTX 4050) |

**Metrics.** Test metrics are computed from one confusion matrix over all test pixels: IoU = TP / (TP + FP + FN) per class, averaged over classes (mIoU). The validation IoU printed during training is a different quantity: the mean over batches of the IoU of the classes present in each batch. It is used only to choose the checkpoint.

## 4. Results

**Overall (test, 448 images)**

| Metric | Value |
| --- | --- |
| mIoU, 9 classes (no background) | 70.7% |
| mIoU, 10 classes | 66.9% |
| Mean Dice, 9 classes | 82.3% |
| Pixel accuracy | 89.3% |

**Per class (test)**

| Class | IoU | Dice |
| --- | --- | --- |
| Grass | 0.869 | 0.930 |
| Tree | 0.811 | 0.895 |
| Road, not flooded | 0.804 | 0.891 |
| Building, not flooded | 0.751 | 0.858 |
| Water | 0.726 | 0.841 |
| Building, flooded | 0.686 | 0.814 |
| Road, flooded | 0.608 | 0.756 |
| Pool | 0.601 | 0.751 |
| Vehicle | 0.508 | 0.674 |
| Background | 0.323 | 0.488 |

**Reference point.** Rahnemoonfar et al. (2021) report 79.7% mIoU for PSPNet, 61.5% for DeepLabV3+ and 42.6% for ENet over the same 9 classes, trained at 713 × 713. The split and resolution differ from this project, so this is context, not a controlled comparison.

**Training runs**

| Run | Epoch budget / patience | Stopped | Best val IoU (training monitor) | Test mIoU (9 classes) |
| --- | --- | --- | --- | --- |
| A (reported) | 60 / 12 | epoch 50, best epoch 38 | 0.667 | 0.707 |
| B | 50 / 5 | epoch 24 | 0.622 | 0.676 |

Run A's validation IoU stopped improving after epoch 38 while training IoU kept rising (about 0.69 to 0.72), the usual sign of overfitting that early stopping guards against. Run B stopped much earlier and scored lower, which suggests patience 5 was too short; the runs differ in more than one setting, so this is not a controlled ablation.

## 5. Error analysis

1. **Small objects.** Vehicle and pool are the two lowest non-background classes. At 256 × 256 a vehicle covers only a few pixels; the FloodNet paper observes the same difficulty for every model.
2. **Flooded vs non-flooded.** Flooded roads (precision 0.65, recall 0.91) and flooded buildings (precision 0.75, recall 0.88) are over-predicted: the model tends to call structures near water "flooded". The difference between the two states is context (surrounding water), not appearance.
3. **Similar natural textures.** The largest confusions are grass → tree, tree → grass, water → grass and grass → water, mostly along blurry boundaries between regions.
4. **Background** (IoU 0.32) is a mixed residual class with no consistent appearance.

## 6. Limitations

- Downsampling to 256 × 256 removes the detail that small classes need.
- One architecture and one seed: no measure of run-to-run variance and no comparison with other architectures under identical conditions.
- No ablation of the Dice term, the class weights or the augmentations, so their individual effect is unknown.

## 7. Next steps

1. Train on 512 × 512 tiles from full-resolution images, targeting vehicles and pools.
2. Compare DeepLabV3+ and SegFormer on the same split and metrics.
3. Run 3 seeds per configuration and report mean ± standard deviation.
4. Ablate the loss: cross-entropy only, Dice only, with and without class weights.

## References

1. Rahnemoonfar, M., et al. (2021). FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding. *IEEE Access*. [arXiv:2012.02951](https://arxiv.org/abs/2012.02951)
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
