# Disaster Segmentation for Emergency Response — Technical Report

## Abstract
*Summarize the project in 150-200 words: the problem, approach, key results, and significance.*

This project applies deep learning semantic segmentation to UAV/aerial flood imagery from the FloodNet dataset. A U-Net architecture with a pretrained ResNet34 encoder was trained to classify each pixel into 10 scene classes — including flooded and non-flooded buildings, roads, water, vehicles, and vegetation — enabling rapid, structured damage assessment for emergency responders. The final model achieved **70.70% mean IoU** (excluding background), **89.31% pixel accuracy**, and **82.33% mean Dice coefficient** on the test set of 448 images, demonstrating strong generalization from 1,445 training samples. The combined Cross-Entropy + Dice loss and cosine annealing schedule contributed to stable convergence with early stopping at epoch 24.

---

## 1. Introduction

### 1.1 Problem Statement
After floods and natural disasters, emergency responders need rapid visibility into affected areas. Manual interpretation of aerial imagery is slow, subjective, and does not scale. Automated semantic segmentation can classify every pixel in flood imagery, enabling structured and objective damage assessment.

### 1.2 Motivation
- Time-critical decisions depend on fast, accurate damage maps
- UAV/satellite imagery is increasingly available but requires automated processing
- Pixel-level segmentation provides actionable granularity for resource allocation

### 1.3 Objectives
1. Build an end-to-end semantic segmentation pipeline for flood imagery
2. Achieve competitive IoU and accuracy on the FloodNet benchmark
3. Produce visual diagnostic outputs suitable for emergency response teams

---

## 2. Related Work

### 2.1 Semantic Segmentation Architectures
- **U-Net** (Ronneberger et al., 2015) — encoder-decoder with skip connections for biomedical and satellite imagery
- **DeepLabV3+** (Chen et al., 2018) — atrous spatial pyramid pooling for multi-scale features
- **SegFormer** (Xie et al., 2021) — transformer-based segmentation with hierarchical features

### 2.2 Disaster and Remote Sensing Segmentation
- FloodNet benchmark (Rahnemoonfar et al., 2021)
- xBD dataset for building damage assessment
- SpaceNet for building and road extraction from satellite imagery

---

## 3. Dataset

### 3.1 FloodNet Overview
| Item | Details |
|---|---|
| Source | FloodNet Challenge Dataset |
| Image type | UAV/aerial post-flood imagery |
| Input size used | 256 × 256 |
| Train images | 1,445 |
| Validation images | 450 |
| Test images | 448 |

### 3.2 Class Distribution
| Class | Description |
|---|---|
| 0 | Background |
| 1 | Flooded building |
| 2 | Non-flooded building |
| 3 | Flooded road |
| 4 | Non-flooded road |
| 5 | Water |
| 6 | Tree |
| 7 | Vehicle |
| 8 | Pool |
| 9 | Grass |

### 3.3 Preprocessing
- Resized to 256×256
- Normalized using ImageNet statistics
- Augmentations: horizontal/vertical flips, brightness/contrast jitter, Gaussian noise

---

## 4. Methodology

### 4.1 Model Architecture
U-Net with pretrained ResNet34 encoder (ImageNet weights). The decoder uses transposed convolutions with skip connections from corresponding encoder stages.

| Component | Value |
|---|---|
| Encoder | ResNet34 (pretrained) |
| Decoder | U-Net |
| Trainable parameters | 24,437,674 |
| Output classes | 10 |

### 4.2 Training Configuration
| Hyperparameter | Value |
|---|---|
| Loss function | 50% Cross-Entropy + 50% Dice Loss |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (T_max=50) |
| Batch size | 8 |
| Max epochs | 50 |
| Early stopping | Triggered at epoch 24 |

### 4.3 Loss Function
The combined loss addresses both pixel-level classification (Cross-Entropy) and region-level overlap (Dice), which is particularly important for imbalanced segmentation classes:

```
L_total = 0.5 × L_CE + 0.5 × L_Dice
```

---

## 5. Results

### 5.1 Quantitative Results
| Metric | Validation | Test |
|---|---:|---:|
| Mean IoU (no bg) | 66.71% | **70.70%** |
| Pixel Accuracy | — | **89.31%** |
| Mean Dice | — | **82.33%** |

### 5.2 Per-Class IoU
*Fill in from `results/metrics/` after final evaluation:*

| Class | IoU | F1 / Dice |
|---|---:|---:|
| Flooded building | — | — |
| Non-flooded building | — | — |
| Flooded road | — | — |
| Non-flooded road | — | — |
| Water | — | — |
| Tree | — | — |
| Vehicle | — | — |
| Pool | — | — |
| Grass | — | — |

### 5.3 Visual Results
*Reference the visualizations in `results/visualizations/`:*
- `evaluation/best_predictions.png` — highest IoU test predictions
- `predictions/val_predictions.png` — validation overlay comparisons
- `evaluation/per_class_metrics.png` — per-class IoU bar chart
- `evaluation/confusion_matrix.png` — pixel-level confusion matrix
- `training/training_curves.png` — loss and IoU over epochs

### 5.4 Error Analysis
*Describe which classes performed best/worst and hypothesize why:*
- Vegetation and water classes are typically well-segmented due to distinct spectral signatures
- Vehicles and pools are challenging due to small object sizes
- Flooded vs non-flooded roads have subtle visual differences under varied lighting

---

## 6. Discussion

### 6.1 Key Findings
1. Transfer learning from ImageNet significantly accelerates convergence on small aerial datasets
2. Combined CE + Dice loss outperforms either loss alone for imbalanced segmentation
3. Early stopping prevents overfitting while preserving peak validation performance
4. Augmentation is critical for generalization on varied flood imagery conditions

### 6.2 Limitations
- Fixed 256×256 resolution may lose small object detail
- Model has only been evaluated on FloodNet; cross-dataset generalization is untested
- No temporal or multi-spectral data was used

### 6.3 Comparison with Baselines
*If you trained multiple architectures, compare them here:*

| Model | Test mIoU | Pixel Acc | Parameters |
|---|---:|---:|---:|
| U-Net + ResNet34 | **70.70%** | **89.31%** | 24.4M |
| *U-Net (baseline)* | — | — | — |
| *DeepLabV3+* | — | — | — |

---

## 7. Future Work
1. Compare with DeepLabV3+, SegFormer, and Mask2Former architectures
2. Add class-specific error analysis (confusion between flooded/non-flooded classes)
3. Export to ONNX for faster CPU/edge inference
4. Build a Streamlit demo for interactive flood mask prediction
5. Explore multi-scale input or higher resolution training
6. Test cross-dataset transfer to xBD or other disaster datasets

---

## 8. References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
3. Rahnemoonfar, M., et al. (2021). FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding.
4. Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.

---

**Author:** Jeevan Raj M  
**Affiliation:** B.E. Artificial Intelligence & Data Science, University of Mysore School of Engineering  
**Date:** 2025
