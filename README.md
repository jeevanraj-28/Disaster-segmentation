# Disaster Segmentation

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Semantic%20Segmentation-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-FloodNet-2563EB?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/jeevanraj-28/Disaster-segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/github/last-commit/jeevanraj-28/Disaster-segmentation?style=for-the-badge" />
</p>

Deep learning semantic segmentation system for identifying disaster-affected regions from satellite and aerial imagery. The project uses a **U-Net decoder with a ResNet34 encoder** to support rapid damage assessment after floods and natural disasters.

## Problem Statement

During floods and large-scale disasters, response teams need fast, visual understanding of affected regions. Manual inspection of aerial imagery is slow and difficult to scale.

This project uses semantic segmentation to classify each pixel in disaster imagery, helping identify regions such as flooded areas, damaged infrastructure, roads, buildings, and background terrain.

## Real-World Motivation

Accurate segmentation of disaster-affected zones can help:

- prioritize rescue and relief operations
- identify inaccessible roads and damaged infrastructure
- estimate affected residential and urban regions
- support emergency response dashboards with visual AI outputs

## Dataset: FloodNet

FloodNet is a disaster-scene understanding dataset built from UAV imagery captured after flood events.

Dataset details to document after final preprocessing:

| Item | Details |
|---|---|
| Dataset | FloodNet |
| Task | Semantic segmentation |
| Image type | UAV/aerial disaster imagery |
| Input format | RGB images |
| Label format | Pixel-level segmentation masks |
| Classes | Background, building-flooded, building-non-flooded, road-flooded, road-non-flooded, water, tree, vehicle, pool, grass |
| Image count | Add exact count from local FloodNet split |
| Train/Val/Test split | Add final split ratio from preprocessing notebook |

## Model Architecture

The segmentation pipeline is implemented in PyTorch using a U-Net style architecture with a pretrained ResNet34 encoder.

```text
Input RGB Image
      |
      v
Image Preprocessing
Resize, normalize, augment
      |
      v
PyTorch Dataset + DataLoader
      |
      v
U-Net + ResNet34 Segmentation Model
Encoder extracts spatial features
Decoder upsamples with skip connections
      |
      v
Per-pixel Class Prediction
      |
      v
Segmentation Mask Overlay
```

Current baseline:

- **Encoder:** ResNet34 pretrained on ImageNet
- **Decoder:** U-Net with skip connections
- **Parameters:** approximately 24M
- **Output:** 256x256 multi-class segmentation mask

Recommended future model comparisons:

- **U-Net:** strong and simple segmentation baseline
- **DeepLabV3+:** better multi-scale context for aerial imagery
- **SegFormer:** transformer-based model for improved boundary understanding

## Training Details

Update this section after training:

| Setting | Value |
|---|---|
| Model | U-Net with ResNet34 encoder |
| Framework | PyTorch |
| Epochs | TBD |
| Batch size | TBD |
| Optimizer | Adam / AdamW |
| Learning rate | TBD |
| Loss function | CrossEntropy + Dice loss |
| Scheduler | Cosine annealing learning rate |
| Augmentations | Random flips, brightness/contrast changes, Gaussian noise |
| Hardware | GPU recommended |

## Results

Current reported results:

| Metric | Score |
|---|---:|
| Mean IoU without background | 70.70% |
| Pixel Accuracy | 89.31% |
| Mean Dice Coefficient | 82.33% |
| Validation IoU | 66.71% |
| Inference Speed | approximately 50 FPS on GPU |

## Visual Outputs

Recommended visual comparison format:

```text
Original Image | Ground Truth Mask | Predicted Mask | Overlay
```

Add sample images here:

```text
assets/results/sample_01.png
assets/results/sample_02.png
assets/results/sample_03.png
```

## Run Locally

```bash
git clone https://github.com/jeevanraj-28/Disaster-segmentation.git
cd Disaster-segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py --config configs/unet_floodnet.yaml
python evaluate.py --checkpoint checkpoints/best_model.pth
python predict.py --image samples/test_image.png --checkpoint checkpoints/best_model.pth
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

## Future Work

- Train and compare U-Net, DeepLabV3+, and SegFormer.
- Add class imbalance handling with Dice/Focal loss.
- Improve small-object segmentation for vehicles and narrow roads.
- Add a Streamlit or Gradio demo for uploading disaster images.
- Deploy inference as a FastAPI endpoint.
- Add Grad-CAM or attention visualization for interpretability.

## Author

**Jeevan Raj M**  
AI/ML Engineer | Computer Vision | Disaster AI  

[LinkedIn](https://linkedin.com/in/jeevan-raj-m-5ba64a383) | [GitHub](https://github.com/jeevanraj-28) | [Email](mailto:jeevanrajm2882004@gmail.com)
