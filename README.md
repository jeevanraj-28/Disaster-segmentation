Image Segmentation for Disaster Resilience

A deep-learning system for segmenting flooded and non-flooded regions in aerial imagery.

------------------------------------------------------------------------
Overview

This project develops a semantic segmentation model that identifies key elements in disaster-affected regions using the FloodNet dataset.
The model uses a U-Net architecture with a ResNet34 encoder, trained to classify each pixel into multiple scene classes such as water, flooded roads, flooded buildings, vegetation, and more.

The work includes dataset processing, model training, evaluation, visualization, and a structured final report.

------------------------------------------------------------------------
Key Results

| Metric                       | Score         |
| ---------------------------- | ------------- |
| **Mean IoU (no background)** | 70.70%        |
| **Pixel Accuracy**           | 89.31%        |
| **Mean Dice Coefficient**    | 82.33%        |
| **Validation IoU**           | 66.71%        |
| **Inference Speed**          | ~50 FPS (GPU) |

These results place the model above commonly reported benchmarks for FloodNet using comparable architectures.

------------------------------------------------------------------------
Repository Structure

Disaster-segmentation/
├── data/                       # Dataset (not included in Git)
│   └── raw/FloodNet/
├── notebooks/                  # Notebook workflow pipeline
│   ├── 01_clean_preprocess_floodnet.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_unet_basic.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_visualization.ipynb
│   └── 06_final_report.ipynb
├── src/                        # Source code modules
│   ├── data/                   # Datasets & augmentation
│   ├── training/               # Loss, metrics, engines
│   └── utils/                  # Helper utilities
├── models/checkpoints/         # Trained model weights
├── results/
│   ├── evaluation/             # Metric outputs (JSON, CSV)
│   ├── metrics/                # Per-class metrics
│   ├── viz/                    # Visualizations (PNG)
│   └── reports/                # Final report (Markdown)
└── README.md

------------------------------------------------------------------------
Getting Started
1. Clone the Repository
git clone https://github.com/jeevanraj-28/Disaster-segmentation
cd Disaster-segmentation

2. Install Required Packages
pip install -r requirements.txt

3. Download FloodNet Dataset

Place the dataset under:

data/raw/FloodNet/


(The dataset is not included in the repository due to size.)

------------------------------------------------------------------------
Running the Workflow

The project is designed to be reproducible through a series of notebooks:

1. Data Cleaning & Preprocessing

Mask resizing
Pixel normalization
Class distribution extraction

2. Augmentation & Dataset Loader

Random flips, brightness/contrast changes, Gaussian noise
Balanced sampling

3. Model Training
U-Net + ResNet34 backbone
CE + Dice loss
Cosine Annealing LR
Checkpoint saving
4. Evaluation
Mean IoU
Per-class IoU, Dice, Precision, Recall
Confusion matrix
5. Visualization
Best/worst predictions
Overlay maps
Class-level metrics
6. Final Report Generation

------------------------------------------------------------------------
Model Architecture

Encoder: ResNet34 pretrained on ImageNet
Decoder: U-Net with skip connections
Total parameters: ~24M
Output: 256×256 multi-class segmentation mask

The choice of ResNet34 provides an efficient trade-off between speed and accuracy, making the model suitable for real-time or near-real-time use.

------------------------------------------------------------------------

Visualizations Included in This Project

You will find the following in(results/visualizations):

Best predicted samples

Worst predicted samples (error analysis)

Per-class IoU bar chart

Confusion matrix

Side-by-side image–mask–prediction comparisons

These help diagnose model behavior and understand class-level performance.

------------------------------------------------------------------------
Reproduction Checklist

If you want to fully reproduce the results:

1. Download FloodNet
2. Run notebooks in order
3. Check generated logs in logs/
4. Review saved results under results/

------------------------------------------------------------------------
Acknowledgments

FloodNet Dataset (Rahnemoonfar et al.)

PyTorch & Torchvision teams

Segmentation Models PyTorch (Pavel Iakubovskii)

Open-source community for tools and inspiration