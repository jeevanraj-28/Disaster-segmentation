"""
Diagnostic Script for Visualization Notebook Preparation
=========================================================
This script checks all necessary files, data, and configurations
needed for creating the visualization notebook.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path

print("="*70)
print("🔍 DISASTER SEGMENTATION PROJECT - DIAGNOSTIC CHECK")
print("="*70)

# Base paths
base_path = Path(r"D:\Projects\Image_Segmentation_for_Disaster_Resilience\Disaster-segmentation")

# ============================================================
# 1. CHECK MODEL CHECKPOINT
# ============================================================
print("\n" + "="*70)
print("1️⃣  CHECKING MODEL CHECKPOINT")
print("="*70)

model_path = base_path / "models" / "checkpoints" / "unet_resnet34_best.pth"
if model_path.exists():
    print(f"✅ Model found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"   📦 Checkpoint keys: {list(checkpoint.keys())}")
        if 'epoch' in checkpoint:
            print(f"   📊 Trained for {checkpoint['epoch']} epochs")
        if 'best_miou' in checkpoint:
            print(f"   🎯 Best mIoU: {checkpoint['best_miou']:.4f}")
        if 'model_state_dict' in checkpoint:
            print(f"   ✅ Model state dict found")
    except Exception as e:
        print(f"   ⚠️  Error loading checkpoint: {e}")
else:
    print(f"❌ Model NOT found at: {model_path}")

# ============================================================
# 2. CHECK SAVED EVALUATION RESULTS
# ============================================================
print("\n" + "="*70)
print("2️⃣  CHECKING EVALUATION RESULTS")
print("="*70)

results_json = base_path / "results" / "evaluation" / "test_evaluation_results.json"
if results_json.exists():
    print(f"✅ Results JSON found: {results_json}")
    try:
        with open(results_json, 'r') as f:
            results = json.load(f)
        print(f"   📊 Available metrics: {list(results.keys())}")
        if 'mean_iou' in results:
            print(f"   🎯 Mean IoU: {results['mean_iou']:.4f}")
        if 'pixel_accuracy' in results:
            print(f"   🎯 Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        if 'per_class_iou' in results:
            print(f"   📈 Per-class IoU available:")
            if isinstance(results['per_class_iou'], dict):
                for class_name, iou in results['per_class_iou'].items():
                    print(f"      - {class_name}: {iou:.4f}")
            else:
                print(f"      - {len(results['per_class_iou'])} classes")
    except Exception as e:
        print(f"   ⚠️  Error loading results: {e}")
        results = None
else:
    print(f"❌ Results JSON NOT found at: {results_json}")
    results = None

confusion_matrix_path = base_path / "results" / "evaluation" / "confusion_matrix.npy"
if confusion_matrix_path.exists():
    print(f"\n✅ Confusion matrix found: {confusion_matrix_path}")
    try:
        cm = np.load(confusion_matrix_path)
        print(f"   📊 Shape: {cm.shape} (classes x classes)")
        print(f"   📊 Total predictions: {cm.sum():.0f}")
    except Exception as e:
        print(f"   ⚠️  Error loading confusion matrix: {e}")
else:
    print(f"\n❌ Confusion matrix NOT found at: {confusion_matrix_path}")

sample_ious_path = base_path / "results" / "metrics" / "test_sample_ious.csv"
if sample_ious_path.exists():
    print(f"\n✅ Sample IoUs CSV found: {sample_ious_path}")
    try:
        import pandas as pd
        df = pd.read_csv(sample_ious_path)
        print(f"   📊 Number of samples: {len(df)}")
        print(f"   📊 Columns: {list(df.columns)}")
        if 'mean_iou' in df.columns:
            print(f"   📊 Mean IoU range: {df['mean_iou'].min():.4f} - {df['mean_iou'].max():.4f}")
    except Exception as e:
        print(f"   ⚠️  Error loading CSV: {e}")
else:
    print(f"\n❌ Sample IoUs CSV NOT found at: {sample_ious_path}")

# ============================================================
# 3. CHECK TEST DATASET
# ============================================================
print("\n" + "="*70)
print("3️⃣  CHECKING TEST DATASET")
print("="*70)

test_path = base_path / "data" / "raw" / "FloodNet" / "test"
if test_path.exists():
    print(f"✅ Test directory found: {test_path}")

    # List all subdirectories
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    print(f"   📁 Subdirectories: {[d.name for d in subdirs]}")

    # Check for images
    image_dirs = ['images', 'image', 'Image', 'test-image']
    image_path = None
    for img_dir in image_dirs:
        potential_path = test_path / img_dir
        if potential_path.exists():
            image_path = potential_path
            break

    if image_path is None:
        image_path = test_path

    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    print(f"\n   🖼️  Image path: {image_path}")
    print(f"   🖼️  Test images found: {len(image_files)}")
    if image_files:
        print(f"   📝 First 3 examples:")
        for img in image_files[:3]:
            print(f"      - {img.name}")

    # Check for masks
    mask_dirs = ['masks', 'mask', 'Mask', 'test-label']
    mask_path = None
    for msk_dir in mask_dirs:
        potential_path = test_path / msk_dir
        if potential_path.exists():
            mask_path = potential_path
            break

    if mask_path is None:
        mask_path = test_path

    mask_files = list(mask_path.glob("*.png")) + list(mask_path.glob("*.jpg"))
    print(f"\n   🎭 Mask path: {mask_path}")
    print(f"   🎭 Test masks found: {len(mask_files)}")
    if mask_files:
        print(f"   📝 First 3 examples:")
        for msk in mask_files[:3]:
            print(f"      - {msk.name}")
else:
    print(f"❌ Test directory NOT found at: {test_path}")

# ============================================================
# 4. CHECK GPU AVAILABILITY
# ============================================================
print("\n" + "="*70)
print("4️⃣  CHECKING GPU")
print("="*70)

if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   🔢 CUDA Version: {torch.version.cuda}")
else:
    print(f"⚠️  CUDA not available, will use CPU")

# ============================================================
# 5. CHECK FOR SAVED PREDICTIONS
# ============================================================
print("\n" + "="*70)
print("5️⃣  CHECKING FOR SAVED PREDICTIONS")
print("="*70)

prediction_paths = [
    base_path / "results" / "predictions",
    base_path / "predictions",
    base_path / "outputs" / "predictions",
]

predictions_found = False
for pred_path in prediction_paths:
    if pred_path.exists():
        pred_files = list(pred_path.glob("*.npy")) + list(pred_path.glob("*.pt")) + list(pred_path.glob("*.png"))
        if pred_files:
            print(f"✅ Predictions found at: {pred_path}")
            print(f"   📊 Number of files: {len(pred_files)}")
            predictions_found = True
            break

if not predictions_found:
    print(f"⚠️  No saved predictions found.")
    print(f"   💡 We'll need to generate predictions from the model during visualization")

# ============================================================
# 6. CHECK PROJECT STRUCTURE
# ============================================================
print("\n" + "="*70)
print("6️⃣  CHECKING PROJECT STRUCTURE")
print("="*70)

important_dirs = {
    "Source code": base_path / "src",
    "Notebooks": base_path / "notebooks",
    "Models": base_path / "models",
    "Data": base_path / "data",
    "Results": base_path / "results",
}

for name, path in important_dirs.items():
    if path.exists():
        print(f"✅ {name}: {path}")
    else:
        print(f"❌ {name}: NOT FOUND at {path}")

# Check for visualization output directory
vis_output = base_path / "results" / "visualizations"
if not vis_output.exists():
    print(f"\n💡 Creating visualization output directory: {vis_output}")
    vis_output.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created: {vis_output}")
else:
    print(f"\n✅ Visualization output directory exists: {vis_output}")

# ============================================================
# 7. SUMMARY AND RECOMMENDATIONS
# ============================================================
print("\n" + "="*70)
print("📋 SUMMARY & RECOMMENDATIONS")
print("="*70)

checks = {
    "✅ Model checkpoint": model_path.exists(),
    "✅ Evaluation results JSON": results_json.exists(),
    "✅ Confusion matrix": confusion_matrix_path.exists(),
    "✅ Sample IoUs CSV": sample_ious_path.exists(),
    "✅ Test dataset": test_path.exists(),
    "✅ GPU available": torch.cuda.is_available(),
}

print("\n📊 STATUS:")
for item, status in checks.items():
    symbol = "✅" if status else "❌"
    print(f"   {symbol} {item.split(maxsplit=1)[1]}")

ready_count = sum(checks.values())
total_count = len(checks)
print(f"\n🎯 Ready: {ready_count}/{total_count} components")

print("\n" + "="*70)
print("🚀 NEXT STEPS")
print("="*70)

if ready_count >= 4:
    print("✅ Sufficient resources available for visualization!")
    print("\n📝 We can create:")
    print("   1. ✅ Comparison grids (Image | GT | Prediction)")
    print("   2. ✅ Per-class IoU bar charts")
    print("   3. ✅ Confusion matrix heatmap")
    print("   4. ✅ Overlay visualizations")
    print("   5. ✅ Error analysis plots")
    print("\n💡 Next: Share this output to generate 05_visualization.ipynb")
else:
    print("⚠️  Some critical resources missing.")
    print("   Review the output above and ensure:")
    print("   - Model checkpoint exists")
    print("   - Evaluation results are saved")
    print("   - Test dataset is accessible")

print("\n" + "="*70)
print("✨ DIAGNOSTIC CHECK COMPLETE")
print("="*70)
