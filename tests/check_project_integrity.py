# check_project_integrity.py
# Run this from your project root to verify everything is complete and healthy!

import os
from pathlib import Path
import json
import torch

BASE_DIR = Path(r"D:\Projects\Image_Segmentation_for_Disaster_Resilience\Disaster-segmentation")

print("PROJECT INTEGRITY & COMPLETENESS CHECK")
print("=" * 80)

issues = []
warnings = []
ok_count = 0

def ok(msg):
    global ok_count
    ok_count += 1
    print(f"OK   {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"WARNING  {msg}")

def error(msg):
    issues.append(msg)
    print(f"ERROR  {msg}")

# 1. Check for empty directories
print("\n1. Checking for empty directories...")
empty_dirs = []
for dirpath, dirnames, filenames in os.walk(BASE_DIR):
    path = Path(dirpath)
    if path == BASE_DIR / ".git": 
        continue
    if not filenames and not any(d.name.startswith('.') for d in path.iterdir() if d.is_dir()):
        if len(list(path.iterdir())) == 0:
            empty_dirs.append(path.relative_to(BASE_DIR))
            error(f"Empty directory: {path.relative_to(BASE_DIR)}")

if not empty_dirs:
    ok("No empty directories found")

# 2. Expected structure & critical files
critical_files = {
    "models/checkpoints/unet_resnet34_best.pth": "Best trained model",
    "results/evaluation/test_evaluation_results.json": "Evaluation metrics",
    "logs/training_history.json": "Training logs",
    "results/visualizations/final/comparison_grid.png": "Comparison visualization",
    "results/visualizations/final/overlay_visualization.png": "Overlay visualization",
    "results/visualizations/final/performance_summary.png": "Performance chart",
    "results/visualizations/final/class_legend.png": "Class legend",
    "results/reports/Final_Project_Report.md": "Final report (Markdown)",
    "results/reports/Final_Project_Report.txt": "Final report (Text)",
    "README.md": "GitHub README",
    "requirements.txt": "Python dependencies",
    "notebooks/05_visualization.ipynb": "Visualization notebook",
    "notebooks/06_final_report.ipynb": "Final report notebook"
}

print("\n2. Checking critical files...")
for rel_path, description in critical_files.items():
    full_path = BASE_DIR / rel_path
    if full_path.exists():
        if full_path.stat().st_size > 100:  # Not completely empty
            ok(f"{description}")
        else:
            warn(f"{description} exists but is nearly empty (<100 bytes)")
    else:
        error(f"MISSING: {description}")

# 3. Quick content validation
print("\n3. Validating file contents...")

# Check if JSON files are valid
json_files = [
    BASE_DIR / "results/evaluation/test_evaluation_results.json",
    BASE_DIR / "logs/training_history.json"
]
for jf in json_files:
    if jf.exists():
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                ok(f"Valid JSON: {jf.name}")
            else:
                warn(f"JSON is empty or invalid: {jf.name}")
        except Exception as e:
            error(f"Corrupted JSON: {jf.name} → {e}")
    else:
        error(f"Missing JSON: {jf.name}")

# Check model can be loaded
model_path = BASE_DIR / "models/checkpoints/unet_resnet34_best.pth"
if model_path.exists():
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet34", classes=10)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            ok("Model checkpoint is valid and loadable")
        else:
            error("Checkpoint missing 'model_state_dict'")
    except Exception as e:
        error(f"Model corrupted or incompatible: {e}")
else:
    error("Model file missing")

# Final Report
print("\n" + "="*80)
print("PROJECT INTEGRITY REPORT")
print("="*80)

print(f"Total checks passed: {ok_count}")
if issues:
    print(f"\nCRITICAL ISSUES ({len(issues)}):")
    for i, msg in enumerate(issues, 1):
        print(f"   {i}. {msg}")
else:
    print("\nNO CRITICAL ISSUES FOUND!")

if warnings:
    print(f"\nWarnings ({len(warnings)}):")
    for w in warnings:
        print(f"   • {w}")

if not issues and len(warnings) <= 1:  # Only Final_Report.md missing is expected before running 06
    print("\nPROJECT IS 100% COMPLETE & PROFESSIONAL!")
    print("You can now run Notebook 06 → Final Report with full confidence.")
    print("After that, your project will be truly flawless.")
else:
    print("\nFix the issues above, then re-run this script.")

print(f"\nCheck completed: {__file__}")