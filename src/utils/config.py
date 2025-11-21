"""
Configuration file for Disaster Image Segmentation Project
Place this in: src/utils/config.py
"""

import os
from pathlib import Path

class Config:
    """Central configuration for the project"""
    
    # ==================== PATHS ====================
    # Base directory (adjust if needed)
    BASE_DIR = Path(r"D:\Projects\Image Segmentation for Disaster Resilience\Disaster-segmentation")
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA = DATA_DIR / "raw" / "FloodNet"
    PROCESSED_DATA = DATA_DIR / "processed"
    SAMPLE_DATA = DATA_DIR / "sample_data"
    
    # Raw data subdirectories
    TRAIN_IMAGES = RAW_DATA / "train" / "train-org-img"
    TRAIN_MASKS = RAW_DATA / "train" / "train-label-img"
    VAL_IMAGES = RAW_DATA / "val" / "val-org-img"
    VAL_MASKS = RAW_DATA / "val" / "val-label-img"
    TEST_IMAGES = RAW_DATA / "test" / "test-org-img"
    TEST_MASKS = RAW_DATA / "test" / "test-label-img"
    
    # Output paths
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    LOGS_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    
    # ==================== IMAGE PARAMETERS ====================
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    
    # ==================== MODEL PARAMETERS ====================
    NUM_CLASSES = 10
    ENCODER_NAME = "resnet34"  # For pretrained encoder
    ENCODER_WEIGHTS = "imagenet"
    
    # ==================== TRAINING PARAMETERS ====================
    BATCH_SIZE = 8  # Reduce if GPU memory issues
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # ==================== DATA SPLIT ====================
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # ==================== CLASS INFORMATION ====================
    # FloodNet has 10 classes
    CLASSES = {
        0: "Background",
        1: "Building-flooded",
        2: "Building-non-flooded", 
        3: "Road-flooded",
        4: "Road-non-flooded",
        5: "Water",
        6: "Tree",
        7: "Vehicle",
        8: "Pool",
        9: "Grass"
    }
    
    NUM_CLASSES = len(CLASSES)
    CLASS_NAMES = list(CLASSES.values())
    
    # RGB colors for visualization
    CLASS_COLORS = {
        0: (0, 0, 0),         # Background - Black
        1: (255, 0, 0),       # Building-flooded - Red
        2: (0, 0, 255),       # Building-non-flooded - Blue
        3: (255, 165, 0),     # Road-flooded - Orange
        4: (128, 128, 128),   # Road-non-flooded - Gray
        5: (0, 255, 255),     # Water - Cyan
        6: (0, 255, 0),       # Tree - Green
        7: (255, 0, 255),     # Vehicle - Magenta
        8: (255, 255, 255),   # Pool - White
        9: (0, 128, 0)        # Grass - Dark Green
    }
    
    # ==================== DEVICE ====================
    @staticmethod
    def get_device():
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    # ==================== CREATE DIRECTORIES ====================
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs_to_create = [
            cls.PROCESSED_DATA / "train" / "images",
            cls.PROCESSED_DATA / "train" / "masks",
            cls.PROCESSED_DATA / "val" / "images",
            cls.PROCESSED_DATA / "val" / "masks",
            cls.SAMPLE_DATA / "images",
            cls.SAMPLE_DATA / "masks",
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR / "tensorboard",
            cls.RESULTS_DIR / "predictions",
            cls.RESULTS_DIR / "metrics",
            cls.RESULTS_DIR / "visualizations" / "data_exploration",
            cls.RESULTS_DIR / "visualizations" / "training",
            cls.RESULTS_DIR / "visualizations" / "predictions",
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print("All directories created successfully!")
    
    # ==================== VALIDATION ====================
    @classmethod
    def validate_paths(cls):
        """Check if data paths exist"""
        paths_to_check = [
            ("Train Images", cls.TRAIN_IMAGES),
            ("Train Masks", cls.TRAIN_MASKS),
            ("Val Images", cls.VAL_IMAGES),
            ("Val Masks", cls.VAL_MASKS),
            ("Test Images", cls.TEST_IMAGES),
        ]
        
        all_valid = True
        print("\n Checking data paths:")
        print("-" * 50)
        
        for name, path in paths_to_check:
            exists = path.exists()
            status = "✅" if exists else "❌"
            count = len(list(path.glob("*"))) if exists else 0
            print(f"{status} {name}: {path}")
            if exists:
                print(f"   └── Found {count} files")
            all_valid = all_valid and exists
        
        print("-" * 50)
        if all_valid:
            print("All paths validated successfully!")
        else:
            print(" Some paths are missing. Check your data directory.")
        
        return all_valid


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    
    print(f"\n Base Directory: {Config.BASE_DIR}")
    print(f" Image Size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print(f" Number of Classes: {Config.NUM_CLASSES}")
    print(f" Batch Size: {Config.BATCH_SIZE}")
    print(f" Epochs: {Config.EPOCHS}")
    print(f" Learning Rate: {Config.LEARNING_RATE}")
    
    # Validate paths
    Config.validate_paths()
    
    # Create directories
    Config.create_directories()
    
    # Check device
    device = Config.get_device()
    print(f"\n Device: {device}")