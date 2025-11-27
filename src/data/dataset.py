"""
FloodNet Dataset Class for PyTorch
Place this in: src/data/dataset.py
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FloodNetDataset(Dataset):
    """
    FloodNet Dataset for Semantic Segmentation
    
    Args:
        image_dir: Path to images directory
        mask_dir: Path to masks directory
        transform: Albumentations transform pipeline
        img_size: Target image size (height, width)
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.images = sorted([
            f for f in self.image_dir.glob("*") 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        print(f"Loaded {len(self.images)} images from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def _get_mask_path(self, image_path):
        """Find corresponding mask for an image"""
        img_stem = image_path.stem
        
        # Try different naming patterns
        patterns = [
            f"{img_stem}.png",
            f"{img_stem}_lab.png",
            f"{img_stem}_mask.png",
            f"{img_stem}_label.png",
        ]
        
        for pattern in patterns:
            mask_path = self.mask_dir / pattern
            if mask_path.exists():
                return mask_path
        
        # Fallback: search for any matching file
        for mask_file in self.mask_dir.glob(f"*{img_stem}*"):
            return mask_file
        
        return None
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self._get_mask_path(img_path)
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask if not found (shouldn't happen)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default: normalize and convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.long()
        
        return image, mask


def get_train_transform(img_size=(256, 256)):
    """Training augmentation pipeline"""
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=30, 
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.MedianBlur(blur_limit=3, p=1),
        ], p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size=(256, 256)):
    """Validation transform (no augmentation)"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Quick test
if __name__ == "__main__":
    # Test paths - update these
    train_img = r"D:\Projects\Image Segmentation for Disaster Resilience\Disaster-segmentation\data\raw\FloodNet\train\train-org-img"
    train_mask = r"D:\Projects\Image Segmentation for Disaster Resilience\Disaster-segmentation\data\raw\FloodNet\train\train-label-img"
    
    # Create dataset
    dataset = FloodNetDataset(
        train_img, train_mask,
        transform=get_train_transform(),
        img_size=(256, 256)
    )
    
    # Test loading
    image, mask = dataset[0]
    print(f"\n✅ Dataset test passed!")
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Image dtype: {image.dtype}")
    print(f"   Mask dtype: {mask.dtype}")
    print(f"   Mask unique values: {torch.unique(mask).tolist()}")