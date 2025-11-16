# src/dataset.py
"""
Image Segmentation Dataset (FloodNet-Compatible)
Professional, clean, and extensible PyTorch Dataset for semantic segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

IMAGE_EXT = ".jpg"
MASK_EXT = ".png"
NUM_CLASSES = 10
IGNORE_INDEX = -1  # For void/unlabeled pixels (if present)


# ============================================================================
# 2. AUGMENTATION PIPELINES
# ============================================================================
def get_training_transforms(
    image_size: int = 512,
    seed: Optional[int] = None,
) -> A.Compose:
    """Strong, reproducible training augmentations."""
    pipeline = A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, p=0.5
            ),
            A.OneOf([
                A.RandomCrop(height=384, width=384, p=0.5),
                A.PadIfNeeded(min_height=image_size, min_width=image_size,
                              border_mode=cv2.BORDER_CONSTANT, p=0.5)
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )
    if seed is not None:
        pipeline = A.ReplayCompose(pipeline.children, seed=seed)  # Full replay
    return pipeline


def get_validation_transforms(image_size: int = 512) -> A.Compose:
    """Deterministic validation preprocessing."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


# ============================================================================
# 3. DATASET CLASS
# ============================================================================
class FloodNetDataset(Dataset):
    def __init__(
        self,
        subset: str = "train",
        image_size: int = 512,
        seed: Optional[int] = None,
    ) -> None:
        if subset not in {"train", "val"}:
            raise ValueError("subset must be 'train' or 'val'")

        self.subset = subset
        self.image_dir = DATA_ROOT / subset / "images"
        self.mask_dir = DATA_ROOT / subset / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths: List[Path] = [
            p for p in self.image_dir.iterdir() if p.suffix.lower() == IMAGE_EXT.lower()
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No {IMAGE_EXT} files in {self.image_dir}")

        self.transform = (
            get_training_transforms(image_size, seed)
            if subset == "train"
            else get_validation_transforms(image_size)
        )

        print(f"[FloodNetDataset] {subset.upper()} | {len(self)} samples | size={image_size}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.mask_dir / f"{img_path.stem}{MASK_EXT}"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {mask_path}")

        # Apply transforms
        augmented = self.transform(image=img, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = mask_tensor.long()

        # Optional: map ignore label
        if IGNORE_INDEX in torch.unique(mask_tensor):
            mask_tensor[mask_tensor == 255] = IGNORE_INDEX  # common void label

        if mask_tensor.max() >= NUM_CLASSES and mask_tensor.max() != IGNORE_INDEX:
            raise ValueError(f"Invalid class {mask_tensor.max().item()}")

        return image_tensor, mask_tensor

    def __repr__(self) -> str:
        return f"FloodNetDataset(subset={self.subset}, samples={len(self)}, size=?)"