# src/train.py
"""
Main training script for the Disaster Segmentation project.
Orchestrates:
    • U-Net + ResNet-50 (smp)
    • Mixed precision + gradient accumulation
    • Weighted CrossEntropyLoss
    • CosineAnnealingLR
    • CSV logging + best/latest checkpoints
    • Early stopping + confusion-matrix metrics
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Local imports
from .config import cfg
from .dataset import FloodNetDataset
from .model import get_unet
from .utils import seed_everything, get_device, compute_class_weights
from .metrics import update_confusion_matrix, compute_metrics_from_confusion_matrix
from .earlystop import EarlyStopping
from .logger import TrainingLogger


# =============================================================================
# Initialization
# =============================================================================
DEVICE = get_device()
seed_everything(cfg.SEED)

print(f"[train] Effective batch size: {cfg.BATCH_SIZE} × {cfg.ACCUM_STEPS} = {cfg.BATCH_SIZE * cfg.ACCUM_STEPS}")


# =============================================================================
# Loss Function
# =============================================================================
def get_loss_criterion(class_weights: torch.Tensor) -> nn.Module:
    return nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        ignore_index=cfg.IGNORE_INDEX,
    )


# =============================================================================
# Datasets & Weights
# =============================================================================
train_dataset = FloodNetDataset(subset="train", image_size=cfg.IMAGE_SIZE, seed=cfg.SEED)
val_dataset = FloodNetDataset(subset="val", image_size=cfg.IMAGE_SIZE)

class_weights = compute_class_weights(train_dataset, num_classes=cfg.NUM_CLASSES)
criterion = get_loss_criterion(class_weights)


# =============================================================================
# Model, Optimizer, Scheduler
# =============================================================================
model = get_unet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
scaler = GradScaler()

scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LEARNING_RATE * 0.01)
earlystop = EarlyStopping(patience=cfg.SCHEDULER_PATIENCE, mode="max", verbose=True)
logger = TrainingLogger(
    log_dir=cfg.LOG_DIR,
    checkpoint_dir=cfg.CHECKPOINT_DIR,
    primary_metric="miou_micro",
    mode="max"
)


# =============================================================================
# DataLoaders
# =============================================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


# =============================================================================
# Training & Validation Loops
# =============================================================================
def train_one_epoch(epoch: int) -> Tuple[float, np.ndarray]:
    model.train()
    running_loss = 0.0
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for step, (imgs, masks) in enumerate(loop):
        imgs, masks = imgs.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, masks) / cfg.ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % cfg.ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * cfg.ACCUM_STEPS
        loop.set_postfix(loss=f"{running_loss/(step+1):.4f}")

        with torch.no_grad():
            preds = outputs.argmax(1)
            update_confusion_matrix(cm, preds, masks, cfg.NUM_CLASSES)

    return running_loss / len(train_loader), cm


def validate_one_epoch() -> Tuple[float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)

    loop = tqdm(val_loader, desc="Validate", leave=False)
    with torch.no_grad():
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            running_loss += loss.item()
            preds = outputs.argmax(1)
            update_confusion_matrix(cm, preds, masks, cfg.NUM_CLASSES)
            loop.set_postfix(loss=f"{running_loss/(len(loop)+1):.4f}")

    return running_loss / len(val_loader), cm


# =============================================================================
# Main Training Loop
# =============================================================================
def main() -> None:
    start_time = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()

        train_loss, _ = train_one_epoch(epoch)
        val_loss, val_cm = validate_one_epoch()

        val_metrics = compute_metrics_from_confusion_matrix(val_cm, ignore_background=True)
        val_miou = val_metrics["miou_micro"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            cm=val_cm,
            lr=current_lr,
            model=model,
            optimizer=optimizer,
        )

        if earlystop.step(val_miou):
            print(f"[train] Early stopping triggered at epoch {epoch}")
            break

        epoch_time = (time.time() - epoch_start) / 60
        print(f"Epoch {epoch} | {epoch_time:.1f}m | Val mIoU: {val_miou:.4f}")

    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.1f} minutes")
    print(f"Best mIoU: {logger.best_value:.4f} @ epoch {logger.best_epoch}")
    print(f"Logs: {cfg.LOG_DIR}/training_log.csv")
    print(f"Best model: {cfg.CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()