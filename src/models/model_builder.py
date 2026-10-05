"""Build the U-Net used in this project and load saved checkpoints."""
from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch

NUM_CLASSES = 10


def build_unet(encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet",
               num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """U-Net with a pretrained encoder; returns raw logits (no activation)."""
    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                    in_channels=3, classes=num_classes, activation=None)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load a checkpoint saved by src/training/train.py or notebook 03.

    Returns the model in eval mode and the checkpoint's metadata
    (epoch, val_iou, val_loss, config).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_unet(config.get("encoder", "resnet34"), encoder_weights=None,
                       num_classes=config.get("num_classes", NUM_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    return model, meta
