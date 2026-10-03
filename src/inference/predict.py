"""Segment one aerial image or a folder of images and save colour masks.

    python -m src.inference.predict --checkpoint models/checkpoints/unet_resnet34_best.pth \
        --input path/to/image.jpg --output predictions/
    python -m src.inference.predict --checkpoint ... --input path/to/folder --output predictions/ --tta

For each image it writes <name>_mask.png (colour-coded classes, original size)
and <name>_overlay.png (mask blended over the photo), and prints the share of
the image covered by each class.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.evaluation.report import CLASS_NAMES
from src.models.model_builder import load_checkpoint

CLASS_COLORS = np.array([
    (0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0), (128, 128, 128),
    (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 255), (0, 128, 0),
], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def predict(model, image_rgb: np.ndarray, device, img_size: int = 256, tta: bool = False) -> np.ndarray:
    """Class index per pixel, at the original image size."""
    h, w = image_rgb.shape[:2]
    x = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
    probs = model(x).softmax(dim=1)
    if tta:
        probs = probs + model(x.flip(-1)).softmax(dim=1).flip(-1) + model(x.flip(-2)).softmax(dim=1).flip(-2)
    mask = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="an image or a folder of images")
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--tta", action="store_true", help="average with flipped predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device)
    paths = sorted(p for p in args.input.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if args.input.is_dir() else [args.input]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint epoch {meta.get('epoch', '?')} on {device}; {len(paths)} image(s)")

    for path in paths:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = predict(model, image, device, args.img_size, args.tta)
        colour = CLASS_COLORS[mask]
        overlay = cv2.addWeighted(image, 0.55, colour, 0.45, 0)
        cv2.imwrite(str(args.output / f"{path.stem}_mask.png"), cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output / f"{path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        shares = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES)) / mask.size
        top = ", ".join(f"{CLASS_NAMES[i]} {shares[i]:.0%}" for i in np.argsort(-shares)[:4] if shares[i] > 0)
        print(f"{path.name}: {top}")
    print(f"Saved masks and overlays to {args.output}")


if __name__ == "__main__":
    main()
