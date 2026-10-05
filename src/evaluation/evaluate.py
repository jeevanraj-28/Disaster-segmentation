"""Evaluate a checkpoint on the FloodNet test set (script version of notebook 04).

    # Full evaluation (needs the dataset and a checkpoint)
    python -m src.evaluation.evaluate --checkpoint models/checkpoints/unet_resnet34_best.pth

    # Re-print metrics from a saved confusion matrix (no PyTorch, no data needed)
    python -m src.evaluation.evaluate --from-confusion results/evaluation/confusion_matrix.npy

Writes a Markdown report (overall + per-class metrics + most confused pairs)
and the confusion matrix it was computed from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.report import markdown_report

ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 10


def confusion_from_checkpoint(checkpoint: Path, data_dir: Path, batch_size: int, img_size: int,
                              tta: bool) -> tuple[np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.data.dataset import FloodNetDataset, get_val_transform
    from src.models.model_builder import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(checkpoint, device)
    size = (img_size, img_size)
    dataset = FloodNetDataset(data_dir / "test" / "test-org-img", data_dir / "test" / "test-label-img",
                              transform=get_val_transform(size), img_size=size)
    if len(dataset) == 0:
        raise SystemExit(f"No test images under {data_dir}. See README: 'Get the data'.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="test"):
            images = images.to(device)
            logits = model(images).softmax(dim=1)
            if tta:  # average with horizontally and vertically flipped predictions
                logits = logits + model(images.flip(-1)).softmax(dim=1).flip(-1)
                logits = logits + model(images.flip(-2)).softmax(dim=1).flip(-2)
            preds = logits.argmax(dim=1).cpu().numpy().ravel()
            target = masks.numpy().ravel()
            valid = (target >= 0) & (target < NUM_CLASSES)
            cm += np.bincount(NUM_CLASSES * target[valid] + preds[valid],
                              minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return cm, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--from-confusion", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "FloodNet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "evaluation" / "test_report.md")
    args = parser.parse_args()

    if args.from_confusion:
        cm = np.load(args.from_confusion)
        title = f"Test set evaluation (from {args.from_confusion.name})"
    else:
        cm, meta = confusion_from_checkpoint(args.checkpoint, args.data_dir, args.batch_size, args.img_size, args.tta)
        title = (f"Test set evaluation: {args.checkpoint.name}, epoch {meta.get('epoch', '?')}"
                 + (" with flip TTA" if args.tta else ""))
        np.save(args.out.with_suffix(".confusion.npy"), cm)
    report = markdown_report(cm, title=title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
