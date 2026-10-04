"""Train U-Net (ResNet34 encoder) on FloodNet. Script version of notebook 03.

    python -m src.training.train --config configs/unet_resnet34.yaml   # settings of the reported run
    python -m src.training.train --epochs 1 --limit 32                 # 2-minute smoke test

Saves the best checkpoint (by validation IoU) to models/checkpoints/ and the
per-epoch history to logs/training_history.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import FloodNetDataset, get_train_transform, get_val_transform
from src.models.model_builder import build_unet
from src.training.losses import CEDiceLoss
from src.utils.config import Config

NUM_CLASSES = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> float:
    """Mean IoU over the non-background classes present in this batch.

    This is the training-time monitor used in notebook 03. It is averaged over
    batches, so it differs from the dataset-level mIoU in src/evaluation.
    """
    ious = []
    for cls in range(1, num_classes):
        p, t = pred == cls, target == cls
        union = (p | t).sum().item()
        if union:
            ious.append((p & t).sum().item() / union)
    return float(np.mean(ious)) if ious else 0.0


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_iou = 0.0
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, desc="train" if training else "val", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            total_iou += batch_iou(outputs.argmax(dim=1), masks)
    return total_loss / len(loader), total_iou / len(loader)


def load_class_weights(path: Path, device) -> torch.Tensor | None:
    if not path.exists():
        print(f"No class weights at {path}; training without them.")
        return None
    weights = json.loads(path.read_text())["balanced_weights"]
    return torch.tensor([weights[str(i)] for i in range(NUM_CLASSES)], dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Config.RAW_DATA)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--patience", type=int, default=Config.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Config.CHECKPOINTS_DIR / "unet_resnet34_best.pth")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file with default values for the options above; flags still override it")
    known, _ = parser.parse_known_args()
    if known.config:
        import yaml
        values = yaml.safe_load(known.config.read_text()) or {}
        unknown = set(values) - {a.dest for a in parser._actions}
        if unknown:
            raise SystemExit(f"Unknown keys in {known.config}: {sorted(unknown)}")
        for key in ("data_dir", "out"):
            if key in values:
                values[key] = Path(values[key])
        parser.set_defaults(**values)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (args.img_size, args.img_size)
    train_ds = FloodNetDataset(args.data_dir / "train" / "train-org-img", args.data_dir / "train" / "train-label-img",
                               transform=get_train_transform(size), img_size=size)
    val_ds = FloodNetDataset(args.data_dir / "val" / "val-org-img", args.data_dir / "val" / "val-label-img",
                             transform=get_val_transform(size), img_size=size)
    if len(train_ds) == 0:
        raise SystemExit(f"No training images under {args.data_dir}. See README: 'Get the data'.")
    if args.limit:
        train_ds, val_ds = Subset(train_ds, range(min(args.limit, len(train_ds)))), \
            Subset(val_ds, range(min(args.limit, len(val_ds))))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_unet().to(device)
    class_weights = None if args.no_class_weights else load_class_weights(
        Config.RESULTS_DIR / "metrics" / "class_weights.json", device)
    criterion = CEDiceLoss(0.5, 0.5, class_weights=class_weights, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": [], "lr": []}
    best_iou, waited, start = 0.0, 0, time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | train {len(train_ds)} | val {len(val_ds)} | epochs {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        for key, value in zip(history, (train_loss, train_iou, val_loss, val_iou, scheduler.get_last_lr()[0])):
            history[key].append(value)
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} IoU {train_iou:.4f} | "
              f"val loss {val_loss:.4f} IoU {val_iou:.4f}")
        if val_iou > best_iou:
            best_iou, waited = val_iou, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_iou": val_iou, "val_loss": val_loss,
                        "config": {"encoder": "resnet34", "num_classes": NUM_CLASSES, "img_size": size}},
                       args.out)
            print(f"           saved best checkpoint (val IoU {val_iou:.4f})")
        else:
            waited += 1
            if waited >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    history_path = Config.LOGS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Done in {(time.time() - start) / 60:.1f} min. Best val IoU {best_iou:.4f}. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
