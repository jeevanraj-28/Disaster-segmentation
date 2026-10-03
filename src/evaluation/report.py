"""Segmentation metrics from a confusion matrix (NumPy only, no PyTorch).

All metrics are computed ONCE over the whole dataset: pixels from every image
are accumulated into one confusion matrix (rows = ground truth, columns =
prediction) before IoU is calculated. This is the standard way to report
FloodNet/Cityscapes-style mIoU.

Note: the IoU printed during training (notebook 03) is different. It averages,
over batches, the mean IoU of only the classes present in each batch. The two
numbers are not directly comparable.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = [
    "Background", "Building-flooded", "Building-non-flooded", "Road-flooded", "Road-non-flooded",
    "Water", "Tree", "Vehicle", "Pool", "Grass",
]


def per_class_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(axis=1)       # pixels that truly belong to each class
    pred = cm.sum(axis=0)     # pixels predicted as each class
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(gt + pred - tp > 0, tp / (gt + pred - tp), 0.0)
        dice = np.where(gt + pred > 0, 2 * tp / (gt + pred), 0.0)
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(gt > 0, tp / gt, 0.0)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "pixel_share": gt / gt.sum()}


def summary(cm: np.ndarray) -> dict[str, float]:
    m = per_class_metrics(cm)
    cm = np.asarray(cm, dtype=np.float64)
    return {
        "pixel_accuracy": float(np.trace(cm) / cm.sum()),
        "mean_iou_all": float(m["iou"].mean()),
        "mean_iou_no_bg": float(m["iou"][1:].mean()),
        "mean_dice_all": float(m["dice"].mean()),
        "mean_dice_no_bg": float(m["dice"][1:].mean()),
    }


def top_confusions(cm: np.ndarray, k: int = 5, names=CLASS_NAMES) -> list[tuple[str, str, int]]:
    off = np.asarray(cm).copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:k]
    return [(names[t], names[p], int(off[t, p])) for t, p in order]


def markdown_report(cm: np.ndarray, title: str = "Test set evaluation", names=CLASS_NAMES) -> str:
    s, m = summary(cm), per_class_metrics(cm)
    lines = [
        f"# {title}", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Mean IoU, excluding background | {s['mean_iou_no_bg']:.2%} |",
        f"| Mean IoU, all 10 classes | {s['mean_iou_all']:.2%} |",
        f"| Mean Dice, excluding background | {s['mean_dice_no_bg']:.2%} |",
        f"| Pixel accuracy | {s['pixel_accuracy']:.2%} |",
        "",
        "| Class | Share of pixels | IoU | Dice | Precision | Recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i in np.argsort(-m["iou"]):
        lines.append(f"| {names[i]} | {m['pixel_share'][i]:.2%} | {m['iou'][i]:.3f} | {m['dice'][i]:.3f} | "
                     f"{m['precision'][i]:.3f} | {m['recall'][i]:.3f} |")
    lines += ["", "| Most confused (true → predicted) | Pixels |", "| --- | --- |"]
    lines += [f"| {t} → {p} | {n:,} |" for t, p, n in top_confusions(cm, names=names)]
    return "\n".join(lines) + "\n"
