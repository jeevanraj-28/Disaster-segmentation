"""
Evaluation Metrics for Semantic Segmentation
Place this in: src/training/metrics.py

Includes:
- IoU (Intersection over Union) / Jaccard Index
- Dice Coefficient
- Pixel Accuracy
"""

import torch
import numpy as np


class SegmentationMetrics:
    """
    Compute segmentation metrics for multi-class segmentation
    
    Metrics computed:
    - Per-class IoU
    - Mean IoU (mIoU)
    - Per-class Dice
    - Mean Dice
    - Pixel Accuracy
    """
    
    def __init__(self, num_classes, class_names=None, ignore_index=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        """
        Update confusion matrix with batch predictions
        
        Args:
            pred: (N, H, W) - predicted class indices
            target: (N, H, W) - ground truth class indices
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        pred = pred.flatten()
        target = target.flatten()
        
        # Mask for valid pixels
        mask = (target >= 0) & (target < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (target != self.ignore_index)
        
        # Update confusion matrix
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        indices = self.num_classes * valid_target + valid_pred
        cm_update = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += cm_update.reshape(self.num_classes, self.num_classes)
    
    def compute_iou(self):
        """Compute IoU for each class"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        
        # Avoid division by zero
        iou = np.where(union > 0, intersection / union, 0)
        return iou
    
    def compute_dice(self):
        """Compute Dice coefficient for each class"""
        intersection = np.diag(self.confusion_matrix)
        sum_pred = self.confusion_matrix.sum(axis=0)
        sum_target = self.confusion_matrix.sum(axis=1)
        
        denominator = sum_pred + sum_target
        dice = np.where(denominator > 0, 2 * intersection / denominator, 0)
        return dice
    
    def compute_pixel_accuracy(self):
        """Compute overall pixel accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0
    
    def compute_class_accuracy(self):
        """Compute accuracy for each class"""
        class_correct = np.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(axis=1)
        return np.where(class_total > 0, class_correct / class_total, 0)
    
    def get_results(self):
        """Get all metrics as a dictionary"""
        iou = self.compute_iou()
        dice = self.compute_dice()
        pixel_acc = self.compute_pixel_accuracy()
        class_acc = self.compute_class_accuracy()
        
        # Calculate mean metrics (excluding classes with no samples)
        valid_mask = self.confusion_matrix.sum(axis=1) > 0
        mean_iou = iou[valid_mask].mean() if valid_mask.any() else 0
        mean_dice = dice[valid_mask].mean() if valid_mask.any() else 0
        
        results = {
            'pixel_accuracy': pixel_acc,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'per_class_iou': {self.class_names[i]: iou[i] for i in range(self.num_classes)},
            'per_class_dice': {self.class_names[i]: dice[i] for i in range(self.num_classes)},
            'per_class_accuracy': {self.class_names[i]: class_acc[i] for i in range(self.num_classes)},
        }
        
        return results
    
    def print_results(self):
        """Print formatted results"""
        results = self.get_results()
        
        print("\n" + "=" * 60)
        print("SEGMENTATION METRICS")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"   Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"   Mean IoU (mIoU): {results['mean_iou']:.4f}")
        print(f"   Mean Dice: {results['mean_dice']:.4f}")
        
        print(f"\nPer-Class IoU:")
        for name, iou in results['per_class_iou'].items():
            bar = "█" * int(iou * 20) + "░" * (20 - int(iou * 20))
            print(f"   {name:25s} {bar} {iou:.4f}")
        
        return results


def calculate_batch_metrics(pred, target, num_classes=10):
    """
    Quick metric calculation for a single batch
    
    Args:
        pred: (N, C, H, W) logits or (N, H, W) class indices
        target: (N, H, W) ground truth
    
    Returns:
        dict with metrics
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # Pixel accuracy
    correct = (pred == target).sum()
    total = len(pred)
    pixel_acc = correct / total
    
    # Mean IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    mean_iou = np.mean(ious) if ious else 0
    
    return {
        'pixel_accuracy': pixel_acc,
        'mean_iou': mean_iou
    }


# Quick test
if __name__ == "__main__":
    print("Testing metrics...")
    
    # FloodNet class names
    class_names = [
        "Background", "Building-flooded", "Building-non-flooded",
        "Road-flooded", "Road-non-flooded", "Water",
        "Tree", "Vehicle", "Pool", "Grass"
    ]
    
    # Create metric tracker
    metrics = SegmentationMetrics(num_classes=10, class_names=class_names)
    
    # Simulate predictions
    for _ in range(5):
        pred = torch.randint(0, 10, (4, 256, 256))
        target = torch.randint(0, 10, (4, 256, 256))
        metrics.update(pred, target)
    
    # Print results
    results = metrics.print_results()
    print("\n✅ Metrics test passed!")