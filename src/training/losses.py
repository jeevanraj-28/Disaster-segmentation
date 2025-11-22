"""
Loss Functions for Semantic Segmentation
Place this in: src/training/losses.py

Includes:
- Focal Loss (for class imbalance)
- Dice Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (tensor or None)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Class index to ignore
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - model predictions (logits)
            targets: (N, H, W) - ground truth labels
        """
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        ignore_index: Class index to ignore
    """
    
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets, num_classes=10):
        """
        Args:
            inputs: (N, C, H, W) - model predictions (logits)
            targets: (N, H, W) - ground truth labels
        """
        # Softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Calculate Dice per class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs + targets_one_hot).sum(dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Ignore specified class
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, device=dice_score.device)
            mask[self.ignore_index] = 0
            dice_score = dice_score * mask
            return 1 - dice_score.sum() / mask.sum()
        
        return 1 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    
    Total Loss = focal_weight * FocalLoss + dice_weight * DiceLoss
    
    Args:
        focal_weight: Weight for focal loss
        dice_weight: Weight for dice loss
        alpha: Class weights for focal loss
        gamma: Focusing parameter for focal loss
        num_classes: Number of segmentation classes
    """
    
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=None, 
                 gamma=2.0, num_classes=10):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets, self.num_classes)
        
        return self.focal_weight * focal + self.dice_weight * dice


def get_class_weights(class_distribution, method='balanced', device='cuda'):
    """
    Calculate class weights from distribution
    
    Args:
        class_distribution: Dict or list of pixel counts per class
        method: 'balanced' or 'effective'
        device: torch device
    
    Returns:
        Tensor of class weights
    """
    if isinstance(class_distribution, dict):
        counts = [class_distribution.get(i, 1) for i in range(len(class_distribution))]
    else:
        counts = class_distribution
    
    counts = torch.tensor(counts, dtype=torch.float32)
    total = counts.sum()
    num_classes = len(counts)
    
    if method == 'balanced':
        weights = total / (num_classes * counts)
    elif method == 'effective':
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        weights = torch.ones(num_classes)
    
    # Normalize
    weights = weights / weights.max()
    
    return weights.to(device)


# Quick test
if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size, num_classes, h, w = 2, 10, 256, 256
    predictions = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(0, num_classes, (batch_size, h, w))
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    fl = focal_loss(predictions, targets)
    print(f"✅ Focal Loss: {fl.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dl = dice_loss(predictions, targets)
    print(f"✅ Dice Loss: {dl.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(focal_weight=1.0, dice_weight=1.0)
    cl = combined_loss(predictions, targets)
    print(f"✅ Combined Loss: {cl.item():.4f}")
    
    # Test class weights
    class_dist = {0: 1000, 1: 100, 2: 500, 3: 200, 4: 300, 
                  5: 800, 6: 1200, 7: 50, 8: 60, 9: 2000}
    weights = get_class_weights(class_dist, method='balanced', device='cpu')
    print(f"✅ Class Weights: {weights.tolist()}")