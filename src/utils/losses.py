"""
Loss functions for change detection.

Includes:
- Binary Cross-Entropy Loss
- Dice Loss
- Focal Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Good for handling class imbalance (fewer changed pixels).
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute Dice
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -α(1-p)^γ * log(p)
    
    Focuses on hard examples by down-weighting easy ones.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) - logits
            target: Ground truth (B, 1, H, W) - binary
        """
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Get probabilities
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)
        
        # Compute focal weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    
    Common choice for segmentation tasks.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CombinedLoss(nn.Module):
    """
    Flexible combined loss function.
    
    Supports BCE, Dice, and Focal loss combinations.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0
        
        if self.bce_weight > 0:
            total_loss += self.bce_weight * self.bce(pred, target)
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice(pred, target)
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal(pred, target)
        
        return total_loss


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Loss-specific arguments
    
    Returns:
        Loss function module
    """
    losses = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "bce_dice": BCEDiceLoss,
        "combined": CombinedLoss,
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name](**kwargs)


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    pred = torch.randn(batch_size, 1, 256, 256)
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test each loss
    for loss_name in ["bce", "dice", "focal", "bce_dice", "combined"]:
        loss_fn = get_loss(loss_name)
        loss_value = loss_fn(pred, target)
        print(f"{loss_name}: {loss_value.item():.4f}")

