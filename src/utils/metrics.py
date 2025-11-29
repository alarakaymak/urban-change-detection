"""
Evaluation metrics for change detection.

Key Metrics:
- IoU (Intersection over Union): Measures overlap between prediction and ground truth
- F1 Score: Harmonic mean of precision and recall
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
"""

import torch
import numpy as np
from typing import Dict, Optional


class IoU:
    """
    Intersection over Union metric.
    
    IoU = TP / (TP + FP + FN)
    
    Also known as Jaccard Index.
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
        
    def reset(self):
        self.intersection = 0
        self.union = 0
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metric with batch predictions.
        
        Args:
            pred: Predicted logits (B, 1, H, W) or (B, H, W)
            target: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Apply sigmoid if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        self.intersection += intersection.item()
        self.union += union.item()
        
    def compute(self) -> float:
        """Compute final IoU score."""
        return (self.intersection + self.smooth) / (self.union + self.smooth)


class F1Score:
    """
    F1 Score metric.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
        
    def reset(self):
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with batch predictions."""
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred = (pred > self.threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        
    def compute(self) -> Dict[str, float]:
        """Compute precision, recall, and F1."""
        precision = (self.tp + self.smooth) / (self.tp + self.fp + self.smooth)
        recall = (self.tp + self.smooth) / (self.tp + self.fn + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class ChangeDetectionMetrics:
    """
    Comprehensive metrics for change detection evaluation.
    
    Computes all required metrics:
    - IoU
    - F1 Score
    - Precision
    - Recall
    - Overall Accuracy
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with predictions.
        
        Args:
            pred: Model predictions (logits or probabilities)
            target: Ground truth binary masks
        """
        with torch.no_grad():
            # Convert to probabilities if logits
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            # Binarize
            pred = (pred > self.threshold).float()
            target = target.float()
            
            # Flatten
            pred = pred.view(-1)
            target = target.view(-1)
            
            # Compute confusion matrix elements
            self.tp += ((pred == 1) & (target == 1)).sum().item()
            self.tn += ((pred == 0) & (target == 0)).sum().item()
            self.fp += ((pred == 1) & (target == 0)).sum().item()
            self.fn += ((pred == 0) & (target == 1)).sum().item()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        smooth = 1e-6
        
        # Precision: TP / (TP + FP)
        precision = (self.tp + smooth) / (self.tp + self.fp + smooth)
        
        # Recall: TP / (TP + FN)
        recall = (self.tp + smooth) / (self.tp + self.fn + smooth)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + smooth)
        
        # IoU: TP / (TP + FP + FN)
        iou = (self.tp + smooth) / (self.tp + self.fp + self.fn + smooth)
        
        # Overall Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn + smooth) / (total + smooth)
        
        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn
        }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all metrics for a single batch.
    
    Args:
        pred: Predictions (B, 1, H, W)
        target: Ground truth (B, 1, H, W)
        threshold: Binarization threshold
    
    Returns:
        Dictionary of metrics
    """
    metrics = ChangeDetectionMetrics(threshold=threshold)
    metrics.update(pred, target)
    return metrics.compute()


if __name__ == "__main__":
    # Test metrics
    batch_size = 4
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, 1, 256, 256)
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Compute metrics
    metrics = compute_metrics(pred, target)
    
    print("Change Detection Metrics:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")

