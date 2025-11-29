"""
Visualization utilities for change detection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


def denormalize(
    img: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize an image tensor.
    
    Args:
        img: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    img = img.cpu().numpy()
    
    # Denormalize
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    # CHW -> HWC
    img = img.transpose(1, 2, 0)
    
    return img


def visualize_predictions(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize change detection predictions.
    
    Args:
        img_a: Before image (C, H, W)
        img_b: After image (C, H, W)
        pred: Predicted mask (1, H, W) - logits or probabilities
        target: Ground truth mask (1, H, W)
        threshold: Threshold for binarizing predictions
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Denormalize images
    img_a_np = denormalize(img_a)
    img_b_np = denormalize(img_b)
    
    # Convert predictions
    if pred.min() < 0 or pred.max() > 1:
        pred_prob = torch.sigmoid(pred)
    else:
        pred_prob = pred
    
    pred_binary = (pred_prob > threshold).float()
    
    # Convert to numpy
    pred_prob_np = pred_prob.squeeze().cpu().numpy()
    pred_binary_np = pred_binary.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Plot
    axes[0, 0].imshow(img_a_np)
    axes[0, 0].set_title("Before (Image A)", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(img_b_np)
    axes[0, 1].set_title("After (Image B)", fontsize=12)
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(target_np, cmap="Reds")
    axes[0, 2].set_title("Ground Truth", fontsize=12)
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(pred_prob_np, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Prediction (Probability)", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(pred_binary_np, cmap="Reds")
    axes[1, 1].set_title(f"Prediction (Binary, τ={threshold})", fontsize=12)
    axes[1, 1].axis("off")
    
    # Overlay: Green = TP, Red = FP, Blue = FN
    overlay = np.zeros((*target_np.shape, 3))
    tp = (pred_binary_np == 1) & (target_np == 1)
    fp = (pred_binary_np == 1) & (target_np == 0)
    fn = (pred_binary_np == 0) & (target_np == 1)
    
    overlay[tp] = [0, 1, 0]  # Green - True Positive
    overlay[fp] = [1, 0, 0]  # Red - False Positive
    overlay[fn] = [0, 0, 1]  # Blue - False Negative
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Comparison (G=TP, R=FP, B=FN)", fontsize=12)
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_batch(
    batch_img_a: torch.Tensor,
    batch_img_b: torch.Tensor,
    batch_pred: torch.Tensor,
    batch_target: torch.Tensor,
    num_samples: int = 4,
    save_dir: Optional[str] = None
):
    """
    Visualize multiple samples from a batch.
    
    Args:
        batch_img_a: Batch of before images (B, C, H, W)
        batch_img_b: Batch of after images (B, C, H, W)
        batch_pred: Batch of predictions (B, 1, H, W)
        batch_target: Batch of targets (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures
    """
    batch_size = min(batch_img_a.shape[0], num_samples)
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(batch_size):
        save_path = f"{save_dir}/sample_{i}.png" if save_dir else None
        visualize_predictions(
            batch_img_a[i],
            batch_img_b[i],
            batch_pred[i],
            batch_target[i],
            save_path=save_path,
            show=False
        )


def plot_metrics(
    train_metrics: dict,
    val_metrics: dict,
    save_path: Optional[str] = None
):
    """
    Plot training and validation metrics.
    
    Args:
        train_metrics: Dict of training metrics (each value is a list)
        val_metrics: Dict of validation metrics
        save_path: Path to save figure
    """
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, train_metrics.keys()):
        epochs = range(1, len(train_metrics[metric_name]) + 1)
        
        ax.plot(epochs, train_metrics[metric_name], "b-", label="Train")
        ax.plot(epochs, val_metrics[metric_name], "r-", label="Val")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
    plt.close()


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot loss and metric curves from training history.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_iou', 'val_iou', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU
    if 'train_iou' in history:
        axes[1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
        axes[1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('Training and Validation IoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Test visualization
    batch_size = 4
    
    # Create dummy data
    img_a = torch.randn(3, 256, 256)
    img_b = torch.randn(3, 256, 256)
    pred = torch.randn(1, 256, 256)
    target = torch.randint(0, 2, (1, 256, 256)).float()
    
    # Test single visualization
    visualize_predictions(img_a, img_b, pred, target, show=False)
    print("Visualization test passed!")

