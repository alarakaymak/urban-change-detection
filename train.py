"""
Training script for Urban Change Detection.

Usage:
    python train.py --model siamese_unet --epochs 100 --batch_size 8
    python train.py --config config/config.yaml
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.data.dataloader import get_dataloaders
from src.models.factory import get_model, list_models
from src.utils.losses import get_loss
from src.utils.metrics import ChangeDetectionMetrics
from src.utils.visualization import visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train change detection model")
    
    # Model
    parser.add_argument("--model", type=str, default="siamese_unet",
                        choices=list_models(),
                        help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder backbone")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/LEVIR-CD",
                        help="Path to LEVIR-CD dataset")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    
    # Loss
    parser.add_argument("--loss", type=str, default="bce_dice",
                        help="Loss function")
    
    # Misc
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    writer: SummaryWriter
):
    """Train for one epoch."""
    model.train()
    metrics = ChangeDetectionMetrics()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (img_a, img_b, target) in enumerate(pbar):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        target = target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(img_a, img_b)
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        metrics.update(pred.detach(), target)
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    epoch_metrics = metrics.compute()
    
    # Log to tensorboard
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/IoU", epoch_metrics["iou"], epoch)
    writer.add_scalar("Train/F1", epoch_metrics["f1"], epoch)
    
    return avg_loss, epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device,
    epoch: int,
    writer: SummaryWriter
):
    """Validate the model."""
    model.eval()
    metrics = ChangeDetectionMetrics()
    total_loss = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for img_a, img_b, target in pbar:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        target = target.to(device)
        
        # Forward pass
        pred = model(img_a, img_b)
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Update metrics
        total_loss += loss.item()
        metrics.update(pred, target)
    
    # Compute epoch metrics
    avg_loss = total_loss / len(val_loader)
    epoch_metrics = metrics.compute()
    
    # Log to tensorboard
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/IoU", epoch_metrics["iou"], epoch)
    writer.add_scalar("Val/F1", epoch_metrics["f1"], epoch)
    
    return avg_loss, epoch_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    save_path: str
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }, save_path)


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values
        for key, value in config.get("training", {}).items():
            setattr(args, key, value)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    save_dir = Path(args.save_dir) / run_name
    log_dir = Path(args.log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create dataloaders
    print("Loading data...")
    loaders = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(
        args.model,
        encoder_name=args.encoder if "unet" in args.model else None
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = get_loss(args.loss)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_iou = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_iou": [], "val_iou": [],
        "train_f1": [], "val_f1": []
    }
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints will be saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print("-" * 50)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, epoch, writer
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        
        # Print progress
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                save_dir / "best_model.pth"
            )
            print(f"  ** New best IoU: {best_iou:.4f} - Model saved!")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                save_dir / f"checkpoint_epoch_{epoch}.pth"
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, val_metrics,
        save_dir / "final_model.pth"
    )
    
    # Save training history
    with open(save_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {save_dir}")
    
    writer.close()


if __name__ == "__main__":
    main()

