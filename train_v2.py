"""
Training Script V2 - Improved Version for Better Results

Improvements over V1:
1. More epochs (50)
2. Learning rate warmup
3. Unfrozen backbone (fine-tune everything)
4. Cosine annealing with warm restarts
5. Gradient accumulation for larger effective batch
6. Mixed precision training (faster on MPS)
7. Better loss weighting
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.data.dataloader import get_dataloaders
from src.models.factory import get_model
from src.utils.losses import get_loss, DiceLoss, FocalLoss
from src.utils.metrics import ChangeDetectionMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train V2 - Improved")
    
    # Model
    parser.add_argument("--model", type=str, default="changeformer_lora")
    parser.add_argument("--encoder", type=str, default="resnet34")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/LEVIR-CD")
    parser.add_argument("--image_size", type=int, default=256)
    
    # Training - IMPROVED DEFAULTS
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)  # Higher LR
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--accumulation_steps", type=int, default=2)  # Effective batch = 8
    
    # Loss - IMPROVED
    parser.add_argument("--dice_weight", type=float, default=0.7)  # More dice
    parser.add_argument("--bce_weight", type=float, default=0.3)
    parser.add_argument("--focal_weight", type=float, default=0.0)
    
    # Fine-tuning strategy
    parser.add_argument("--unfreeze_backbone", action="store_true", default=True,
                        help="Fine-tune backbone (not just LoRA)")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="LR multiplier for backbone")
    
    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--log_dir", type=str, default="logs_v2")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImprovedLoss(nn.Module):
    """Combined loss with Dice + BCE + optional Focal."""
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, focal_weight=0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        if focal_weight > 0:
            self.focal = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, pred, target):
        loss = 0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(pred, target)
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(pred, target)
        
        return loss


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, 
    writer, accumulation_steps=1
):
    model.train()
    metrics = ChangeDetectionMetrics()
    total_loss = 0
    
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (img_a, img_b, target) in enumerate(pbar):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        target = target.to(device)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Forward
        pred = model(img_a, img_b)
        loss = criterion(pred, target)
        
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        metrics.update(pred.detach(), target)
        pbar.set_postfix({"loss": loss.item() * accumulation_steps})
    
    avg_loss = total_loss / len(train_loader)
    epoch_metrics = metrics.compute()
    
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/IoU", epoch_metrics["iou"], epoch)
    writer.add_scalar("Train/F1", epoch_metrics["f1"], epoch)
    
    return avg_loss, epoch_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    metrics = ChangeDetectionMetrics()
    total_loss = 0
    
    for img_a, img_b, target in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        target = target.to(device)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        pred = model(img_a, img_b)
        loss = criterion(pred, target)
        
        total_loss += loss.item()
        metrics.update(pred, target)
    
    avg_loss = total_loss / len(val_loader)
    epoch_metrics = metrics.compute()
    
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/IoU", epoch_metrics["iou"], epoch)
    writer.add_scalar("Val/F1", epoch_metrics["f1"], epoch)
    
    return avg_loss, epoch_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics
    }, save_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_v2_{timestamp}"
    save_dir = Path(args.save_dir) / run_name
    log_dir = Path(args.log_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # Data
    print("Loading data...")
    loaders = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Model - WITH UNFROZEN BACKBONE
    print(f"Creating model: {args.model}")
    if "changeformer_lora" in args.model:
        model = get_model(
            args.model,
            pretrained=True,
            use_lora=True,
            freeze_backbone=not args.unfreeze_backbone  # UNFREEZE!
        )
    elif "changeformer_swin" in args.model:
        model = get_model(
            args.model,
            pretrained=True,
            freeze_backbone=not args.unfreeze_backbone
        )
    elif "unet" in args.model or "siamese" in args.model:
        model = get_model(args.model, encoder_name=args.encoder)
    else:
        model = get_model(args.model)
    
    model = model.to(device)
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    
    # Loss
    criterion = ImprovedLoss(
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        focal_weight=args.focal_weight
    )
    
    # Optimizer with layer-wise LR
    if hasattr(model, 'backbone') and args.backbone_lr_scale < 1.0:
        backbone_params = list(model.backbone.parameters())
        other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        
        param_groups = [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
            {"params": other_params, "lr": args.lr}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"Using layer-wise LR: backbone={args.lr * args.backbone_lr_scale}, other={args.lr}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler: Warmup + Cosine Annealing
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    # Training loop
    best_iou = 0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    
    print(f"\n{'='*60}")
    print(f"TRAINING V2 - IMPROVED")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulation_steps})")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Backbone unfrozen: {args.unfreeze_backbone}")
    print(f"Loss weights: Dice={args.dice_weight}, BCE={args.bce_weight}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, 
            epoch, writer, args.accumulation_steps
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, loaders["val"], criterion, device, epoch, writer
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", current_lr, epoch)
        
        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        
        print(f"\nEpoch {epoch}/{args.epochs} (LR: {current_lr:.2e})")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_dir / "best_model.pth"
            )
            print(f"  ** New best IoU: {best_iou:.4f} - Model saved!")
        
        # Periodic save
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                save_dir / f"checkpoint_epoch_{epoch}.pth"
            )
    
    # Final save
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, val_metrics,
        save_dir / "final_model.pth"
    )
    
    with open(save_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}")
    
    writer.close()


if __name__ == "__main__":
    main()


