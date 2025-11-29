"""
Evaluation script for trained change detection models.

Usage:
    python evaluate.py --checkpoint checkpoints/siamese_unet/best_model.pth --data_dir data/LEVIR-CD
"""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.dataloader import get_dataloaders
from src.models.factory import get_model, list_models
from src.utils.metrics import ChangeDetectionMetrics
from src.utils.visualization import visualize_predictions, visualize_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate change detection model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="siamese_unet",
                        choices=list_models(),
                        help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Encoder backbone")
    parser.add_argument("--data_dir", type=str, default="data/LEVIR-CD",
                        help="Path to LEVIR-CD dataset")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization of predictions")
    parser.add_argument("--num_vis", type=int, default=20,
                        help="Number of samples to visualize")
    
    return parser.parse_args()


def get_device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader,
    device,
    output_dir: Path = None,
    visualize: bool = False,
    num_vis: int = 20
):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to use
        output_dir: Directory for saving results
        visualize: Whether to save visualizations
        num_vis: Number of samples to visualize
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = ChangeDetectionMetrics()
    
    vis_count = 0
    vis_dir = output_dir / "visualizations" if output_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluating...")
    for batch_idx, (img_a, img_b, target) in enumerate(tqdm(test_loader)):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        target = target.to(device)
        
        # Forward pass
        pred = model(img_a, img_b)
        
        # Update metrics
        metrics.update(pred, target)
        
        # Save visualizations
        if visualize and vis_count < num_vis:
            for i in range(min(img_a.shape[0], num_vis - vis_count)):
                visualize_predictions(
                    img_a[i].cpu(),
                    img_b[i].cpu(),
                    pred[i].cpu(),
                    target[i].cpu(),
                    save_path=vis_dir / f"sample_{vis_count}.png" if vis_dir else None,
                    show=False
                )
                vis_count += 1
    
    # Compute final metrics
    results = metrics.compute()
    
    return results


def print_results(results: dict, model_name: str):
    """Print evaluation results in a nice format."""
    print("\n" + "=" * 50)
    print(f"Evaluation Results - {model_name}")
    print("=" * 50)
    print(f"IoU (Jaccard):     {results['iou']:.4f}")
    print(f"F1 Score:          {results['f1']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"Overall Accuracy:  {results['accuracy']:.4f}")
    print("=" * 50)
    
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {results['tp']:,}")
    print(f"  True Negatives:  {results['tn']:,}")
    print(f"  False Positives: {results['fp']:,}")
    print(f"  False Negatives: {results['fn']:,}")
    print("=" * 50)


def main():
    args = parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloader (test set only)
    print("Loading test data...")
    loaders = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    test_loader = loaders["test"]
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(
        args.model,
        encoder_name=args.encoder if "unet" in args.model else None
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        print(f"Checkpoint val IoU: {checkpoint['metrics'].get('iou', 'N/A')}")
    
    # Evaluate
    results = evaluate(
        model, test_loader, device,
        output_dir=output_dir,
        visualize=args.visualize,
        num_vis=args.num_vis
    )
    
    # Print results
    print_results(results, args.model)
    
    # Save results
    results_path = output_dir / "results.yaml"
    with open(results_path, "w") as f:
        yaml.dump({k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in results.items()}, f)
    print(f"\nResults saved to: {results_path}")
    
    if args.visualize:
        print(f"Visualizations saved to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()

