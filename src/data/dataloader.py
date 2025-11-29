"""
DataLoader factory for LEVIR-CD dataset.
"""

from torch.utils.data import DataLoader
from .dataset import LEVIRCDDataset, LEVIRCDPatchDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


def get_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    use_patches: bool = False,
    patch_stride: int = 256
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Path to LEVIR-CD dataset
        batch_size: Batch size for dataloaders
        image_size: Size to resize/crop images to
        num_workers: Number of workers for data loading
        use_patches: Whether to use patch-based dataset
        patch_stride: Stride for patch extraction
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Get transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    test_transform = get_test_transforms(image_size)
    
    # Create datasets
    if use_patches:
        DatasetClass = LEVIRCDPatchDataset
        train_dataset = DatasetClass(
            root_dir=root_dir,
            split="train",
            transform=train_transform,
            patch_size=image_size,
            stride=patch_stride
        )
        val_dataset = LEVIRCDDataset(
            root_dir=root_dir,
            split="val",
            transform=val_transform,
            patch_size=image_size
        )
        test_dataset = LEVIRCDDataset(
            root_dir=root_dir,
            split="test",
            transform=test_transform,
            patch_size=image_size
        )
    else:
        train_dataset = LEVIRCDDataset(
            root_dir=root_dir,
            split="train",
            transform=train_transform,
            patch_size=image_size
        )
        val_dataset = LEVIRCDDataset(
            root_dir=root_dir,
            split="val",
            transform=val_transform,
            patch_size=image_size
        )
        test_dataset = LEVIRCDDataset(
            root_dir=root_dir,
            split="test",
            transform=test_transform,
            patch_size=image_size
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test dataloader creation
    loaders = get_dataloaders(
        root_dir="data/LEVIR-CD",
        batch_size=4,
        image_size=256,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")

