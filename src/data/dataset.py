"""
LEVIR-CD Dataset Loader
Dataset: https://justchenhao.github.io/LEVIR/

The LEVIR-CD dataset contains:
- A (before): Images before change
- B (after): Images after change  
- label: Binary change masks (ground truth)

Ground truth masks are binary:
- 0 (black): No change
- 255 (white): Change detected
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD Change Detection Dataset
    
    Expects directory structure:
    root_dir/
    ├── train/
    │   ├── A/          # Before images
    │   ├── B/          # After images
    │   └── label/      # Ground truth change masks
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        patch_size: int = 256,
        use_patches: bool = True
    ):
        """
        Args:
            root_dir: Path to LEVIR-CD dataset root
            split: One of 'train', 'val', 'test'
            transform: Albumentations transform to apply
            patch_size: Size of patches to extract (LEVIR images are 1024x1024)
            use_patches: Whether to use pre-cut patches or full images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.use_patches = use_patches
        
        # Set up paths
        self.split_dir = self.root_dir / split
        self.img_a_dir = self.split_dir / "A"
        self.img_b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"
        
        # Get all image filenames
        self.image_names = self._get_image_list()
        
        print(f"[{split.upper()}] Loaded {len(self.image_names)} samples from {self.split_dir}")
    
    def _get_image_list(self) -> List[str]:
        """Get list of image names in the dataset split."""
        if not self.img_a_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_a_dir}")
        
        # Get all png/jpg files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif']:
            image_files.extend(self.img_a_dir.glob(ext))
        
        # Return sorted list of filenames (without path)
        return sorted([f.name for f in image_files])
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            img_a: Tensor of shape (C, H, W) - before image
            img_b: Tensor of shape (C, H, W) - after image
            label: Tensor of shape (1, H, W) - binary change mask
        """
        img_name = self.image_names[idx]
        
        # Load images
        img_a = np.array(Image.open(self.img_a_dir / img_name).convert("RGB"))
        img_b = np.array(Image.open(self.img_b_dir / img_name).convert("RGB"))
        
        # Load label (convert to binary: 0 or 1)
        label_path = self.label_dir / img_name
        if not label_path.exists():
            # Try different extension
            label_path = self.label_dir / img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        
        label = np.array(Image.open(label_path).convert("L"))
        label = (label > 127).astype(np.float32)  # Binarize
        
        # Apply transforms
        if self.transform:
            # Albumentations expects images as numpy arrays
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed["image"]
            img_b = transformed["image2"]
            label = transformed["mask"]
        else:
            # Default: convert to tensor and normalize
            img_a = self._to_tensor(img_a)
            img_b = self._to_tensor(img_b)
            label = torch.from_numpy(label).unsqueeze(0)
        
        return img_a, img_b, label
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor and normalize."""
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255.0
        return img


class LEVIRCDPatchDataset(LEVIRCDDataset):
    """
    LEVIR-CD Dataset with patch extraction for training.
    Useful when working with limited GPU memory.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        patch_size: int = 256,
        stride: int = 256
    ):
        super().__init__(root_dir, split, transform, patch_size, use_patches=True)
        self.stride = stride
        
        # Pre-compute patch indices
        self.patches = self._compute_patches()
        print(f"[{split.upper()}] Total patches: {len(self.patches)}")
    
    def _compute_patches(self) -> List[Tuple[int, int, int]]:
        """Compute all patch indices (image_idx, row, col)."""
        patches = []
        
        # Load first image to get dimensions
        sample_img = Image.open(self.img_a_dir / self.image_names[0])
        img_h, img_w = sample_img.size[1], sample_img.size[0]
        
        for img_idx in range(len(self.image_names)):
            for row in range(0, img_h - self.patch_size + 1, self.stride):
                for col in range(0, img_w - self.patch_size + 1, self.stride):
                    patches.append((img_idx, row, col))
        
        return patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx, row, col = self.patches[idx]
        img_name = self.image_names[img_idx]
        
        # Load images
        img_a = np.array(Image.open(self.img_a_dir / img_name).convert("RGB"))
        img_b = np.array(Image.open(self.img_b_dir / img_name).convert("RGB"))
        label = np.array(Image.open(self.label_dir / img_name.replace('.jpg', '.png')).convert("L"))
        label = (label > 127).astype(np.float32)
        
        # Extract patch
        img_a = img_a[row:row+self.patch_size, col:col+self.patch_size]
        img_b = img_b[row:row+self.patch_size, col:col+self.patch_size]
        label = label[row:row+self.patch_size, col:col+self.patch_size]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed["image"]
            img_b = transformed["image2"]
            label = transformed["mask"]
        else:
            img_a = self._to_tensor(img_a)
            img_b = self._to_tensor(img_b)
            label = torch.from_numpy(label).unsqueeze(0)
        
        return img_a, img_b, label


if __name__ == "__main__":
    # Test dataset loading
    dataset = LEVIRCDDataset(
        root_dir="data/LEVIR-CD",
        split="train"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img_a, img_b, label = dataset[0]
        print(f"Image A shape: {img_a.shape}")
        print(f"Image B shape: {img_b.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Label unique values: {torch.unique(label)}")

