"""
LEVIR-CD Dataset Download Script

Dataset: LEVIR-CD (Large-scale Building Change Detection Dataset)
- 637 high-resolution image pairs (1024×1024)
- 21,000+ annotated building changes
- Binary change masks

Download options:
1. Google Drive (recommended)
2. Baidu Drive

NOTE: You may need to manually download from the website:
https://justchenhao.github.io/LEVIR/
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import urllib.request


# Dataset structure expected:
# data/LEVIR-CD/
# ├── train/
# │   ├── A/          (before images)
# │   ├── B/          (after images)
# │   └── label/      (change masks)
# ├── val/
# │   ├── A/
# │   ├── B/
# │   └── label/
# └── test/
#     ├── A/
#     ├── B/
#     └── label/


def create_directory_structure(base_dir: str):
    """Create the expected directory structure."""
    base = Path(base_dir)
    
    for split in ["train", "val", "test"]:
        for subdir in ["A", "B", "label"]:
            (base / split / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {base}")
    return base


def download_with_progress(url: str, filepath: str):
    """Download file with progress bar."""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print()


def print_manual_instructions():
    """Print instructions for manual download."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LEVIR-CD Dataset Download Instructions                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DOWNLOAD LINKS (Google Drive):                                              ║
║                                                                              ║
║  Option 1 - All in one folder:                                               ║
║  https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim    ║
║    • train.zip (1.6 GB)                                                      ║
║    • val.zip (234.7 MB)                                                      ║
║    • test.zip (473.3 MB)                                                     ║
║                                                                              ║
║  Option 2 - Individual links:                                                ║
║  https://drive.google.com/open?id=18GuoCuBn48oZKAlEo-LrNwABrFhVALU-          ║
║  https://drive.google.com/open?id=1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG          ║
║  https://drive.google.com/open?id=1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0          ║
║  https://drive.google.com/open?id=1QZOYP-N2Nr27aDG38zTVJMunfzmrseLv          ║
║  https://drive.google.com/open?id=1C2B5_wiZVUt-PWHn14H9lSKF5sA6jowR          ║
║                                                                              ║
║  STEPS:                                                                      ║
║  1. Download train.zip, val.zip, test.zip from links above                   ║
║  2. Extract each zip file                                                    ║
║  3. Organize into this structure:                                            ║
║                                                                              ║
║     data/LEVIR-CD/                                                           ║
║     ├── train/                                                               ║
║     │   ├── A/        (before images)                                        ║
║     │   ├── B/        (after images)                                         ║
║     │   └── label/    (change masks)                                         ║
║     ├── val/                                                                 ║
║     │   ├── A/                                                               ║
║     │   ├── B/                                                               ║
║     │   └── label/                                                           ║
║     └── test/                                                                ║
║         ├── A/                                                               ║
║         ├── B/                                                               ║
║         └── label/                                                           ║
║                                                                              ║
║  4. Verify setup: python scripts/download_data.py --verify                   ║
║                                                                              ║
║  Official website: https://justchenhao.github.io/LEVIR/                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


def verify_dataset(base_dir: str) -> bool:
    """Verify the dataset is correctly set up."""
    base = Path(base_dir)
    
    required_dirs = [
        "train/A", "train/B", "train/label",
        "val/A", "val/B", "val/label",
        "test/A", "test/B", "test/label"
    ]
    
    print("\nVerifying dataset structure...")
    all_ok = True
    
    for d in required_dirs:
        path = base / d
        if path.exists():
            # Count files
            files = list(path.glob("*"))
            image_files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]]
            print(f"  ✓ {d}: {len(image_files)} files")
            if len(image_files) == 0:
                print(f"    ⚠ Warning: No image files found in {d}")
                all_ok = False
        else:
            print(f"  ✗ {d}: MISSING")
            all_ok = False
    
    if all_ok:
        print("\n✓ Dataset verification PASSED!")
        print("  You can now run training with: python train.py")
    else:
        print("\n✗ Dataset verification FAILED!")
        print("  Please check the directory structure and re-download if needed.")
    
    return all_ok


def create_sample_data(base_dir: str, num_samples: int = 5):
    """
    Create sample/dummy data for testing the pipeline.
    Useful for debugging before downloading the full dataset.
    """
    import numpy as np
    from PIL import Image
    
    base = Path(base_dir)
    create_directory_structure(base_dir)
    
    print(f"\nCreating {num_samples} sample images per split for testing...")
    
    for split in ["train", "val", "test"]:
        for i in range(num_samples):
            # Create dummy images (256x256 RGB)
            img_a = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img_b = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            # Create dummy change mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Add some random "changes"
            cx, cy = np.random.randint(50, 200, 2)
            r = np.random.randint(10, 50)
            y, x = np.ogrid[:256, :256]
            mask_area = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
            mask[mask_area] = 255
            
            # Save
            Image.fromarray(img_a).save(base / split / "A" / f"sample_{i:04d}.png")
            Image.fromarray(img_b).save(base / split / "B" / f"sample_{i:04d}.png")
            Image.fromarray(mask).save(base / split / "label" / f"sample_{i:04d}.png")
    
    print(f"✓ Sample data created at: {base}")
    print("  Note: This is dummy data for testing. Download the real dataset for actual training.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LEVIR-CD Dataset Setup")
    parser.add_argument("--data_dir", type=str, default="data/LEVIR-CD",
                        help="Directory to store the dataset")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset structure")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create sample data for testing")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sample images to create")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.data_dir)
    elif args.create_sample:
        create_sample_data(args.data_dir, args.num_samples)
        verify_dataset(args.data_dir)
    else:
        # Create directory structure and print instructions
        create_directory_structure(args.data_dir)
        print_manual_instructions()


if __name__ == "__main__":
    main()

