"""
Data augmentation transforms for change detection.
Using Albumentations library for consistent transformations
across image pairs (A, B) and labels.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size: int = 256):
    """
    Training augmentations.
    
    Important: For change detection, we need to apply the SAME
    geometric transforms to both images and different color
    transforms to each image independently.
    """
    return A.Compose([
        # Geometric transforms (applied to both images)
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Color transforms (applied independently)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.5),
        
        # Noise (GaussianNoise in newer albumentations)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ],
    additional_targets={'image2': 'image'}  # Apply same geometric transforms to image2
    )


def get_val_transforms(image_size: int = 256):
    """
    Validation/Test transforms.
    Only resize, normalize, and convert to tensor.
    """
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ],
    additional_targets={'image2': 'image'}
    )


def get_test_transforms(image_size: int = 256):
    """
    Test transforms - same as validation.
    """
    return get_val_transforms(image_size)


class PairedTransform:
    """
    Custom transform class for paired images in change detection.
    Ensures consistent geometric transforms across both images.
    """
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image_a, image_b, mask):
        """
        Apply transforms to image pair and mask.
        
        Args:
            image_a: Before image (numpy array, HWC)
            image_b: After image (numpy array, HWC)
            mask: Binary change mask (numpy array, HW)
        
        Returns:
            Dictionary with transformed images and mask
        """
        transformed = self.transform(
            image=image_a,
            image2=image_b,
            mask=mask
        )
        return {
            'image': transformed['image'],
            'image2': transformed['image2'],
            'mask': transformed['mask']
        }


if __name__ == "__main__":
    import numpy as np
    
    # Test transforms
    train_tf = get_train_transforms(256)
    val_tf = get_val_transforms(256)
    
    # Create dummy images
    img_a = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img_b = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.float32)
    
    # Apply transforms
    result = train_tf(image=img_a, image2=img_b, mask=mask)
    
    print(f"Transformed image A shape: {result['image'].shape}")
    print(f"Transformed image B shape: {result['image2'].shape}")
    print(f"Transformed mask shape: {result['mask'].shape}")

