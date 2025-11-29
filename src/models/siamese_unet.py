"""
Siamese U-Net for Change Detection

The main model for this project. Uses weight-shared encoders
to extract features from both images, then compares them.

Owner: [TEAM MEMBER 2 - Laura Li]

Model Architecture:
- Dual input: Before image (A) and After image (B)
- Siamese encoder: Same weights process both images
- Feature difference: Compute difference between encoded features
- Decoder: Reconstruct change mask from difference features

Reference:
- "Fully Convolutional Siamese Networks for Change Detection"
- https://github.com/qubvel/segmentation_models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Tuple


class SiameseEncoder(nn.Module):
    """
    Shared encoder for Siamese network.
    Processes both images with the same weights.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3
    ):
        super().__init__()
        
        # Use SMP encoder
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        self.out_channels = self.encoder.out_channels
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        return self.encoder(x)


class DifferenceModule(nn.Module):
    """
    Computes difference between feature maps from two time points.
    Multiple strategies available.
    """
    
    def __init__(self, mode: str = "subtract"):
        """
        Args:
            mode: How to compute difference
                - 'subtract': Simple subtraction |F_a - F_b|
                - 'concat': Concatenate features
                - 'concat_diff': Concatenate + difference
        """
        super().__init__()
        self.mode = mode
        
    def forward(
        self, 
        features_a: List[torch.Tensor], 
        features_b: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute difference between multi-scale features.
        
        Args:
            features_a: Features from image A at multiple scales
            features_b: Features from image B at multiple scales
        
        Returns:
            Difference features at multiple scales
        """
        diff_features = []
        
        for fa, fb in zip(features_a, features_b):
            if self.mode == "subtract":
                diff = torch.abs(fa - fb)
            elif self.mode == "concat":
                diff = torch.cat([fa, fb], dim=1)
            elif self.mode == "concat_diff":
                diff = torch.cat([fa, fb, torch.abs(fa - fb)], dim=1)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            diff_features.append(diff)
        
        return diff_features


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for change detection.
    
    Architecture:
    1. Siamese encoder processes both images
    2. Difference module computes feature differences
    3. U-Net decoder reconstructs change mask
    
    This is the MAIN MODEL for the project.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        diff_mode: str = "subtract",
        activation: str = None
    ):
        """
        Args:
            encoder_name: Backbone encoder
            encoder_weights: Pre-trained weights
            in_channels: Input channels per image
            num_classes: Output classes
            diff_mode: How to compute feature differences
            activation: Final activation
        """
        super().__init__()
        
        # Siamese encoder (shared weights)
        self.encoder = SiameseEncoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )
        
        # Difference module
        self.diff_mode = diff_mode
        self.difference = DifferenceModule(mode=diff_mode)
        
        # Calculate decoder input channels based on diff mode
        encoder_channels = self.encoder.out_channels
        if diff_mode == "subtract":
            decoder_channels = encoder_channels
        elif diff_mode == "concat":
            decoder_channels = [c * 2 for c in encoder_channels]
        elif diff_mode == "concat_diff":
            decoder_channels = [c * 3 for c in encoder_channels]
        else:
            decoder_channels = encoder_channels
        
        # U-Net decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=decoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            activation=activation,
            kernel_size=3
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img_a: Before image (B, 3, H, W)
            img_b: After image (B, 3, H, W)
        
        Returns:
            Change mask logits (B, 1, H, W)
        """
        # Extract features with shared encoder
        features_a = self.encoder(img_a)
        features_b = self.encoder(img_b)
        
        # Compute difference features
        diff_features = self.difference(features_a, features_b)
        
        # Decode
        decoder_output = self.decoder(*diff_features)
        
        # Segmentation head
        output = self.segmentation_head(decoder_output)
        
        return output


class SiameseUNetAttention(nn.Module):
    """
    Siamese U-Net with attention mechanisms.
    Adds spatial attention to focus on changed regions.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        
        # Siamese encoder
        self.encoder = SiameseEncoder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels
        )
        
        encoder_channels = self.encoder.out_channels
        
        # Attention modules for each scale
        self.attention_modules = nn.ModuleList([
            SpatialAttention(c) for c in encoder_channels[1:]  # Skip first (input)
        ])
        
        # Decoder with attention
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            attention_type="scse"  # Squeeze-and-Excitation
        )
        
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=3
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        # Extract features
        features_a = self.encoder(img_a)
        features_b = self.encoder(img_b)
        
        # Compute attended difference features
        diff_features = [features_a[0]]  # First feature unchanged
        
        for i, (fa, fb, attn) in enumerate(zip(
            features_a[1:], features_b[1:], self.attention_modules
        )):
            diff = torch.abs(fa - fb)
            attended_diff = attn(diff)
            diff_features.append(attended_diff)
        
        # Decode
        decoder_output = self.decoder(*diff_features)
        output = self.segmentation_head(decoder_output)
        
        return output


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


if __name__ == "__main__":
    # Test Siamese U-Net
    model = SiameseUNet(
        encoder_name="resnet34",
        diff_mode="subtract"
    )
    
    batch_size = 2
    img_a = torch.randn(batch_size, 3, 256, 256)
    img_b = torch.randn(batch_size, 3, 256, 256)
    
    output = model(img_a, img_b)
    
    print(f"Input A shape: {img_a.shape}")
    print(f"Input B shape: {img_b.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test attention variant
    print("\n--- Siamese U-Net with Attention ---")
    model_attn = SiameseUNetAttention(encoder_name="resnet34")
    output_attn = model_attn(img_a, img_b)
    print(f"Output shape: {output_attn.shape}")

