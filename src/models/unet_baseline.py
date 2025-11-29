"""
U-Net Baseline Model for Change Detection

This is a simple baseline that concatenates the two images
and uses a standard U-Net for segmentation.

Owner: [TEAM MEMBER 1 - Maggie Tu]

Model Architecture:
- Input: Concatenated before/after images (6 channels)
- Encoder: Pre-trained ResNet/EfficientNet backbone
- Decoder: U-Net style decoder with skip connections
- Output: Binary change mask
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetBaseline(nn.Module):
    """
    Baseline U-Net for change detection.
    
    Simply concatenates before/after images and treats
    change detection as semantic segmentation.
    
    Pros:
    - Simple to implement and train
    - Fast inference
    - Good baseline performance
    
    Cons:
    - Doesn't explicitly model temporal relationship
    - May miss subtle changes
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 6,  # 3 (before) + 3 (after) = 6
        num_classes: int = 1,
        activation: str = None
    ):
        """
        Args:
            encoder_name: Name of encoder backbone (resnet34, efficientnet-b0, etc.)
            encoder_weights: Pre-trained weights to use
            in_channels: Number of input channels (6 for concatenated RGB pairs)
            num_classes: Number of output classes (1 for binary change detection)
            activation: Final activation function
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
        self.encoder_name = encoder_name
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img_a: Before image (B, 3, H, W)
            img_b: After image (B, 3, H, W)
        
        Returns:
            Change mask logits (B, 1, H, W)
        """
        # Concatenate images along channel dimension
        x = torch.cat([img_a, img_b], dim=1)  # (B, 6, H, W)
        
        # Forward through U-Net
        output = self.model(x)
        
        return output
    
    def get_encoder_params(self):
        """Get encoder parameters for differential learning rates."""
        return self.model.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates."""
        return self.model.decoder.parameters()


class UNetPlusPlus(nn.Module):
    """
    U-Net++ variant for change detection.
    Uses nested skip connections for better feature fusion.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 6,
        num_classes: int = 1,
        activation: str = None
    ):
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_a, img_b], dim=1)
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ variant for change detection.
    Uses atrous convolutions for multi-scale features.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 6,
        num_classes: int = 1,
        activation: str = None
    ):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_a, img_b], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Test model
    model = UNetBaseline(encoder_name="resnet34")
    
    # Create dummy inputs
    batch_size = 2
    img_a = torch.randn(batch_size, 3, 256, 256)
    img_b = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    output = model(img_a, img_b)
    
    print(f"Input A shape: {img_a.shape}")
    print(f"Input B shape: {img_b.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

