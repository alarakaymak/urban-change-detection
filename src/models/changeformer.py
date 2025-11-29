"""
ChangeFormer - Transformer-based Change Detection

Advanced model using Vision Transformer concepts for change detection.
Uses self-attention to capture long-range dependencies.

Owner: [TEAM MEMBER 3 - Alara Kaymak]

Model Architecture:
- Siamese transformer encoder
- Multi-scale feature extraction
- Cross-attention between time points
- Transformer decoder for change prediction

Reference:
- "A Transformer-Based Siamese Network for Change Detection"
- ChangeFormer: https://github.com/wgcban/ChangeFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x, (H, W)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention between two feature maps."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Query features (B, N, C)
            context: Key/Value features (B, M, C)
        """
        B, N, C = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DifferenceTransformerBlock(nn.Module):
    """Transformer block with cross-attention for change detection."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads)
        
        # Cross-attention (A attends to B)
        self.norm2 = nn.LayerNorm(dim)
        self.norm2_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_a: Features from image A
            x_b: Features from image B
        Returns:
            Difference features
        """
        # Compute difference
        diff = x_a - x_b
        
        # Self-attention on difference
        diff = diff + self.self_attn(self.norm1(diff))
        
        # Cross-attention: diff attends to both original features
        context = torch.cat([x_a, x_b], dim=1)
        diff = diff + self.cross_attn(self.norm2(diff), self.norm2_ctx(context))
        
        # MLP
        diff = diff + self.mlp(self.norm3(diff))
        
        return diff


class ChangeFormer(nn.Module):
    """
    Transformer-based Change Detection Network.
    
    Uses:
    - Siamese CNN encoder for initial feature extraction
    - Transformer layers for global context
    - Cross-attention for temporal comparison
    - Hierarchical decoder for multi-scale prediction
    """
    
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Hierarchical CNN backbone (shared for both images)
        self.backbone = nn.ModuleList([
            self._make_stage(in_channels, embed_dim, stride=2),      # 1/2
            self._make_stage(embed_dim, embed_dim * 2, stride=2),    # 1/4
            self._make_stage(embed_dim * 2, embed_dim * 4, stride=2),# 1/8
            self._make_stage(embed_dim * 4, embed_dim * 8, stride=2) # 1/16
        ])
        
        # Transformer blocks at bottleneck
        self.transformer_blocks = nn.ModuleList([
            DifferenceTransformerBlock(
                dim=embed_dim * 8,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate
            )
            for _ in range(depth)
        ])
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList([
            self._make_decoder_stage(embed_dim * 8, embed_dim * 4),  # 1/8
            self._make_decoder_stage(embed_dim * 4, embed_dim * 2),  # 1/4
            self._make_decoder_stage(embed_dim * 2, embed_dim),      # 1/2
            self._make_decoder_stage(embed_dim, embed_dim // 2)      # 1/1
        ])
        
        # Difference modules for skip connections
        self.diff_convs = nn.ModuleList([
            nn.Conv2d(embed_dim * 4 * 2, embed_dim * 4, 1),  # For stage 3
            nn.Conv2d(embed_dim * 2 * 2, embed_dim * 2, 1),  # For stage 2
            nn.Conv2d(embed_dim * 2, embed_dim, 1),          # For stage 1
        ])
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, 1)
        )
        
    def _make_stage(self, in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
        """Create a backbone stage."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_stage(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create a decoder stage with upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
        # Extract hierarchical features
        features_a = []
        features_b = []
        
        x_a, x_b = img_a, img_b
        for stage in self.backbone:
            x_a = stage(x_a)
            x_b = stage(x_b)
            features_a.append(x_a)
            features_b.append(x_b)
        
        # Get bottleneck features
        B, C, H, W = x_a.shape
        
        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        x_a_flat = x_a.flatten(2).transpose(1, 2)
        x_b_flat = x_b.flatten(2).transpose(1, 2)
        
        # Apply transformer blocks
        diff = x_a_flat
        for block in self.transformer_blocks:
            diff = block(x_a_flat, x_b_flat)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        diff = diff.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder with skip connections
        for i, (decoder_stage, diff_conv) in enumerate(zip(
            self.decoder[:-1], self.diff_convs
        )):
            diff = decoder_stage(diff)
            
            # Skip connection with difference features
            skip_idx = len(features_a) - 2 - i
            if skip_idx >= 0:
                skip_diff = torch.cat([
                    features_a[skip_idx],
                    features_b[skip_idx]
                ], dim=1)
                skip_diff = diff_conv(skip_diff)
                
                # Resize if needed
                if diff.shape[2:] != skip_diff.shape[2:]:
                    skip_diff = F.interpolate(
                        skip_diff, size=diff.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                diff = diff + skip_diff
        
        # Final decoder stage
        diff = self.decoder[-1](diff)
        
        # Resize to original size if needed
        if diff.shape[2:] != img_a.shape[2:]:
            diff = F.interpolate(
                diff, size=img_a.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # Prediction head
        output = self.head(diff)
        
        return output


class ChangeFormerLite(nn.Module):
    """
    Lightweight version of ChangeFormer.
    Uses fewer parameters and simpler architecture.
    Good for faster training and experimentation.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 32,
        num_heads: int = 2,
        depth: int = 2
    ):
        super().__init__()
        
        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.ReLU(inplace=True)
        )
        
        # Simple attention for comparison
        self.attention = nn.MultiheadAttention(
            embed_dim * 4, num_heads=num_heads, batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim * 2, embed_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim, num_classes, 4, stride=2, padding=1)
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        # Encode both images
        feat_a = self.encoder(img_a)
        feat_b = self.encoder(img_b)
        
        B, C, H, W = feat_a.shape
        
        # Compute difference
        diff = torch.abs(feat_a - feat_b)
        
        # Apply attention
        diff_flat = diff.flatten(2).transpose(1, 2)  # (B, H*W, C)
        diff_attn, _ = self.attention(diff_flat, diff_flat, diff_flat)
        diff = diff_attn.transpose(1, 2).reshape(B, C, H, W)
        
        # Decode
        output = self.decoder(diff)
        
        # Resize if needed
        if output.shape[2:] != img_a.shape[2:]:
            output = F.interpolate(
                output, size=img_a.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        return output


if __name__ == "__main__":
    # Test ChangeFormer
    print("--- ChangeFormer ---")
    model = ChangeFormer(
        img_size=256,
        embed_dim=64,
        num_heads=4,
        depth=2
    )
    
    batch_size = 2
    img_a = torch.randn(batch_size, 3, 256, 256)
    img_b = torch.randn(batch_size, 3, 256, 256)
    
    output = model(img_a, img_b)
    
    print(f"Input shape: {img_a.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test lite version
    print("\n--- ChangeFormer Lite ---")
    model_lite = ChangeFormerLite(embed_dim=32)
    output_lite = model_lite(img_a, img_b)
    print(f"Output shape: {output_lite.shape}")
    
    total_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"Total parameters: {total_params_lite:,}")

