"""
ChangeFormer with Pre-trained Backbone + LoRA Support

This version uses:
1. Pre-trained Vision Transformer (ViT) or Swin as backbone
2. Optional LoRA/DoRA for efficient fine-tuning
3. Only trains ~1-5% of parameters!

Owner: Alara Kaymak

Why this is better:
- Pre-trained on ImageNet = already understands visual features
- LoRA = only train small adapter layers (much faster!)
- Less overfitting, better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import timm


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    
    Instead of updating W directly, we learn:
    W' = W + BA where B is (d, r) and A is (r, d)
    
    r << d, so we only train r*d*2 parameters instead of d*d
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling


class DoRALayer(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).
    
    DoRA decomposes weight into magnitude and direction:
    W' = m * (W + BA) / ||W + BA||
    
    Generally better than LoRA for vision tasks.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude parameter (DoRA specific)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        # Compute adapted weight
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        adapted_weight = original_weight + lora_weight
        
        # Normalize and apply magnitude (DoRA)
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        normalized_weight = adapted_weight / (weight_norm + 1e-8)
        final_weight = self.magnitude.unsqueeze(1) * normalized_weight
        
        return F.linear(self.dropout(x), final_weight)


class ChangeFormerPretrained(nn.Module):
    """
    ChangeFormer with Pre-trained ViT Backbone + LoRA.
    
    Architecture:
    1. Pre-trained ViT extracts features from both images
    2. LoRA layers fine-tune the transformer efficiently
    3. Difference module compares features
    4. Decoder predicts change mask
    
    Benefits:
    - Pre-trained = starts with good features
    - LoRA = only ~1-5% trainable parameters
    - Faster training, less overfitting
    """
    
    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",  # Small ViT, good balance
        pretrained: bool = True,
        num_classes: int = 1,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        freeze_backbone: bool = True
    ):
        """
        Args:
            backbone: timm model name (vit_small_patch16_224, swin_tiny_patch4_window7_224, etc.)
            pretrained: Use ImageNet pre-trained weights
            num_classes: Output classes (1 for binary change detection)
            use_lora: Whether to use LoRA adapters
            lora_rank: LoRA rank (lower = fewer params, higher = more capacity)
            lora_alpha: LoRA scaling factor
            freeze_backbone: Freeze backbone weights (only train LoRA + decoder)
        """
        super().__init__()
        
        self.use_lora = use_lora
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""  # Keep spatial features
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 3:  # ViT: (B, N, C)
                self.feat_dim = features.shape[-1]
                self.num_patches = features.shape[1]
            else:  # CNN: (B, C, H, W)
                self.feat_dim = features.shape[1]
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Backbone frozen ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
        else:
            print(f"✓ Backbone UNFROZEN - full fine-tuning ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
        
        # Add LoRA layers to attention
        if use_lora:
            self.lora_layers = nn.ModuleList()
            self._add_lora_to_backbone(lora_rank, lora_alpha)
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"✓ LoRA added: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
        
        # Difference module
        self.diff_conv = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU()
        )
        
        # Decoder (simple but effective)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feat_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
    def _add_lora_to_backbone(self, rank: int, alpha: float):
        """Add LoRA layers to transformer attention."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear) and 'attn' in name:
                # Add LoRA to attention projections
                in_feat = module.in_features
                out_feat = module.out_features
                
                lora = LoRALayer(in_feat, out_feat, rank=rank, alpha=alpha)
                self.lora_layers.append(lora)
                
                # Store original forward
                original_forward = module.forward
                
                # Create new forward that adds LoRA
                def make_lora_forward(orig_fn, lora_layer):
                    def lora_forward(x):
                        return orig_fn(x) + lora_layer(x)
                    return lora_forward
                
                module.forward = make_lora_forward(original_forward, lora)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using backbone."""
        # Resize if needed (ViT expects 224x224)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)
        
        # Handle different output shapes
        if len(features.shape) == 3:  # ViT: (B, N, C)
            # Remove CLS token if present, reshape to spatial
            B, N, C = features.shape
            H = W = int((N - 1) ** 0.5) if N > 1 else 1
            
            if N == H * W + 1:  # Has CLS token
                features = features[:, 1:, :]  # Remove CLS
            
            features = features.transpose(1, 2).reshape(B, C, H, W)
        
        return features
    
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img_a: Before image (B, 3, H, W)
            img_b: After image (B, 3, H, W)
        
        Returns:
            Change mask logits (B, 1, H, W)
        """
        original_size = img_a.shape[-2:]
        
        # Extract features
        feat_a = self.extract_features(img_a)
        feat_b = self.extract_features(img_b)
        
        B, C, H, W = feat_a.shape
        
        # Compute difference features
        feat_a_flat = feat_a.flatten(2).transpose(1, 2)  # (B, H*W, C)
        feat_b_flat = feat_b.flatten(2).transpose(1, 2)
        
        # Concatenate and process difference
        diff = torch.cat([feat_a_flat, feat_b_flat], dim=-1)  # (B, H*W, 2C)
        diff = self.diff_conv(diff)  # (B, H*W, C)
        
        # Also add absolute difference
        abs_diff = torch.abs(feat_a_flat - feat_b_flat)
        diff = diff + abs_diff
        
        # Reshape for decoder
        diff = diff.transpose(1, 2).reshape(B, C, H, W)
        
        # Decode
        output = self.decoder(diff)
        
        # Resize to original
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


class ChangeFormerSwin(nn.Module):
    """
    ChangeFormer using Swin Transformer backbone.
    
    Swin is often better than ViT for dense prediction tasks
    because it has hierarchical features.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Swin-Tiny is a good balance of speed and accuracy
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True  # Get multi-scale features
        )
        
        # Freeze if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Swin backbone frozen")
        
        # Get actual feature channels from backbone
        self.feat_channels = self.backbone.feature_info.channels()
        print(f"  Feature channels: {self.feat_channels}")
        
        # Fusion modules for each scale
        self.diff_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            )
            for c in self.feat_channels
        ])
        
        # Simple decoder - upsample and reduce channels
        self.decoder = nn.Sequential(
            nn.Conv2d(self.feat_channels[-1], 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, 1)
        )
        
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        original_size = img_a.shape[-2:]
        
        # Resize for Swin
        if img_a.shape[-1] != 224:
            img_a = F.interpolate(img_a, size=(224, 224), mode='bilinear', align_corners=False)
            img_b = F.interpolate(img_b, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract multi-scale features
        feats_a = self.backbone(img_a)
        feats_b = self.backbone(img_b)
        
        # Use deepest features for simplicity
        fa = feats_a[-1]
        fb = feats_b[-1]
        
        # Swin outputs (B, H, W, C), need to convert to (B, C, H, W)
        if fa.dim() == 4 and fa.shape[-1] == self.feat_channels[-1]:
            fa = fa.permute(0, 3, 1, 2)
            fb = fb.permute(0, 3, 1, 2)
        
        # Compute difference
        diff = torch.abs(fa - fb)
        
        # Decode
        output = self.decoder(diff)
        
        # Resize to original
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


def get_trainable_params(model):
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model):
    """Get total number of parameters."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pre-trained ChangeFormer with LoRA")
    print("=" * 60)
    
    # Test ViT + LoRA version
    print("\n1. ViT + LoRA:")
    model_lora = ChangeFormerPretrained(
        backbone="vit_small_patch16_224",
        pretrained=True,
        use_lora=True,
        lora_rank=8,
        freeze_backbone=True
    )
    
    img_a = torch.randn(2, 3, 256, 256)
    img_b = torch.randn(2, 3, 256, 256)
    
    output = model_lora(img_a, img_b)
    print(f"   Output shape: {output.shape}")
    print(f"   Trainable: {get_trainable_params(model_lora):,}")
    print(f"   Total: {get_total_params(model_lora):,}")
    
    # Test Swin version
    print("\n2. Swin Transformer:")
    model_swin = ChangeFormerSwin(pretrained=True, freeze_backbone=True)
    
    output_swin = model_swin(img_a, img_b)
    print(f"   Output shape: {output_swin.shape}")
    print(f"   Trainable: {get_trainable_params(model_swin):,}")
    print(f"   Total: {get_total_params(model_swin):,}")
    
    print("\n" + "=" * 60)
    print("✅ All models working!")
    print("=" * 60)

