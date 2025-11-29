"""
Model factory for easy model creation.
"""

import torch.nn as nn
from .unet_baseline import UNetBaseline, UNetPlusPlus, DeepLabV3Plus
from .siamese_unet import SiameseUNet, SiameseUNetAttention
from .changeformer import ChangeFormer, ChangeFormerLite


MODELS = {
    # Baseline models (Maggie)
    "unet_baseline": UNetBaseline,
    "unet_plusplus": UNetPlusPlus,
    "deeplabv3plus": DeepLabV3Plus,
    
    # Siamese models (Laura)
    "siamese_unet": SiameseUNet,
    "siamese_unet_attention": SiameseUNetAttention,
    
    # Transformer models (Alara)
    "changeformer": ChangeFormer,
    "changeformer_lite": ChangeFormerLite,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Get model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    
    Example:
        >>> model = get_model("siamese_unet", encoder_name="resnet50")
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODELS.keys())}"
        )
    
    return MODELS[model_name](**kwargs)


def list_models():
    """List all available models."""
    return list(MODELS.keys())


if __name__ == "__main__":
    print("Available models:")
    for name in list_models():
        print(f"  - {name}")

