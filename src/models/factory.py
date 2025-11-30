"""
Model factory for easy model creation.

Each team member implements their own model:
- Maggie: unet_baseline.py
- Laura: siamese_unet.py  
- Alara: changeformer.py, changeformer_pretrained.py
"""

import torch.nn as nn

# Initialize empty model dict
MODELS = {}

# Try importing each model (they may not exist for all team members)
try:
    from .unet_baseline import UNetBaseline, UNetPlusPlus, DeepLabV3Plus
    MODELS["unet_baseline"] = UNetBaseline
    MODELS["unet_plusplus"] = UNetPlusPlus
    MODELS["deeplabv3plus"] = DeepLabV3Plus
except ImportError:
    print("Note: unet_baseline.py not found (Maggie's model)")

try:
    from .siamese_unet import SiameseUNet, SiameseUNetAttention
    MODELS["siamese_unet"] = SiameseUNet
    MODELS["siamese_unet_attention"] = SiameseUNetAttention
except ImportError:
    print("Note: siamese_unet.py not found (Laura's model)")

try:
    from .changeformer import ChangeFormer, ChangeFormerLite
    MODELS["changeformer"] = ChangeFormer
    MODELS["changeformer_lite"] = ChangeFormerLite
except ImportError:
    print("Note: changeformer.py not found (Alara's model)")

try:
    from .changeformer_pretrained import ChangeFormerPretrained, ChangeFormerSwin
    MODELS["changeformer_lora"] = ChangeFormerPretrained
    MODELS["changeformer_swin"] = ChangeFormerSwin
except ImportError:
    print("Note: changeformer_pretrained.py not found (Alara's model)")


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

