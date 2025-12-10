# ChangeFormer for Satellite Image Change Detection

**Link to GitHub repository:** [https://github.com/alarakaymak/urban-change-detection](https://github.com/alarakaymak/urban-change-detection)

---

## Overview

Implemented and trained a ChangeFormer deep learning model with LoRA fine-tuning to detect changes between satellite images captured at different time points using the LEVIR-CD dataset.

---

## Model Architecture

### ChangeFormer: Transformer-based Siamese Architecture

- **Vision Transformer (ViT) backbone** pre-trained on ImageNet
- **Siamese feature extraction** with shared weights for T1 and T2 images
- **Difference module** compares and concatenates temporal features
- **MLP decoder** reconstructs change mask through progressive upsampling

### LoRA Fine-tuning

- Trains only **442K parameters** (0.5% of 86M total)
- Freezes pretrained ViT weights
- Injects trainable rank decomposition matrices (rank=8, alpha=16)
- Enables efficient training on consumer hardware

---

## Dataset

### LEVIR-CD (Remote Sensing Building Change Detection Dataset)

- **Train:** 445 image pairs
- **Validation:** 64 image pairs
- **Test:** 128 image pairs
- **Image size:** 256×256 pixels
- **Resolution:** 0.5m/pixel
- **Coverage:** Urban areas in Texas, USA over 5-14 year time span
- **Annotations:** 21,000+ building-level changes (new constructions, demolitions, modifications)

---

## Training Configuration

- **Loss Function:** Combined BCE + Dice Loss (weights: 0.3 + 0.7)
- **Optimizer:** AdamW
  - Backbone learning rate: 5×10⁻⁵
  - Decoder learning rate: 5×10⁻⁴
- **Batch Size:** 4 (with gradient accumulation = 8 effective batch size)
- **Epochs:** 50
- **Scheduler:** Cosine annealing with warm restarts
- **Warmup:** 5 epochs linear warmup
- **Best Model:** Epoch 34 (validation IoU: 69.62%)

### Data Augmentation

- Random horizontal/vertical flips
- Random rotations (90°, 180°, 270°)
- Random crops to 256×256
- Color augmentations (brightness/contrast, hue/saturation)
- Gaussian blur
- All geometric transforms applied consistently to both temporal images

---

## Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **IoU** | 69.61% |
| **F1 Score** | 82.08% |
| **Precision** | 81.94% |
| **Recall** | 82.21% |
| **Accuracy** | 98.20% |

---

## Key Findings

✅ **Balanced Precision/Recall (81.94%/82.21%)**: Model detects most changes with few false alarms

✅ **Strong Generalization**: Test IoU (69.61%) close to validation IoU (69.62%) - no overfitting

✅ **Parameter Efficient**: LoRA trains only 0.5% of parameters, enabling training on consumer hardware

✅ **Global Context**: Transformer captures long-range dependencies for large building detection

✅ **Large Building Detection**: Model excels at detecting changes >500 square meters

⚠️ **Limitations**: 
- Small buildings (<100 square meters) sometimes missed due to 16×16 patch size
- Predicted boundaries tend to be smoother than ground truth
- Higher memory requirements than CNN-based approaches

---

## Sample Results

The model successfully detects building construction, demolition, and urban development changes between satellite image pairs, with accurate binary segmentation masks. Visualizations show:

- Accurate detection of large building changes
- Good handling of lighting and seasonal variations
- Reliable change boundaries for major urban modifications

---

## Implementation

- **Framework:** PyTorch
- **Environment:** Google Colab with GPU / Local with MPS (Apple Silicon)
- **Trainable Parameters:** 442K (with LoRA)
- **Total Parameters:** ~86M
- **Training Time:** ~50 epochs with early stopping
- **Code Structure:**
  - `train_v2.py`: Training script with advanced techniques (warmup, gradient accumulation, layer-wise LR)
  - `src/models/changeformer_pretrained.py`: ChangeFormer model with LoRA support
  - `evaluate.py`: Model evaluation on test set
  - `src/utils/metrics.py`: IoU, F1, Precision, Recall implementations

---

## Future Work

- Exploring hybrid CNN-Transformer architectures to combine local and global features
- Implementing multi-scale patch sizes (8×8 + 16×16) to better capture small structures
- Applying CRF post-processing for sharper boundary predictions
- Experimenting with different LoRA ranks and alpha values
- Testing on additional change detection datasets

---

## References

- LEVIR-CD Dataset: [https://justchenhao.github.io/LEVIR/](https://justchenhao.github.io/LEVIR/)
- LoRA Paper: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
- ChangeFormer Architecture: Based on transformer-based change detection methods

---

## Team

- **Alara Kaymak** - ChangeFormer implementation and training
- **Laura Li** - Siamese U-Net implementation
- **Maggie Tu** - U-Net baseline implementation

---

*Project completed as part of Deep Learning course final project.*

