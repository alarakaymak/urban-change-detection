link to colab notebook: https://colab.research.google.com/drive/1N96d9CGa19eNKYu4eBCLvi8Hltv4Bptk?usp=sharing
# Siamese U-Net for Satellite Image Change Detection

## Overview
Implemented and trained a **Siamese U-Net** deep learning model to detect changes between satellite images captured at different time points using the **LEVIR-CD dataset**.

## Model Architecture
- **Siamese U-Net**: Dual-encoder architecture with shared weights
- Processes two input images (T1 and T2) through parallel encoders
- Concatenates multi-scale features at each level
- Decoder produces binary change mask (changed/unchanged)
- **Total Parameters**: ~54.4 million

## Dataset
- **LEVIR-CD** (Remote Sensing Building Change Detection Dataset)
- Train: 445 image pairs
- Validation: 64 image pairs  
- Test: 128 image pairs
- Image size: 256×256 pixels

## Training Configuration
- **Loss Function**: Combined BCE + Dice Loss (0.5 + 0.5)
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 8
- **Epochs**: 50
- **Best Model**: Epoch 48

## Results

### Test Set Performance
| Metric | Score |
|--------|-------|
| **IoU** | 68.58% |
| **F1 Score** | 81.09% |
| **Precision** | 83.84% |
| **Recall** | 79.07% |

### Key Findings
- ✅ **High Precision (83.84%)**: Model produces few false positives - most detected changes are real
- ✅ **Good Recall (79.07%)**: Captures majority of actual changes
- ✅ **Strong Generalization**: Test IoU (68.58%) close to validation IoU (69.67%) - no overfitting
- ✅ **Balanced Performance**: F1 score of 81.09% indicates well-balanced precision-recall tradeoff

## Sample Results
The model successfully detects building construction, demolition, and urban development changes between satellite image pairs, with accurate binary segmentation masks.

## Implementation
- **Framework**: PyTorch
- **Environment**: Google Colab with GPU
- **Training Time**: ~19 seconds/epoch
- **Total Training**: ~16 minutes
