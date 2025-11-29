# 🏙️ Urban Change Detection Using Siamese U-Net on Satellite Imagery

**Team 17:** Alara Kaymak, Laura Li, Maggie Tu

Deep Learning Mini Project - Dr. Huo

---

## 📋 Project Overview

Cities change constantly through construction, demolition, and land-use modifications. These changes need to be monitored to support:
- 🏗️ Urban planning
- 🔧 Infrastructure maintenance  
- 🚨 Disaster response

**Problem:** Manual inspection of satellite imagery is slow and expensive.

**Solution:** Automated change detection using deep learning to identify where significant changes occur between two different time periods.

---

## 🎯 Challenges

| Challenge | Description |
|-----------|-------------|
| Subtle Changes | Small or gradual changes are difficult to detect |
| Scene Variability | Lighting, seasons, and shadows vary across images |
| Pixel Precision | Precise pixel-level localization is required |
| Alignment | Siamese networks require strong alignment between image pairs |
| Class Imbalance | Limited labeled change pixels risk overfitting |

---

## 🔬 Method

### Pipeline
1. **Input:** Two satellite images of same location (T1, T2)
2. **Feature Extraction:** Shared U-Net encoder
3. **Comparison:** Decoder merges differences
4. **Output:** Binary change mask

### Models

| Model | Description | Owner |
|-------|-------------|-------|
| **U-Net Baseline** | Standard U-Net with concatenated inputs | Maggie |
| **Siamese U-Net** | Shared encoder weights for temporal comparison | Laura |
| **ChangeFormer** | Transformer-based with attention mechanisms | Alara |

### Loss Functions
- **BCE Loss:** Binary Cross-Entropy for pixel-wise classification
- **Dice Loss:** Handles class imbalance (fewer changed pixels)
- **Combined:** BCE + Dice (weighted combination)

### Evaluation Metrics
- **IoU (Intersection over Union):** Overlap between prediction and ground truth
- **F1 Score:** Harmonic mean of precision and recall
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)

---

## 📊 Dataset: LEVIR-CD

| Property | Value |
|----------|-------|
| Resolution | 1024×1024 pixels (256×256 patches) |
| Image Pairs | 637 |
| Annotated Changes | 21,000+ building changes |
| Format | RGB images + Binary masks |

**Links:**
- 📥 [Dataset Download](https://justchenhao.github.io/LEVIR/)
- 📚 [Segmentation Models Reference](https://github.com/qubvel/segmentation_models)

**Ground Truth:** Binary masks showing where new structures appeared (white) or old ones were removed (white). Black = no change.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/[your-repo]/urban-change-detection.git
cd urban-change-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Create directory structure and see download instructions
python scripts/download_data.py

# After downloading, verify the dataset
python scripts/download_data.py --verify

# OR create sample data for testing (optional)
python scripts/download_data.py --create_sample --num_samples 20
```

### 3. Train a Model

```bash
# Train Siamese U-Net (main model)
python train.py --model siamese_unet --epochs 100 --batch_size 8

# Train U-Net baseline
python train.py --model unet_baseline --epochs 100 --batch_size 8

# Train ChangeFormer (transformer-based)
python train.py --model changeformer --epochs 100 --batch_size 4
```

### 4. Evaluate

```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/siamese_unet_*/best_model.pth --model siamese_unet --visualize
```

---

## 📁 Project Structure

```
urban-change-detection/
├── config/
│   └── config.yaml           # Training configuration
├── data/
│   └── LEVIR-CD/             # Dataset (download separately)
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data/
│   │   ├── dataset.py        # LEVIR-CD dataset loader
│   │   ├── transforms.py     # Data augmentation
│   │   └── dataloader.py     # DataLoader factory
│   ├── models/
│   │   ├── unet_baseline.py  # U-Net baseline [Maggie]
│   │   ├── siamese_unet.py   # Siamese U-Net [Laura]
│   │   ├── changeformer.py   # Transformer model [Alara]
│   │   └── factory.py        # Model factory
│   └── utils/
│       ├── metrics.py        # IoU, F1, Precision, Recall
│       ├── losses.py         # BCE, Dice, Focal losses
│       └── visualization.py  # Plotting utilities
├── scripts/
│   └── download_data.py      # Dataset download helper
├── checkpoints/              # Saved models
├── logs/                     # TensorBoard logs
├── results/                  # Evaluation results
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 👥 Task Division

### Equal Distribution: Each member handles ONE model end-to-end

---

### 🟦 Maggie Tu - U-Net Baseline

**Technical Work:**
- [ ] Implement `unet_baseline.py` - U-Net with concatenated inputs
- [ ] Test with different encoders (ResNet34, EfficientNet-B0)
- [ ] Implement U-Net++ variant as additional baseline

**Experiments:**
- [ ] Train baseline model for 100 epochs
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Ablation: Compare different encoders

**Documentation:**
- [ ] Write baseline results section in report
- [ ] Create baseline performance comparison table

---

### 🟩 Laura Li - Siamese U-Net (Main Model)

**Technical Work:**
- [ ] Implement `siamese_unet.py` - Shared encoder architecture
- [ ] Implement different feature comparison methods:
  - Subtraction: `|F_a - F_b|`
  - Concatenation: `[F_a, F_b]`
  - Concat + Diff: `[F_a, F_b, |F_a - F_b|]`
- [ ] Add attention mechanism variant

**Experiments:**
- [ ] Train Siamese U-Net for 100 epochs
- [ ] Compare feature comparison methods
- [ ] Experiment with different loss weights (BCE vs Dice)

**Documentation:**
- [ ] Write Siamese U-Net methodology section
- [ ] Create architecture diagram
- [ ] Analyze attention maps (if using attention variant)

---

### 🟨 Alara Kaymak - ChangeFormer (Transformer)

**Technical Work:**
- [ ] Implement `changeformer.py` - Transformer-based detection
- [ ] Implement cross-attention between time points
- [ ] Create lightweight variant for faster training

**Experiments:**
- [ ] Train ChangeFormer for 100 epochs
- [ ] Compare with CNN-based models
- [ ] Analyze transformer attention patterns

**Documentation:**
- [ ] Write transformer methodology section
- [ ] Visualize attention patterns
- [ ] Compare computational cost vs performance

---

### 🟪 Shared Tasks (Everyone)

| Task | Deadline | Owner |
|------|----------|-------|
| Dataset setup & verification | Week 1 | All |
| Training pipeline debugging | Week 1 | All |
| Model comparison experiments | Week 3 | All |
| Final presentation slides | Week 4 | All |
| Report writing | Week 4 | All |

---

## 📈 Expected Outcomes

1. **Trained Models:**
   - U-Net Baseline
   - Siamese U-Net (main)
   - ChangeFormer (stretch goal)

2. **Quantified Performance:**
   - IoU comparison across models
   - F1 Score comparison
   - Precision/Recall trade-offs

3. **Visualizations:**
   - Change detection examples
   - Model predictions vs ground truth
   - Training curves

4. **Stretch Goal:**
   - Compare with state-of-the-art change detection methods

---

## 📊 Results Template

| Model | IoU | F1 | Precision | Recall | Params |
|-------|-----|-------|-----------|--------|--------|
| U-Net Baseline | - | - | - | - | ~24M |
| Siamese U-Net | - | - | - | - | ~24M |
| ChangeFormer | - | - | - | - | ~15M |

---

## 🛠️ Development Guide

### Running TensorBoard

```bash
tensorboard --logdir logs/
```

### Testing Models

```bash
# Quick test with sample data
python scripts/download_data.py --create_sample --num_samples 5
python train.py --model siamese_unet --epochs 2 --batch_size 2
```

### GPU Memory Tips

- Use `--batch_size 4` for 8GB GPU
- Use `--batch_size 8` for 16GB GPU
- Use `--image_size 128` for limited memory

---

## 📚 References

1. **LEVIR-CD Dataset:** Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.

2. **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

3. **Siamese Networks for Change Detection:** Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection.

4. **segmentation_models.pytorch:** https://github.com/qubvel/segmentation_models.pytorch

---

## 📝 License

This project is for educational purposes as part of the Deep Learning course.

---

**Last Updated:** November 2024

