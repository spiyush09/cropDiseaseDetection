# Technical Report — Crop Disease Detection System

**Project:** Crop Disease Detection using Deep Learning  
**Architecture:** EfficientNetB3 (Transfer Learning)  
**Institution:** Parul University  
**Platform:** Google Colab (Training) · Local CPU (Inference)

---

## 1. Problem Statement

Crop diseases cause significant agricultural losses worldwide. Early detection is critical but relies heavily on manual expert inspection — which is slow, expensive, and unavailable in remote areas. This project builds an automated system that can identify 38 disease conditions across 14 crop species from a single leaf photograph, making disease diagnosis accessible to anyone with a smartphone.

---

## 2. Dataset

**Source:** New Plant Diseases Dataset (PlantVillage) — Kaggle  
**Link:** `kaggle.com/datasets/vipoooool/new-plant-diseases-dataset`

| Property | Value |
|---|---|
| Total Images | ~87,000 |
| Training Images | ~70,295 |
| Validation Images | ~17,572 |
| Classes | 38 |
| Crop Species | 14 |
| Image Format | JPG, 256×256 (original), resized to 300×300 |

The dataset is organized into class-named folders and is pre-split into `train/` and `valid/` directories. All images are close-up shots of single leaves under controlled lighting conditions.

**Class Distribution:**  
Classes are reasonably balanced. Tomato has the highest representation (10 classes), while crops like Blueberry, Raspberry, and Soybean only have a "healthy" class — no disease data is available for those.

---

## 3. Model Architecture

### 3.1 Why EfficientNetB3?

EfficientNetB3 was selected for this project for three reasons:

- **Compound scaling:** EfficientNet scales network depth, width, and resolution together, achieving better accuracy at a given parameter budget compared to older architectures like VGG or ResNet.
- **Transfer Learning compatibility:** Pretrained on ImageNet (1.2M images, 1000 classes), the model already understands visual features like edges, textures, and shapes — which are directly useful for analysing leaf surface patterns.
- **Efficiency:** EfficientNetB3 achieves ~96% accuracy on this task while being far smaller than models like ResNet-50 or InceptionV3, making it suitable for local CPU inference.

### 3.2 Architecture Overview

```
Input (300×300×3)
    ↓
EfficientNetB3 Base (pretrained, frozen)
    ↓
GlobalAveragePooling2D          # Reduces feature map to a 1D vector
    ↓
BatchNormalization               # Stabilises training, speeds convergence
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.4)                     # Regularisation — prevents overfitting
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(38, Softmax)               # 38 output probabilities
```

**Total Parameters:** ~12.3M  
**Trainable Parameters (Phase 1):** ~1.1M (custom head only)

---

## 4. Training Strategy

### 4.1 Preprocessing

EfficientNet requires its own preprocessing function (`efficientnet.preprocess_input`) which scales pixel values from [0, 255] to [-1, 1]. Using standard `/255.0` normalization instead is a common mistake that measurably degrades accuracy.

### 4.2 Data Augmentation

Applied to training data only (not validation):

| Augmentation | Value | Reason |
|---|---|---|
| Rotation | ±30° | Leaves are photographed at various angles |
| Width/Height Shift | 20% | Simulate off-center framing |
| Shear | 20% | Perspective distortion |
| Zoom | 25% | Variable shooting distance |
| Horizontal Flip | Yes | Leaves can face either direction |
| Vertical Flip | No | Leaves don't grow upside down |
| Brightness Range | [0.8, 1.2] | Variable lighting conditions |

### 4.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Max Epochs | 15 |
| Early Stopping Patience | 5 epochs |
| LR Reduction Factor | 0.3 (on plateau, patience=3) |
| Minimum LR | 1e-7 |

### 4.4 Callbacks

- **EarlyStopping** — monitors `val_accuracy`, stops if no improvement for 5 epochs and restores best weights
- **ReduceLROnPlateau** — reduces learning rate by 0.3× when `val_loss` plateaus for 3 epochs
- **ModelCheckpoint** — saves `plant_disease_model.keras` whenever `val_accuracy` improves

---

## 5. Results

| Metric | Value |
|---|---|
| Best Validation Accuracy | **96.50%** |
| Best Epoch | 11 / 15 |
| Training stopped | Early (EarlyStopping triggered) |

The model converged cleanly with no significant overfitting, as indicated by training and validation accuracy curves tracking closely.

### 5.1 Per-Class Performance

Overall precision, recall, and F1-score are above 0.95 for most classes. Classes that share similar visual symptoms across crops (e.g., Tomato Early Blight and Potato Early Blight) show slightly lower precision due to inter-class visual similarity.

Full per-class breakdown is available in `classification_report.txt` (generated during training).

---

## 6. Inference Pipeline

The following steps are applied at inference time (in `app.py` and `predict.py`):

1. **Load image** — PIL opens the image and converts to RGB
2. **Resize** — image is resized to 300×300 pixels
3. **Convert to array** — PIL image → NumPy float32 array
4. **Add batch dimension** — shape becomes (1, 300, 300, 3)
5. **Preprocess** — `efficientnet.preprocess_input()` scales to [-1, 1]
6. **Predict** — model outputs 38 softmax probabilities
7. **Post-process** — top-3 predictions are extracted and labels formatted

**Confidence Threshold:** Predictions below 60% confidence trigger a warning, as they typically indicate poor image quality or an out-of-distribution input.

---

## 7. Web Application

The Streamlit app (`app.py`) provides:

- **Image upload** — supports JPG, PNG, WebP, BMP, TIFF, GIF, AVIF
- **Photo guidance** — tips for taking a good leaf photo
- **Real-time prediction** — result appears within seconds
- **Confidence display** — top-3 predictions with progress bars
- **Low confidence warning** — alerts user when result is unreliable
- **Model info sidebar** — architecture, accuracy, supported crops

---

## 8. Limitations

| Limitation | Explanation |
|---|---|
| Lab-trained dataset | PlantVillage images are taken in controlled conditions. Real-world field photos with cluttered backgrounds reduce accuracy. |
| Inter-class confusion | Tomato Early Blight and Potato Early Blight look almost identical visually — the model may occasionally confuse them. |
| Missing diseases | Many crop-disease combinations are absent from PlantVillage (e.g., most diseases of Blueberry, Raspberry, Soybean). |
| Single-leaf assumption | The model expects one leaf filling the frame. Group shots or whole-plant photos degrade performance significantly. |
| No treatment advice | The system identifies diseases but does not provide treatment recommendations. |

---

## 9. Future Improvements

- Fine-tune the top layers of EfficientNetB3 (Phase 2 training) for additional accuracy gain
- Collect and augment real-world field images to improve robustness
- Add treatment and pesticide recommendation database linked to detected disease
- Deploy as a mobile app for offline field use
- Add Grad-CAM visualisation to highlight which part of the leaf triggered the prediction

---

## 10. References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
2. Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
3. TensorFlow Documentation — `tf.keras.applications.EfficientNetB3`
4. Kaggle Dataset — New Plant Diseases Dataset: `vipoooool/new-plant-diseases-dataset`