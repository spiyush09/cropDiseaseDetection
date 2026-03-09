# 🌿 Crop Disease Detection System

An end-to-end Deep Learning application that detects diseases in crop leaves using Computer Vision and Transfer Learning.

> **Major Project — B.Tech, Parul University**

---

## Overview

This project uses **EfficientNetB3** (pretrained on ImageNet) to classify leaf images into **38 disease categories** across **14 crop species**. The model was trained on the **PlantVillage Dataset** (~87,000 images) using Google Colab and is served through a **Streamlit** web application for real-time inference.

**Achieved Validation Accuracy: 96.50%**

---

## Technology Stack

| Component | Technology |
|---|---|
| Deep Learning Model | EfficientNetB3 (Transfer Learning) |
| Framework | TensorFlow 2.15, Keras 3 |
| Data Processing | NumPy, Pillow, scikit-learn |
| Web Interface | Streamlit |
| Training Platform | Google Colab (T4 GPU) |
| Inference Platform | Local CPU |

---

## Project Structure

```
CropDiseaseDetection/
├── notebooks/
│   └── train_efficientnetb3.ipynb   # Full training notebook (Colab)
├── models/
│   ├── plant_disease_model.keras    # Trained model weights (not in repo — see below)
│   ├── class_indices.json           # Class ID → Disease name mapping
│   └── model_metadata.json          # Model configuration and accuracy info
├── documentation/
│   └── TECHNICAL_REPORT.md          # Detailed technical documentation
├── app.py                           # Streamlit web application
├── predict.py                       # Command-line prediction script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚠️ Model File

The trained model (`plant_disease_model.keras`, ~45MB) is **not included in this repository** due to file size limits.

To get the model:
- **Option A:** Train it yourself using `notebooks/train_efficientnetb3.ipynb` on Google Colab (~30–40 min with T4 GPU)
- **Option B:** Download from the project submission folder (shared separately)

Place the downloaded file at: `models/plant_disease_model.keras`

---

## How to Run

### 1. Clone the repository
```bash
git clone <repo-url>
cd CropDiseaseDetection
```

### 2. Set up environment (Anaconda recommended)
```bash
conda create -n cropenv python=3.10
conda activate cropenv
pip install -r requirements.txt
```

### 3. Add the model file
Place `plant_disease_model.keras` inside the `models/` folder.

### 4. Run the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### 5. (Optional) Run command-line prediction
```bash
python predict.py path/to/leaf_image.jpg
```

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | EfficientNetB3 |
| Input Size | 300 × 300 × 3 |
| Output Classes | 38 |
| Preprocessing | `efficientnet.preprocess_input` (scales to [-1, 1]) |
| Optimizer | Adam (lr=1e-3) |
| Best Val Accuracy | **96.50%** (Epoch 11/15) |
| Training Strategy | Transfer Learning — frozen base, custom head |

### Supported Crops & Diseases

| Crop | Diseases Covered |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper (Bell) | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Known Limitations

- The model was trained on controlled lab-style images from PlantVillage. Real-world field photos (busy backgrounds, multiple leaves, poor lighting) may reduce accuracy.
- Some visually similar diseases across different crops (e.g., Tomato Early Blight vs. Potato Early Blight) may be misclassified.
- Blueberry, Raspberry, and Soybean classes only have "healthy" — no disease data available in the dataset.

---

## License

This project is open-source and available for educational purposes.