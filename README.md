# 🌿 Crop Doctor — AI-Powered Plant Disease Detection

> Upload a leaf photo. Get a diagnosis. That's it.

---

## What is this?

Crop Doctor is a web app that detects diseases in crop leaves using a deep learning model. You take a photo of a leaf, upload it, and the model tells you what's wrong (or if everything looks fine). It was built as part of our final year project to explore how computer vision can actually be useful in agriculture.

The app covers **14 crops** and **38 disease classes** — including healthy variants — trained on the well-known PlantVillage dataset.

---

## How we built the model

We didn't train from scratch. That would've taken forever and honestly wouldn't have been as good. Instead, we used **transfer learning** — which means we took a model that was already trained on millions of images (ImageNet) and fine-tuned it for our specific problem.

### The base model: EfficientNetB3

We went with **EfficientNetB3** from the EfficientNet family. The reason we picked this over something like VGG16 or ResNet50 is that EfficientNet scales depth, width, and resolution together in a principled way — so you get better accuracy without just throwing more parameters at the problem. B3 specifically hits a sweet spot between size and performance.

### Transfer learning approach

The process looked roughly like this:

1. **Load EfficientNetB3** pretrained on ImageNet, without the top classification layer
2. **Freeze the base** — we didn't want to mess with the features it already learned (edges, textures, shapes — all useful for leaves too)
3. **Add our own head** — GlobalAveragePooling → Dropout → Dense(38, softmax)
4. **Train phase 1** — only our new layers trained, base frozen

This two-phase approach is pretty standard for transfer learning and it worked well here. The idea is that ImageNet features are surprisingly good for plant leaves too — the model already knows how to detect textures, spots, and color patterns, which is basically what disease detection needs.

### Preprocessing

Images were resized to **300×300** (EfficientNetB3's native input size) and passed through `efficientnet.preprocess_input` which scales pixel values the way the model expects. We also used data augmentation during training — flips, rotations, zoom, brightness shifts — to make the model more robust to different lighting and angles.

---

## Dataset

**PlantVillage** — a publicly available dataset of ~87,000 leaf images across 38 classes (disease + healthy combinations). Images are lab-controlled close-ups of single leaves, which is why the app works best with similar photos.

| Stat | Value |
|------|-------|
| Total images | ~87,000 |
| Classes | 38 |
| Crops covered | 14 |
| Split | 80% train / 20% val |

---

## Results

The model hit **96.5% validation accuracy** after fine-tuning. Honestly better than we expected. Performance drops on real-world photos that look nothing like the training data — cluttered backgrounds, bad lighting, whole-plant shots — which is a known limitation of models trained on controlled datasets.

---

## Tech stack

| Layer | What we used |
|-------|-------------|
| Model | EfficientNetB3 (Keras / TensorFlow) |
| App | Streamlit |
| Image processing | PIL, NumPy |
| Training | Google Colab (T4 GPU) |

---

## Running it locally

```bash
git clone https://github.com/yourusername/crop-doctor
cd crop-doctor
pip install -r requirements.txt
streamlit run app.py
```

Make sure the `models/` folder has:
- `plant_disease_model.keras`
- `class_indices.json`
- `model_metadata.json`

---

## Project structure

```
crop-doctor/
├── app.py                  # Streamlit frontend
├── models/
│   ├── plant_disease_model.keras
│   ├── class_indices.json
│   └── model_metadata.json
├── requirements.txt
└── README.md
```

---

## Limitations

- Works best on close-up, single-leaf images (similar to how PlantVillage images look)
- Doesn't handle multiple diseases on one leaf well
- Real-field photos with soil/background lower confidence noticeably
- Only covers the 14 crops in the dataset — won't generalize to other plants

---

## Notes

This was a learning project. The model performs well on clean inputs but we're aware it's not production-ready for actual farmers without more diverse training data. The full methodology, experiments, and analysis are in the project report.

---
