"""
Command-line prediction script for the Crop Disease Detection model.
Usage: python predict.py path/to/leaf_image.jpg
"""

import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input  # fixed: consistent import
from PIL import Image

MODEL_PATH   = 'models/plant_disease_model.keras'
INDICES_PATH = 'models/class_indices.json'
IMG_SIZE     = (300, 300)   # EfficientNetB3 native size

def load_resources():
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure plant_disease_model.keras is in the models/ folder.")
        return None, None

    try:
        with open(INDICES_PATH) as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"✅ Class indices loaded: {len(idx_to_class)} classes")
    except Exception as e:
        print(f"❌ Error loading class indices: {e}")
        return model, None

    return model, idx_to_class

def predict(model, idx_to_class, img_path, top_k=3):
    # Load and preprocess
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet-specific preprocessing

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]

    results = []
    for idx in top_indices:
        label = idx_to_class.get(int(idx), "Unknown")
        parts = label.split("___")
        if len(parts) == 2:
            formatted = f"{parts[0].replace('_', ' ')} — {parts[1].replace('_', ' ')}"
        else:
            formatted = label.replace("_", " ")
        results.append((formatted, float(preds[idx])))

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py leaf.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    model, idx_to_class = load_resources()

    if model is None or idx_to_class is None:
        sys.exit(1)

    print(f"\nAnalyzing: {img_path}")
    print("-" * 40)

    results = predict(model, idx_to_class, img_path)

    for i, (label, confidence) in enumerate(results):
        rank = ["1st", "2nd", "3rd"][i]
        bar = "█" * int(confidence * 30)
        print(f"  {rank}: {label}")
        print(f"       {bar} {confidence*100:.2f}%\n")

    top_label, top_conf = results[0]
    is_healthy = "healthy" in top_label.lower()

    print("-" * 40)
    if top_conf < 0.60:
        print("⚠️  Low confidence — result may be unreliable.")
    elif is_healthy:
        print(f"✅ Leaf appears HEALTHY ({top_conf*100:.1f}% confidence)")
    else:
        print(f"🔴 Disease detected: {top_label} ({top_conf*100:.1f}% confidence)")