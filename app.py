import streamlit as st
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
from PIL import Image

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Doctor 🌿",
    page_icon="🌿",
    layout="centered"
)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH        = 'models/plant_disease_model.keras'
INDICES_PATH      = 'models/class_indices.json'
METADATA_PATH     = 'models/model_metadata.json'
IMG_SIZE          = (300, 300)      # EfficientNetB3 native size
CONFIDENCE_THRESHOLD = 0.60        # Below this = warn user

# ─── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = keras_load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.info("Make sure 'models/plant_disease_model.keras' exists. Train it using the Colab notebook first.")
        return None

@st.cache_data
def load_class_indices():
    try:
        with open(INDICES_PATH) as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}   # flip to idx → name
    except Exception as e:
        st.error(f"❌ Could not load class indices: {e}")
        return {}

@st.cache_data
def load_metadata():
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

model       = load_model()
idx_to_class = load_class_indices()
metadata    = load_metadata()

# ─── UI Header ─────────────────────────────────────────────────────────────────
st.title("🌿 Crop Disease Detection System")
st.write("Upload a photo of a crop leaf to instantly detect diseases using AI.")

# Show model info in sidebar
with st.sidebar:
    st.header("ℹ️ Model Info")
    if metadata:
        st.write(f"**Architecture:** {metadata.get('architecture', 'EfficientNetB3')}")
        st.write(f"**Input Size:** {metadata.get('input_size', [300,300,3])[:2]}")
        st.write(f"**Classes:** {metadata.get('num_classes', 38)}")
        val_acc = metadata.get('val_accuracy', None)
        if val_acc:
            st.write(f"**Val Accuracy:** {val_acc*100:.2f}%")
    st.write(f"**Crops Covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")
    st.markdown("---")
    st.write("**How to use:**")
    st.write("1. Upload a clear photo of a single leaf")
    st.write("2. Make sure the leaf fills most of the image")
    st.write("3. Click 'Analyze Leaf'")

# ─── Photo Guide ───────────────────────────────────────────────────────────────
with st.expander("📸 How to take a good photo for best results", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Do this:**")
        st.markdown("""
- Hold phone **20–30 cm** from the leaf
- Make sure **one leaf fills** most of the frame
- Shoot in **natural daylight** or bright light
- Keep the phone **steady** (no blur)
- Focus on the **most affected** part of the leaf
        """)
    with col2:
        st.markdown("**❌ Avoid this:**")
        st.markdown("""
- Don't shoot the **whole plant** from far away
- Avoid **shadows** falling on the leaf
- Don't shoot in **dim indoor lighting**
- Avoid **blurry or shaky** photos
- Don't include **soil or pots** in the frame
        """)
    st.info("💡 **Tip:** The model was trained on close-up single-leaf images. The closer and cleaner your photo, the more accurate the result.")

# ─── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif", "avif"],
    help="Upload a clear, well-lit photo of a single crop leaf"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    if st.button("🔍 Analyze Leaf", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please check the setup.")
        else:
            with st.spinner("Analyzing leaf..."):
                # ── Preprocessing ───────────────────────────────────────────
                img_resized  = img.resize(IMG_SIZE)
                img_array    = np.array(img_resized, dtype=np.float32)
                img_array    = np.expand_dims(img_array, axis=0)
                img_array    = preprocess_input(img_array)   # ← EfficientNet specific

                # ── Prediction ──────────────────────────────────────────────
                preds       = model.predict(img_array, verbose=0)[0]
                top3_idx    = np.argsort(preds)[::-1][:3]
                confidence  = float(preds[top3_idx[0]])
                label       = idx_to_class.get(top3_idx[0], "Unknown")
                is_healthy  = "healthy" in label.lower()

                # ── Format label: "Tomato___Early_blight" → "Tomato — Early Blight"
                parts = label.split("___")
                if len(parts) == 2:
                    crop    = parts[0].replace("_", " ").replace("(", "").replace(")", "").title()
                    disease = parts[1].replace("_", " ").title()
                    formatted = f"{crop} — {disease}"
                else:
                    formatted = label.replace("_", " ").title()

                st.markdown("---")

                # ── Low confidence warning ───────────────────────────────────
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(f"⚠️ **Low Confidence ({confidence*100:.1f}%)** — Result may not be reliable.")
                    st.markdown("""
**Most likely reasons:**
- Leaf is too small in the frame — get closer
- Background is too busy (other leaves, soil visible)
- Poor lighting or blurry image

👆 Check the photo tips above and retake the photo.
                    """)
                else:
                    # ── Main result ─────────────────────────────────────────
                    if is_healthy:
                        st.success(f"✅ **{formatted}**")
                        st.write("This leaf appears **healthy**. No disease detected.")
                    else:
                        st.error(f"🔴 **{formatted}**")
                        st.write("Disease detected. Consider consulting an agronomist for treatment options.")

                    st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

                # ── Top 3 predictions ────────────────────────────────────────
                st.markdown("#### Top 3 Predictions")
                for i, idx in enumerate(top3_idx):
                    name = idx_to_class.get(idx, "Unknown")
                    parts = name.split("___")
                    if len(parts) == 2:
                        display = f"{parts[0].replace('_',' ').title()} — {parts[1].replace('_',' ').title()}"
                    else:
                        display = name.replace("_", " ").title()

                    prob = float(preds[idx])
                    rank_icon = ["🥇", "🥈", "🥉"][i]
                    st.progress(prob, text=f"{rank_icon} {display}: {prob*100:.1f}%")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with EfficientNetB3 + Transfer Learning • PlantVillage Dataset • 38 disease classes")