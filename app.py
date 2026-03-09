import streamlit as st
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Crop Doctor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ─── Reset & Base ─── */
:root {
    --ink:       #0e0e0e;
    --paper:     #f5f0e8;
    --cream:     #ede8dc;
    --sage:      #4a6741;
    --sage-light:#6b8f62;
    --sage-pale: #d4e3d1;
    --rust:      #c0392b;
    --rust-pale: #f5d5d2;
    --gold:      #b8860b;
    --gold-pale: #f5eac8;
    --border:    #d4cfc5;
    --muted:     #8a8478;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--paper) !important;
    color: var(--ink);
}
.stApp { background: var(--paper) !important; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 860px !important;
    margin: 0 auto;
}

/* ─── Grain overlay ─── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.6;
}

/* ─── Masthead ─── */
.masthead {
    text-align: center;
    padding: 3.5rem 2.5rem 2.5rem;
    border-bottom: 2px solid var(--ink);
    position: relative;
    animation: fadeDown 0.7s ease both;
}
.masthead-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
}
.eyebrow-line { flex: 1; height: 1px; background: var(--border); max-width: 80px; }
.masthead h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(3.4rem, 8vw, 5.5rem) !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    letter-spacing: -2px !important;
    color: var(--ink) !important;
    margin: 0 0 0.5rem !important;
}
.masthead h1 em {
    font-style: italic;
    color: var(--sage);
}
.masthead-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.3px;
    margin-bottom: 1.8rem;
}
.stat-strip {
    display: flex;
    justify-content: center;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: white;
    max-width: 500px;
    margin: 0 auto;
}
.stat-item {
    flex: 1;
    padding: 0.7rem 0;
    border-right: 1px solid var(--border);
    text-align: center;
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--sage);
    line-height: 1;
    display: block;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.2rem;
    display: block;
}

/* ─── Section ─── */
.section {
    padding: 2rem 2.5rem;
    border-bottom: 1px solid var(--border);
    animation: fadeUp 0.6s ease both;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ─── Guide cards ─── */
.guide-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 0.9rem;
}
.guide-col {
    background: white;
    padding: 1.3rem 1.4rem;
}
.guide-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.22rem 0.6rem;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 1rem;
}
.tag-do   { background: var(--sage-pale); color: var(--sage); }
.tag-dont { background: var(--rust-pale); color: var(--rust); }
.guide-col ul { list-style: none; margin: 0; padding: 0; }
.guide-col li {
    font-size: 0.97rem;
    color: #555;
    font-weight: 300;
    padding: 0.32rem 0 0.32rem 1.2rem;
    border-bottom: 1px solid #f0ece4;
    position: relative;
    line-height: 1.4;
}
.guide-col li:last-child { border-bottom: none; }
.do-col   li::before { content: "↗"; position: absolute; left: 0; color: var(--sage); font-size: 0.75rem; }
.dont-col li::before { content: "×"; position: absolute; left: 0; color: var(--rust); font-size: 0.8rem; }
.guide-note {
    background: var(--gold-pale);
    border: 1px solid #e8d898;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    font-size: 0.8rem;
    color: #7a6020;
    line-height: 1.6;
    font-weight: 400;
}
.guide-note strong { color: var(--gold); font-weight: 500; }

/* ─── Upload zone ─── */
.upload-section { padding: 2rem 2.5rem; border-bottom: 1px solid var(--border); }
.stFileUploader > div {
    background: white !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stFileUploader > div:hover {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 4px var(--sage-pale) !important;
}
[data-testid="stFileUploaderDropzone"] { padding: 1.8rem !important; }
[data-testid="stFileUploaderDropzone"] p { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ─── Analyze button ─── */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    margin-top: 0.8rem !important;
    transition: background 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--sage) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─── Image display ─── */
.stImage > img {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* ─── Result section ─── */
.result-wrap { padding: 2rem 2.5rem; border-bottom: 1px solid var(--border); }

.result-verdict {
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s ease both;
}
.verdict-healthy {
    background: white;
    border: 2px solid var(--sage);
}
.verdict-disease {
    background: white;
    border: 2px solid var(--rust);
}
.verdict-warning {
    background: white;
    border: 2px solid var(--gold);
}
.verdict-healthy::before  { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--sage); }
.verdict-disease::before  { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--rust); }
.verdict-warning::before  { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--gold); }

.verdict-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.6rem;
    padding-left: 0.3rem;
}
.verdict-healthy  .verdict-status { color: var(--sage); }
.verdict-disease  .verdict-status { color: var(--rust); }
.verdict-warning  .verdict-status { color: var(--gold); }

.verdict-crop {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -1px;
    line-height: 1;
    padding-left: 0.3rem;
    margin-bottom: 0.2rem;
}
.verdict-disease-name {
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 300;
    padding-left: 0.3rem;
    margin-bottom: 0.9rem;
    font-style: italic;
}
.verdict-desc {
    font-size: 0.83rem;
    color: #666;
    font-weight: 300;
    padding-left: 0.3rem;
    margin-bottom: 1.2rem;
    line-height: 1.6;
}

/* ─── Confidence pill — dark chip, coloured text, always legible ─── */
.verdict-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    padding: 0.55rem 1.3rem 0.55rem 1.1rem;
    border-radius: 5px;
    font-weight: 500;
    margin-left: 0.3rem;
    letter-spacing: 0.8px;
    background: var(--ink);
    border: none;
}
.verdict-pill::before {
    content: '◈';
    font-size: 0.68rem;
    opacity: 0.7;
}
.verdict-healthy .verdict-pill  { background: #1e4d1a; color: #b6f0b0; }
.verdict-disease .verdict-pill  { background: #5a1a1a; color: #f5b0a8; }
.verdict-warning .verdict-pill  { background: #4a3800; color: #f5d878; }

/* ─── Top 3 ─── */
.top3-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
.top3-head {
    padding: 0.75rem 1.2rem;
    border-bottom: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--cream);
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid #f0ece4;
    transition: background 0.15s;
}
.pred-row:last-child { border-bottom: none; }
.pred-row:hover { background: #faf8f5; }
.pred-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--border);
    width: 1.4rem;
    flex-shrink: 0;
}
.pred-info { flex: 1; }
.pred-name { font-size: 0.92rem; color: #444; font-weight: 400; margin-bottom: 5px; }
.pred-bar  { height: 3px; background: var(--cream); border-radius: 2px; }
.pred-fill { height: 3px; border-radius: 2px; transition: width 0.8s ease; }
.fill-1    { background: var(--sage); }
.fill-2    { background: var(--sage-light); }
.fill-3    { background: var(--sage-pale); }
.pred-pct  {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--sage);
    min-width: 3.5rem;
    text-align: right;
}

/* ─── Footer ─── */
.app-footer {
    padding: 1.5rem 2.5rem;
    text-align: center;
}
.footer-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 2px solid var(--ink);
    padding-top: 1.2rem;
}
.footer-l {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--ink);
    font-style: italic;
}
.footer-r {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    text-align: right;
    line-height: 1.8;
}

/* ─── Animations ─── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH           = 'models/plant_disease_model.keras'
INDICES_PATH         = 'models/class_indices.json'
METADATA_PATH        = 'models/model_metadata.json'
IMG_SIZE             = (300, 300)
CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource
def load_model():
    try:
        return keras_load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

@st.cache_data
def load_class_indices():
    try:
        with open(INDICES_PATH) as f:
            return {v: k for k, v in json.load(f).items()}
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

model        = load_model()
idx_to_class = load_class_indices()
metadata     = load_metadata()
val_acc      = metadata.get('val_accuracy', 0.965)

# ── Masthead ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="masthead">
    <div class="masthead-eyebrow">
        <span class="eyebrow-line"></span>
        EfficientNetB3 · PlantVillage Dataset · 38 Classes
        <span class="eyebrow-line"></span>
    </div>
    <h1>Crop <em>Doctor</em></h1>
    <p class="masthead-sub">AI-powered leaf disease detection — upload a photo, get an instant diagnosis</p>
    <div class="stat-strip">
        <div class="stat-item">
            <span class="stat-num">{val_acc*100:.1f}%</span>
            <span class="stat-lbl">Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">38</span>
            <span class="stat-lbl">Classes</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">87K</span>
            <span class="stat-lbl">Images</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">14</span>
            <span class="stat-lbl">Crops</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Photo Guidelines ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section">
    <div class="section-label">📸 Photo Guidelines</div>
    <div class="guide-grid">
        <div class="guide-col do-col">
            <span class="guide-tag tag-do">↗ Do this</span>
            <ul>
                <li>Hold phone 20–30 cm from leaf</li>
                <li>One leaf fills most of the frame</li>
                <li>Shoot in natural daylight</li>
                <li>Keep phone steady, no blur</li>
                <li>Focus on the most affected area</li>
                <li>Use original camera, no filters</li>
            </ul>
        </div>
        <div class="guide-col dont-col">
            <span class="guide-tag tag-dont">× Avoid this</span>
            <ul>
                <li>Shooting whole plant from far away</li>
                <li>Shadows or glare on the leaf</li>
                <li>Dim indoor or artificial lighting</li>
                <li>Blurry or shaky photos</li>
                <li>Soil, pots, or many leaves in frame</li>
                <li>Filtered or edited images</li>
            </ul>
        </div>
    </div>
    <div class="guide-note">
        <strong>Important:</strong> This model was trained on close-up, single-leaf images from PlantVillage.
        The cleaner and closer your photo, the more accurate the diagnosis.
        Cluttered backgrounds reduce confidence significantly.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-section"><div class="section-label" style="font-family:\'DM Mono\',monospace;font-size:0.6rem;letter-spacing:3px;text-transform:uppercase;color:#8a8478;margin-bottom:1rem;display:flex;align-items:center;gap:0.7rem;">🔬 Upload Leaf Image<span style="flex:1;height:1px;background:#d4cfc5;display:block"></span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a leaf image here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif", "avif"],
    help="Upload a clear, well-lit photo of a single crop leaf",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col_img, _ = st.columns([1, 1])
    with col_img:
        st.image(img, caption="", use_container_width=True)
    analyze = st.button("ANALYZE LEAF", type="primary", use_container_width=True)
else:
    analyze = False

st.markdown('</div>', unsafe_allow_html=True)

# ── Analyze ────────────────────────────────────────────────────────────────────
if uploaded_file is not None and analyze:
    if model is None:
        st.error("Model not loaded. Please check the setup.")
    else:
        with st.spinner("Analyzing..."):
            img_array = preprocess_input(
                np.expand_dims(np.array(img.resize(IMG_SIZE), dtype=np.float32), 0)
            )
            preds      = model.predict(img_array, verbose=0)[0]
            top3_idx   = np.argsort(preds)[::-1][:3]
            confidence = float(preds[top3_idx[0]])
            label      = idx_to_class.get(top3_idx[0], "Unknown")
            is_healthy = "healthy" in label.lower()

            parts = label.split("___")
            if len(parts) == 2:
                crop_name    = parts[0].replace("_"," ").replace("(","").replace(")","").strip().title()
                disease_name = parts[1].replace("_"," ").title()
            else:
                crop_name    = label.replace("_"," ").title()
                disease_name = ""

        # ── Build verdict ──
        conf_str = f"{confidence*100:.1f}%"
        if confidence < CONFIDENCE_THRESHOLD:
            verdict = (
                '<div class="result-verdict verdict-warning">'
                '<span class="verdict-status">⚠ Low Confidence</span>'
                f'<div class="verdict-crop">{crop_name}</div>'
                f'<div class="verdict-disease-name">{disease_name or "Uncertain"}</div>'
                f'<div class="verdict-desc">Only {conf_str} confident — result may not be reliable. Retake closer, in brighter light, with just the leaf in frame.</div>'
                f'<span class="verdict-pill">{conf_str} confidence</span>'
                '</div>'
            )
        elif is_healthy:
            verdict = (
                '<div class="result-verdict verdict-healthy">'
                '<span class="verdict-status">✓ Diagnosis Complete</span>'
                f'<div class="verdict-crop">{crop_name}</div>'
                '<div class="verdict-disease-name">Healthy Leaf</div>'
                '<div class="verdict-desc">No disease detected. This leaf appears to be in good health.</div>'
                f'<span class="verdict-pill">{conf_str} confidence</span>'
                '</div>'
            )
        else:
            verdict = (
                '<div class="result-verdict verdict-disease">'
                '<span class="verdict-status">⚑ Disease Detected</span>'
                f'<div class="verdict-crop">{crop_name}</div>'
                f'<div class="verdict-disease-name">{disease_name}</div>'
                '<div class="verdict-desc">Disease identified. Consider consulting an agronomist for treatment options.</div>'
                f'<span class="verdict-pill">{conf_str} confidence</span>'
                '</div>'
            )

        # ── Build top-3 rows ──
        fill_classes = ["fill-1", "fill-2", "fill-3"]
        rows_html = ""
        for i, idx in enumerate(top3_idx):
            name  = idx_to_class.get(idx, "Unknown")
            parts = name.split("___")
            display = (f"{parts[0].replace('_',' ').title()} — {parts[1].replace('_',' ').title()}"
                       if len(parts) == 2 else name.replace("_", " ").title())
            prob = float(preds[idx])
            rows_html += (
                '<div class="pred-row">'
                f'<div class="pred-rank">0{i+1}</div>'
                '<div class="pred-info">'
                f'<div class="pred-name">{display}</div>'
                f'<div class="pred-bar"><div class="pred-fill {fill_classes[i]}" style="width:{int(prob*100)}%"></div></div>'
                '</div>'
                f'<div class="pred-pct">{prob*100:.1f}%</div>'
                '</div>'
            )

        # ── Single markdown call — no leaking ──
        section_label = (
            "<div class=\"section-label\" style=\"font-family:'DM Mono',monospace;font-size:0.6rem;"
            "letter-spacing:3px;text-transform:uppercase;color:#8a8478;margin-bottom:1rem;"
            "display:flex;align-items:center;gap:0.7rem;\">✦ Diagnosis Result"
            "<span style=\"flex:1;height:1px;background:#d4cfc5;display:block\"></span></div>"
        )
        top3_block = (
            '<div class="top3-card">'
            '<div class="top3-head">Top 3 Predictions</div>'
            + rows_html +
            '</div>'
        )
        st.markdown(
            '<div class="result-wrap">' + section_label + verdict + top3_block + '</div>',
            unsafe_allow_html=True
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-inner">
        <div class="footer-l">Crop Doctor</div>
        <div class="footer-r">
            EfficientNetB3 · PlantVillage · 2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)