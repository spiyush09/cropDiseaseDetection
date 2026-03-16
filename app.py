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
    --sage:      #2d5a27;
    --sage-light:#3d7a35;
    --sage-pale: #d4e3d1;
    --rust:      #c0392b;
    --rust-pale: #f5d5d2;
    --gold:      #8a6200;
    --gold-pale: #f5eac8;
    --border:    #b8b2a8;
    --muted:     #3a3530;
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
    font-size: 0.72rem;
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
    font-size: 1.1rem;
    color: #2a2520;
    font-weight: 400;
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
    font-size: 0.7rem;
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
    font-size: 0.68rem;
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
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.22rem 0.6rem;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 1rem;
}
.tag-do   { background: var(--sage-pale); color: #1a3d17; }
.tag-dont { background: var(--rust-pale); color: #8a1a10; }
.guide-col ul { list-style: none; margin: 0; padding: 0; }
.guide-col li {
    font-size: 1.0rem;
    color: #1a1612;
    font-weight: 400;
    padding: 0.38rem 0 0.38rem 1.4rem;
    border-bottom: 1px solid #f0ece4;
    position: relative;
    line-height: 1.5;
}
.guide-col li:last-child { border-bottom: none; }
.do-col   li::before { content: "↗"; position: absolute; left: 0; color: var(--sage); font-size: 0.8rem; }
.dont-col li::before { content: "×"; position: absolute; left: 0; color: var(--rust); font-size: 0.85rem; }
.guide-note {
    background: var(--gold-pale);
    border: 1px solid #c8a830;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #5a3e00;
    line-height: 1.7;
    font-weight: 400;
}
.guide-note strong { color: #6a4a00; font-weight: 600; }

/* ─── Upload zone ─── */
.upload-section { padding: 2rem 2.5rem; border-bottom: 1px solid var(--border); }

/* Full-width centered drop zone — stack filename below */
[data-testid="stFileUploader"] {
    display: block !important;
    width: 100% !important;
}
/* Force the whole uploader wrapper to column layout so filename goes below */
[data-testid="stFileUploader"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.6rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    min-height: 200px !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    border: 2px dashed #2d5a27 !important;
    border-radius: 10px !important;
    background: #eaf3e8 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: none !important;
    cursor: pointer !important;
    padding: 2.5rem !important;
    box-sizing: border-box !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border: 3px dashed #1a3d17 !important;
    background: #dff0db !important;
    box-shadow: 0 0 0 5px rgba(45,90,39,0.12), 0 4px 18px rgba(45,90,39,0.1) !important;
}
/* All text inside dropzone — dark and readable */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #0e2a0c !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #0e2a0c !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
    font-size: 0.78rem !important;
    color: #1e4018 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.5px !important;
}
/* Cloud icon */
[data-testid="stFileUploaderDropzone"] svg {
    color: #2d5a27 !important;
    opacity: 1 !important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    background: white !important;
    color: #0e2a0c !important;
    border: 1.5px solid #2d5a27 !important;
    border-radius: 5px !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #2d5a27 !important;
    color: white !important;
    border-color: #1a3d17 !important;
    box-shadow: 0 2px 10px rgba(45,90,39,0.25) !important;
}

/* ─── Streamlit expander — fix arrow text bleed, bg, overflow ─── */
[data-testid="stExpander"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--paper) !important;
    margin-bottom: 0.9rem !important;
    overflow: hidden !important;
    position: relative !important;
    z-index: 0 !important;
    contain: layout !important;
}
/* Header summary row */
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] summary {
    background: var(--paper) !important;
    border-radius: 8px !important;
    list-style: none !important;
    cursor: pointer !important;
    padding: 0 !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary::marker { display: none !important; }

/* The actual header button Streamlit renders inside summary */
[data-testid="stExpander"] summary > div,
[data-testid="stExpander"] summary button,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
}
/* Hide the raw text node "arrow_right" / "arrow_drop_down" that leaks */
[data-testid="stExpander"] summary span[data-testid],
[data-testid="stExpander"] .eyJhbGciOiJIUzI1NiJ9,
[data-testid="stExpander"] summary > div > div:first-child > span:first-child {
    display: none !important;
}
/* Expander label text */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary label {
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    background: transparent !important;
}
[data-testid="stExpander"] svg {
    color: var(--ink) !important;
    flex-shrink: 0 !important;
}
/* Content area */
[data-testid="stExpander"] > details > div,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    overflow: visible !important;
    background: white !important;
    position: relative !important;
    z-index: 0 !important;
    padding-bottom: 1.4rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(45,90,39,0.08) !important;
}

/* ─── Uploaded filename — below dropzone, dark and visible ─── */
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
.stFileUploader [class*="uploadedFileName"],
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span:not([data-testid]) {
    color: #1a1612 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}
/* File info row — styled as a neat card below the dropzone */
[data-testid="stFileUploader"] > div > div:nth-child(2) {
    order: 2 !important;
    margin-top: 0 !important;
    padding: 0.6rem 1rem !important;
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: #1a1612 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
[data-testid="stFileUploader"] > div > div:nth-child(2) span,
[data-testid="stFileUploader"] > div > div:nth-child(2) p {
    color: #1a1612 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] > div > div:nth-child(2) svg {
    color: #2d5a27 !important;
}
/* ─── Spinner — dark text so it's visible on light bg ─── */
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span,
.stSpinner p,
.stSpinner span,
[data-testid="stStatusWidget"] span {
    color: #1a1612 !important;
    font-weight: 500 !important;
}
[data-testid="stSpinner"] svg,
.stSpinner svg {
    color: #2d5a27 !important;
    stroke: #2d5a27 !important;
}
.stButton > button {
    background: var(--ink) !important;
    color: #ffffff !important;
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
    color: #ffffff !important;
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
    font-size: 0.7rem;
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
    font-size: 1.05rem;
    color: #2a2520;
    font-weight: 400;
    padding-left: 0.3rem;
    margin-bottom: 0.9rem;
    font-style: italic;
}
.verdict-desc {
    font-size: 0.93rem;
    color: #1a1612;
    font-weight: 400;
    padding-left: 0.3rem;
    margin-bottom: 1.2rem;
    line-height: 1.7;
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
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--cream);
}
.pred-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.95rem 1.2rem;
    border-bottom: 1px solid #f0ece4;
    transition: background 0.15s;
}
.pred-row:last-child { border-bottom: none; }
.pred-row:hover { background: #faf8f5; }

/* FIX 5: pred-rank was too faint (#b8b2a8), now darker and more visible */
.pred-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #6b6560;
    width: 1.4rem;
    flex-shrink: 0;
}
.pred-info { flex: 1; }
.pred-name { font-size: 1.0rem; color: #0e0e0e; font-weight: 500; margin-bottom: 6px; }
.pred-bar  { height: 3px; background: var(--cream); border-radius: 2px; }
.pred-fill { height: 3px; border-radius: 2px; transition: width 0.8s ease; }
.fill-1    { background: var(--sage); }
.fill-2    { background: var(--sage-light); }
/* FIX 2: fill-3 was #d4e3d1 (sage-pale, nearly invisible on white) — now a visible mid-green */
.fill-3    { background: #8fbc8a; }
.pred-pct  {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
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
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
    font-style: italic;
}
.footer-r {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    text-align: right;
    line-height: 1.8;
}

/* ─── Diagnose a Leaf button ─── */
.diagnose-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: #0e0e0e;
    color: #f5f0e8 !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.75rem 1.8rem;
    border-radius: 5px;
    text-decoration: none !important;
    border: 1.5px solid #0e0e0e;
    transition: background 0.18s, border-color 0.18s, box-shadow 0.18s;
}
.diagnose-btn:hover {
    background: var(--sage) !important;
    color: #ffffff !important;
    border-color: var(--sage) !important;
    box-shadow: 0 4px 16px rgba(45,90,39,0.3) !important;
    text-decoration: none !important;
}
.diagnose-btn:visited,
.diagnose-btn:active,
.diagnose-btn:focus {
    color: #f5f0e8 !important;
    text-decoration: none !important;
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
    <div style="margin-top:1.8rem;">
        <a href="#upload-anchor" class="diagnose-btn">
            ↓ &nbsp;Diagnose a Leaf
        </a>
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

# ── Supported Crops ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section">
    <div class="section-label">🌱 Supported Crops</div>
    <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:0.9rem;">
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍎</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Apple</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🫐</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Blueberry</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍒</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Cherry</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🌽</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Corn</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍇</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Grape</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍊</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Orange</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍑</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Peach</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🌶️</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Bell Pepper</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🥔</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Potato</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍓</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Raspberry</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🫘</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Soybean</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🎃</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Squash</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍓</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Strawberry</span>
        </div>
        <div style="background:white;padding:1.1rem 0.5rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.45rem;text-align:center;">
            <span style="font-size:2rem;line-height:1">🍅</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#3a3530;">Tomato</span>
        </div>
    </div>
    <div class="guide-note">
        <strong>Heads up:</strong> Only upload leaves from the crops listed above.
        If you upload something else, the model will still output a result — but it'll be forced into one of these 14 categories and will almost certainly be wrong.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Dataset Classes Expander ───────────────────────────────────────────────────
_TH = "style=\"font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#3a3530;text-align:left;padding:0.6rem 0.9rem;border-bottom:2px solid #b8b2a8;background:#f5f0e8;\""
_TD = "style=\"padding:0.55rem 0.9rem;border-bottom:1px solid #c8c3b8;color:#1a1612;font-weight:400;vertical-align:middle;font-size:0.92rem;\""
_CR = "style=\"padding:0.55rem 0.9rem;border-bottom:1px solid #c8c3b8;font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;color:#2d5a27;font-weight:600;vertical-align:middle;\""
_TC = "style=\"padding:0.55rem 0.9rem;border-bottom:1px solid #c8c3b8;text-align:center;vertical-align:middle;\""
_BH = "style=\"display:inline-block;background:#d4e3d1;color:#1a3d17;font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:3px;\""
_BD = "style=\"display:inline-block;background:#f5d5d2;color:#8a1a10;font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:3px;\""

def _row(crop, condition, badge_style, badge_label):
    if crop:
        parts = crop.split(' ', 1)
        if len(parts) == 2:
            emoji, name = parts
            crop_html = f"<span style='font-size:1.3rem;line-height:1;margin-right:0.4rem;vertical-align:middle;'>{emoji}</span><span style='vertical-align:middle;'>{name}</span>"
        else:
            crop_html = crop
        cr = f"<td {_CR}>{crop_html}</td>"
    else:
        cr = f"<td {_CR}></td>"
    return f"<tr>{cr}<td {_TD}>{condition}</td><td {_TC}><span {badge_style}>{badge_label}</span></td></tr>"

def H(crop, cond): return _row(crop, cond, _BH, "Healthy")
def D(crop, cond): return _row(crop, cond, _BD, "Disease")

_rows = (
    D("🍎 Apple",      "Apple Scab")
  + D("",              "Black Rot")
  + D("",              "Cedar Apple Rust")
  + H("",              "Healthy")
  + H("🫐 Blueberry",  "Healthy")
  + D("🍒 Cherry",     "Powdery Mildew")
  + H("",              "Healthy")
  + D("🌽 Corn",       "Cercospora Leaf Spot / Gray Leaf Spot")
  + D("",              "Common Rust")
  + D("",              "Northern Leaf Blight")
  + H("",              "Healthy")
  + D("🍇 Grape",      "Black Rot")
  + D("",              "Esca (Black Measles)")
  + D("",              "Leaf Blight (Isariopsis Leaf Spot)")
  + H("",              "Healthy")
  + D("🍊 Orange",     "Huanglongbing (Citrus Greening)")
  + D("🍑 Peach",      "Bacterial Spot")
  + H("",              "Healthy")
  + D("🌶️ Bell Pepper","Bacterial Spot")
  + H("",              "Healthy")
  + D("🥔 Potato",     "Early Blight")
  + D("",              "Late Blight")
  + H("",              "Healthy")
  + H("🍓 Raspberry",  "Healthy")
  + H("🫘 Soybean",    "Healthy")
  + D("🎃 Squash",     "Powdery Mildew")
  + D("🍓 Strawberry", "Leaf Scorch")
  + H("",              "Healthy")
  + D("🍅 Tomato",     "Bacterial Spot")
  + D("",              "Early Blight")
  + D("",              "Late Blight")
  + D("",              "Leaf Mold")
  + D("",              "Septoria Leaf Spot")
  + D("",              "Spider Mites / Two-spotted Spider Mite")
  + D("",              "Target Spot")
  + D("",              "Tomato Yellow Leaf Curl Virus")
  + D("",              "Tomato Mosaic Virus")
  + H("",              "Healthy")
)

_expander_html = f"""
<div style="border:1px solid #b8b2a8;border-radius:8px;overflow:hidden;margin-bottom:0.9rem;">
<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
<thead><tr>
  <th {_TH} style="width:22%;font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3a3530;text-align:left;padding:0.5rem 0.8rem;border-bottom:2px solid #b8b2a8;background:#f5f0e8;">Crop</th>
  <th {_TH} style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3a3530;text-align:left;padding:0.5rem 0.8rem;border-bottom:2px solid #b8b2a8;background:#f5f0e8;">Detectable Condition</th>
  <th {_TH} style="width:12%;text-align:center;font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3a3530;padding:0.5rem 0.8rem;border-bottom:2px solid #b8b2a8;background:#f5f0e8;">Type</th>
</tr></thead>
<tbody>{_rows}</tbody>
</table>
</div>
<div style="background:#f5eac8;border:1px solid #c8a830;border-radius:6px;padding:0.8rem 1.1rem;font-size:0.8rem;color:#5a3e00;line-height:1.6;">
  <strong style="color:#6a4a00;">Note:</strong>
  Orange has <strong>no healthy class</strong> — a healthy orange leaf will always return a disease result.
  Blueberry, Raspberry, and Soybean have <strong>no disease classes</strong> — they can only be identified as healthy.
  Squash has no healthy class either, only Powdery Mildew.
</div>
"""

with st.expander("📋  What can this model predict? — See all 38 classes"):
    st.markdown(_expander_html, unsafe_allow_html=True)

# FIX 4: Replaced fragile setTimeout JS with a MutationObserver so the arrow
# text fix survives Streamlit re-renders triggered by user interaction
st.markdown("""
<script>
(function fixExpander() {
    function clean() {
        document.querySelectorAll('[data-testid="stExpander"] summary').forEach(function(summary) {
            summary.childNodes.forEach(function(node) {
                if (node.nodeType === 3 && node.textContent.trim().length > 0) {
                    node.textContent = '';
                }
            });
        });
    }
    clean();
    var observer = new MutationObserver(function() { clean(); });
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-section" id="upload-anchor">
    <div style="text-align:center;margin-bottom:1.6rem;">
        <div style="font-family:'Playfair Display',serif;font-size:2.6rem;font-weight:900;color:#0e0e0e;letter-spacing:-0.5px;margin-bottom:0.5rem;">Upload a <em style="color:#2d5a27;font-style:italic;">Leaf</em> Photo</div>
        <div style="font-size:1.05rem;color:#1a1612;font-weight:400;letter-spacing:0.2px;">Close-up · Single leaf · Good lighting · No filters</div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop image here",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif", "avif"],
    help="Upload a clear, well-lit photo of a single crop leaf",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
    # FIX 3: was st.columns([1,1]) which put image in left half only.
    # Now uses full width so image is properly centered and larger on all screen sizes.
    st.image(img, caption="", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
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
            label      = idx_to_class.get(int(top3_idx[0]), "Unknown")
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
            # FIX 1: cast to int() so dict lookup never silently returns "Unknown"
            # on numpy int64 vs Python int mismatch across library versions
            name  = idx_to_class.get(int(idx), "Unknown")
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
            "letter-spacing:3px;text-transform:uppercase;color:#3a3530;margin-bottom:1rem;"
            "display:flex;align-items:center;gap:0.7rem;\">✦ Diagnosis Result"
            "<span style=\"flex:1;height:1px;background:#b8b2a8;display:block\"></span></div>"
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

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 2.5rem;border-top:1px solid #b8b2a8;">
    <div style="background:#fff8e6;border:1.5px solid #e8c84a;border-left:5px solid #8a6200;border-radius:6px;padding:1.2rem 1.4rem;line-height:1.8;">
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#8a6200;display:block;margin-bottom:0.5rem;font-weight:600;">⚠ Disclaimer</span>
        <span style="font-size:0.88rem;color:#2a2520;">
        This is a <strong style="color:#0e0e0e;">student research project</strong> built for learning purposes — it is not a professional diagnostic tool.
        The model performs well under controlled conditions but real-world accuracy can vary significantly.
        If your crop shows signs of disease, please <strong style="color:#0e0e0e;">consult a qualified agronomist or agricultural specialist</strong> before taking any action.
        Do not make farming or treatment decisions based solely on this tool.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

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