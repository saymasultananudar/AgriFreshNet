import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LABELS =================
fruit_classes = ["Bittermelon", "Pineapple", "Papaya", "Cucumber",
                 "Tomato", "Eggplant", "Orange", "Banana"]
freshness_classes = ["Fresh", "Semi-Fresh", "Rotten"]

# ================= SHELF LIFE DATABASE =================
shelf_life_db = {
    ("Fresh", "Banana"): 3,
    ("Fresh", "Bittermelon"): 2,
    ("Fresh", "Cucumber"): 5,
    ("Fresh", "Eggplant"): 3,
    ("Fresh", "Orange"): 8,
    ("Fresh", "Papaya"): 3,
    ("Fresh", "Pineapple"): 7,
    ("Fresh", "Tomato"): 9,

    ("Rotten", "Banana"): 1,
    ("Rotten", "Bittermelon"): 3,
    ("Rotten", "Cucumber"): 3,
    ("Rotten", "Eggplant"): 2,
    ("Rotten", "Orange"): 3,
    ("Rotten", "Papaya"): 2,
    ("Rotten", "Pineapple"): 4,
    ("Rotten", "Tomato"): 1,

    ("Semi-Fresh", "Banana"): 3,
    ("Semi-Fresh", "Bittermelon"): 2,
    ("Semi-Fresh", "Cucumber"): 4,
    ("Semi-Fresh", "Eggplant"): 4,
    ("Semi-Fresh", "Orange"): 5,
    ("Semi-Fresh", "Papaya"): 3,
    ("Semi-Fresh", "Pineapple"): 10,
    ("Semi-Fresh", "Tomato"): 6,
}

def get_shelf_life(freshness, fruit):
    return shelf_life_db.get((freshness, fruit), None)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= LOAD MODEL =================
def load_model(path, num_classes):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

fruit_model = load_model("EfficientNet_fruit.pth", len(fruit_classes))
fresh_model = load_model("EfficientNet_freshness.pth", len(freshness_classes))

# ================= GRAD-CAM =================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts  = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

# ================= PREDICT =================
def predict(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        fruit_out = fruit_model(img)
        fresh_out = fresh_model(img)
        fruit_prob = torch.softmax(fruit_out, dim=1)
        fresh_prob = torch.softmax(fresh_out, dim=1)

    f_idx  = torch.argmax(fruit_prob).item()
    fr_idx = torch.argmax(fresh_prob).item()

    return (
        fruit_classes[f_idx],      fruit_prob[0][f_idx].item(),
        freshness_classes[fr_idx], fresh_prob[0][fr_idx].item(),
        img
    )

# ================= HELPERS =================
def freshness_color(label):
    return {"Fresh": "#4ade80", "Semi-Fresh": "#c084fc", "Rotten": "#f87171"}.get(label, "#ffffff")

def freshness_icon(label):
    return {"Fresh": "✅", "Semi-Fresh": "⚠️", "Rotten": "❌"}.get(label, "❓")

def shelf_badge_color(freshness):
    return {"Fresh": "#11B951", "Semi-Fresh": "#d5e02e", "Rotten": "#7f1d1d"}.get(freshness, "#1e1b2e")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AgriFreshNET", layout="wide", page_icon="🌿")

# ================= CSS =================
st.markdown("""
<style>
/* ---- ALL FONTS → Times New Roman ---- */
html, body, [class*="css"], p, div, span, label, button,
h1, h2, h3, h4, h5, input, textarea, select, option {
    font-family: 'Times New Roman', Times, Georgia, serif !important;
}

/* ---- APP BACKGROUND ---- */
.stApp {
    background:
        radial-gradient(ellipse at 15% 5%,  #3b0764aa 0%, transparent 50%),
        radial-gradient(ellipse at 85% 95%, #14532d88 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, #1a0a2e   0%, transparent 70%),
        #0b0914;
    min-height: 100vh;
}

/* ---- STREAMLIT CHROME ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }

/* ---- HEADER ---- */
.agri-header {
    text-align: center;
    padding: 2rem 1rem 0.4rem;
}
.agri-header h1 {
    font-size: 3.5rem;
    font-weight: bold;
    font-style: italic;
    background: linear-gradient(130deg, #a855f7 0%, #4ade80 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 3px;
    line-height: 1.1;
}
.agri-header p {
    color: #6b7280;
    font-size: 0.88rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 0.45rem;
    font-style: italic;
}
.agri-divider {
    width: 90px;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #22c55e, #a855f7);
    border-radius: 2px;
    margin: 0.65rem auto 1.8rem;
}

/* ---- CARDS ---- */
.upload-card, .result-panel {
    background: #110e1f99;
    border: 1px solid #2e1065;
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 0 50px #7c3aed18, 0 4px 30px #00000055;
    backdrop-filter: blur(12px);
}

/* ---- SECTION LABELS ---- */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    font-style: italic;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.section-label.purple { color: #a855f7; }
.section-label.green  { color: #4ade80; }

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] {
    background: #0b091499;
    border: 1.5px dashed #3b0764;
    border-radius: 12px;
    padding: 0.4rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: #a855f7; }

/* ---- IMAGES ---- */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #2e1065;
    object-fit: cover;
}

/* ---- BUTTON ---- */
.stButton > button {
    background: linear-gradient(135deg, #2e1065, #14532d) !important;
    color: #e9d5ff !important;
    border: 1px solid #7c3aed !important;
    border-radius: 10px !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1rem !important;
    font-style: italic !important;
    font-weight: bold !important;
    letter-spacing: 0.1em !important;
    padding: 0.55rem 1.2rem !important;
    width: 100%;
    transition: all 0.25s !important;
    box-shadow: 0 0 20px #7c3aed33 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4c1d95, #166534) !important;
    box-shadow: 0 0 35px #a855f755, 0 0 15px #22c55e33 !important;
    transform: translateY(-1px) !important;
    color: #bbf7d0 !important;
}

/* ---- PREDICTION CARDS ---- */
.pred-row { display: flex; gap: 0.75rem; margin-bottom: 0.85rem; }
.pred-card {
    flex: 1;
    background: #0b091499;
    border-radius: 14px;
    padding: 0.9rem 1rem 0.8rem;
    border: 1px solid #2e1065;
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.pred-card.fruit::before  { background: linear-gradient(90deg, #a855f7, #7c3aed); }
.pred-card.fresh::before  { background: linear-gradient(90deg, #22c55e, #15803d); }
.pred-label {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #6b7280;
    margin-bottom: 0.28rem;
    font-style: italic;
}
.pred-value {
    font-size: 1.3rem;
    font-weight: bold;
    font-style: italic;
    margin-bottom: 0.22rem;
}
.pred-conf { font-size: 0.7rem; color: #9ca3af; font-style: italic; }
.conf-bar-bg {
    background: #1e1b2e;
    border-radius: 99px;
    height: 4px;
    margin-top: 0.45rem;
    overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 99px; }

/* ---- SHELF LIFE CARD ---- */
.shelf-card {
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border: 1px solid;
}
.shelf-big-icon { font-size: 2.2rem; }
.shelf-text { flex: 1; }
.shelf-heading {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: #9ca3af;
    margin-bottom: 0.15rem;
    font-style: italic;
}
.shelf-days {
    font-size: 1.75rem;
    font-weight: bold;
    font-style: italic;
    line-height: 1.1;
}
.shelf-sub {
    font-size: 0.72rem;
    color: #9ca3af;
    margin-top: 0.15rem;
    font-style: italic;
}

/* ---- IMG LABELS ---- */
.img-label {
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b7280;
    text-align: center;
    margin-bottom: 4px;
    font-style: italic;
}

/* ---- EMPTY STATE ---- */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 1rem;
    gap: 0.75rem;
    color: #374151;
}
.empty-state .eicon { font-size: 2.8rem; }
.empty-state p {
    font-size: 0.9rem;
    text-align: center;
    font-style: italic;
    line-height: 1.65;
    color: #4b5563;
}

/* ---- SPINNER ---- */
[data-testid="stSpinner"] { color: #a855f7 !important; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="agri-header">
    <h1>🌿 AgriFreshNET</h1>
    <p>AI-Powered Fruit &amp; Freshness Detection</p>
    <div class="agri-divider"></div>
</div>
""", unsafe_allow_html=True)

# ================= LAYOUT =================
col1, col2 = st.columns([1, 1.7], gap="large")

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label purple">📤 &nbsp;Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
        analyze = st.button("🔬 Run Analysis")
    else:
        analyze = False

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label green">📊 &nbsp;Detection Results</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <span class="eicon">🍃</span>
            <p>Upload a fruit image on the left<br>to see AI predictions here.</p>
        </div>""", unsafe_allow_html=True)

    elif not analyze:
        st.markdown("""
        <div class="empty-state">
            <span class="eicon">🔍</span>
            <p>Click <strong style="color:#a855f7">Run Analysis</strong><br>to detect fruit &amp; freshness.</p>
        </div>""", unsafe_allow_html=True)

    else:
        with st.spinner("Analysing image..."):
            fruit, f_conf, fresh, fr_conf, img_tensor = predict(image)
            cam        = generate_gradcam(fruit_model, img_tensor)
            img_np     = np.array(image.resize((224, 224)))
            heatmap    = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            superimposed = np.clip(heatmap * 0.4 + img_np, 0, 255).astype(np.uint8)

        f_color     = freshness_color(fresh)
        f_icon      = freshness_icon(fresh)
        f_conf_pct  = f_conf  * 100
        fr_conf_pct = fr_conf * 100

        # ---- Prediction cards ----
        st.markdown(f"""
        <div class="pred-row">
            <div class="pred-card fruit">
                <div class="pred-label">🍌 Fruit Type</div>
                <div class="pred-value" style="color:#c084fc">{fruit}</div>
                <div class="pred-conf">{f_conf_pct:.1f}% confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{f_conf_pct:.1f}%;
                         background:linear-gradient(90deg,#a855f7,#7c3aed);"></div>
                </div>
            </div>
            <div class="pred-card fresh">
                <div class="pred-label">{f_icon} Freshness</div>
                <div class="pred-value" style="color:{f_color}">{fresh}</div>
                <div class="pred-conf">{fr_conf_pct:.1f}% confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{fr_conf_pct:.1f}%;
                         background:linear-gradient(90deg,{f_color}88,{f_color});"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Shelf Life card ----
        shelf = get_shelf_life(fresh, fruit)

        if shelf is not None:
            bg_col = shelf_badge_color(fresh)
            st.markdown(f"""
            <div class="shelf-card" style="background:{bg_col}55; border-color:{f_color}44;">
                <div class="shelf-big-icon">🕰️</div>
                <div class="shelf-text">
                    <div class="shelf-heading">Estimated Shelf Life</div>
                    <div class="shelf-days" style="color:{f_color}">{shelf} days</div>
                    <div class="shelf-sub">{fresh} &nbsp;·&nbsp; {fruit} &nbsp;·&nbsp; Based on detection result</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="shelf-card" style="background:#1e1b2e; border-color:#374151;">
                <div class="shelf-big-icon">❓</div>
                <div class="shelf-text">
                    <div class="shelf-heading">Estimated Shelf Life</div>
                    <div class="shelf-days" style="color:#6b7280">Not Available</div>
                    <div class="shelf-sub">No shelf-life data for this combination</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Images: Original + Grad-CAM ----
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
            st.image(img_np, use_column_width=True)
        with ic2:
            st.markdown('<div class="img-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(superimposed, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
