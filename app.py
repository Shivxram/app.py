import streamlit as st
import numpy as np
import pandas as pd
import cv2
import colorsys
import cv2
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from PIL import Image

# ---------------------------
# Load your dataset
# ---------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("color_names.csv")
    df['rgb'] = df.apply(lambda x: (x['Red (8 bit)'], x['Green (8 bit)'], x['Blue (8 bit)']), axis=1)
    df['lab'] = df['rgb'].apply(rgb_to_lab)
    tree = KDTree(np.array(df['lab'].tolist()))
    return df, tree

# ---------------------------
# Helper functions
# ---------------------------
def rgb_to_lab(rgb):
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0,0]
    return tuple(int(v) for v in lab)

def find_nearest_color(rgb, df, tree):
    lab = np.array(rgb_to_lab(rgb)).reshape(1, -1)
    dist, idx = tree.query(lab, k=1)
    name = df.iloc[idx[0][0]]['Name']
    return name, float(dist[0][0])

def rgb_to_hsl(rgb):
    r, g, b = [x/255 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l

def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

def shift_hue(h, degrees):
    return (h + degrees/360.0) % 1.0

def generate_harmony(rgb, mode="analogous", spread=30):
    h, s, l = rgb_to_hsl(rgb)
    result = []
    if mode == "analogous":
        offsets = [-spread, 0, spread]
    elif mode == "complementary":
        offsets = [0, 180]
    elif mode == "triadic":
        offsets = [0, 120, -120]
    elif mode == "tetradic":
        offsets = [0, 90, 180, 270]
    else:
        offsets = [0]
    for deg in offsets:
        result.append(hsl_to_rgb(shift_hue(h, deg), s, l))
    return result

def mood_from_rgb(rgb):
    r, g, b = [x/255 for x in rgb]
    h, l, s = colorsys.hls_to_rgb(r, g, b)
    h_deg = (colorsys.rgb_to_hsv(r,g,b)[0]) * 360
    if s < 0.15:
        return "Neutral"
    if 0 <= h_deg <= 60:
        return "Energetic"
    if 60 < h_deg <= 160:
        return "Fresh"
    if 160 < h_deg <= 260:
        return "Calm"
    return "Neutral"

def srgb_to_linear(c):
    c = c/255
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

def luminance(rgb):
    r, g, b = rgb
    return (
        0.2126 * srgb_to_linear(r) +
        0.7152 * srgb_to_linear(g) +
        0.0722 * srgb_to_linear(b)
    )

def contrast_ratio(fg, bg):
    L1, L2 = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (L1 + 0.05) / (L2 + 0.05)

def simulate_cvd(img_rgb, matrix):
    arr = img_rgb.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    sim = flat @ matrix.T
    sim = np.clip(sim, 0, 1)
    return (sim.reshape(h, w, 3) * 255).astype(np.uint8)

def extract_palette(img_rgb, k=5):
    img = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]

# CVD matrices
PROTAN = np.array([[0.567, 0.433, 0.000],
                   [0.558, 0.442, 0.000],
                   [0.000, 0.242, 0.758]])

DEUTAN = np.array([[0.625, 0.375, 0.000],
                   [0.700, 0.300, 0.000],
                   [0.000, 0.300, 0.700]])

TRITAN = np.array([[0.950, 0.050, 0.000],
                   [0.000, 0.433, 0.567],
                   [0.000, 0.475, 0.525]])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 Color Intelligence AI")
st.write("Real-time color analysis, harmony, mood, accessibility, palettes, and color-blind simulations.")

df, tree = load_dataset()

# --- Color Picker ---
color_hex = st.color_picker("Pick a color", "#78C850")
rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))

st.subheader("Selected Color")
st.write(f"RGB: {rgb}")

# Nearest Color Name
name, dist = find_nearest_color(rgb, df, tree)
st.metric("Nearest Named Color", name)

# Harmony
st.subheader("Color Harmonies")
modes = ["analogous", "complementary", "triadic", "tetradic"]
mode = st.selectbox("Harmony mode", modes)
harmonies = generate_harmony(rgb, mode)

cols = st.columns(len(harmonies))
for c, h in zip(cols, harmonies):
    c.markdown(
        f"<div style='width:80px;height:80px;background:rgb{h};border-radius:8px;'></div>",
        unsafe_allow_html=True
    )
    c.caption(str(h))

# Mood
st.subheader("Color Mood")
st.success(mood_from_rgb(rgb))

# WCAG
white = contrast_ratio((255,255,255), rgb)
black = contrast_ratio((0,0,0), rgb)

st.subheader("Accessibility (WCAG Contrast)")
st.write(f"White text contrast: {white:.2f}")
st.write(f"Black text contrast: {black:.2f}")

# CVD Simulation
st.subheader("Color Blind Simulation")
img = np.zeros((120,120,3), dtype=np.uint8)
img[:] = rgb

pro = simulate_cvd(img, PROTAN)
deu = simulate_cvd(img, DEUTAN)
tri = simulate_cvd(img, TRITAN)

st.image([img, pro, deu, tri], caption=["Original", "Protanopia", "Deuteranopia", "Tritanopia"])

# Image Upload Palette Extraction
st.subheader("Palette Extraction from Image")
uploaded = st.file_uploader("Upload an image", type=["jpg","png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(pil_img)
    st.image(pil_img, caption="Uploaded Image")

    palette = extract_palette(img_rgb, k=5)
    st.write("Dominant Colors:")

    cols = st.columns(len(palette))
    for c, p in zip(cols, palette):
        c.markdown(
            f"<div style='width:80px;height:80px;background:rgb{p};border-radius:8px;'></div>",
            unsafe_allow_html=True
        )
        c.caption(str(p))
