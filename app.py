import streamlit as st
import numpy as np
import pandas as pd
import colorsys
import cv2
import zipfile
import os
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from PIL import Image

# -----------------------------------------------------
# Streamlit Config
# -----------------------------------------------------
st.set_page_config(page_title="ChromoAI — Color Assistant", layout="wide")

ZIP_FALLBACK = "/mnt/data/color set.zip"
CSV_NAME = "color_names.csv"


# -----------------------------------------------------
# Dataset Loader
# -----------------------------------------------------
def ensure_dataset():
    if os.path.exists(CSV_NAME):
        return CSV_NAME

    if os.path.exists(ZIP_FALLBACK):
        try:
            with zipfile.ZipFile(ZIP_FALLBACK, 'r') as z:
                for member in z.namelist():
                    if member.lower().endswith('.csv'):
                        z.extract(member, '.')
                        return member
        except Exception as e:
            st.error(f"Failed to extract dataset from fallback: {e}")

    st.error("Dataset not found. Place color_names.csv in repo root.")
    return None


# -----------------------------------------------------
# Color Conversion & KDTree
# -----------------------------------------------------
def rgb_to_lab(rgb):
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return tuple(int(v) for v in lab)


def build_kdtree(df):
    df["rgb"] = df.apply(lambda x: (x["Red (8 bit)"], x["Green (8 bit)"], x["Blue (8 bit)"]), axis=1)
    df["lab"] = df["rgb"].apply(rgb_to_lab)
    pts = np.array(df["lab"].tolist())
    return KDTree(pts)


def find_nearest_color(rgb, df, tree):
    lab = np.array(rgb_to_lab(rgb)).reshape(1, -1)
    dist, idx = tree.query(lab, k=1)
    name = df.iloc[int(idx[0][0])]["Name"]
    return name, float(dist[0][0])


# -----------------------------------------------------
# Color Harmony
# -----------------------------------------------------
def rgb_to_hsl(rgb):
    r, g, b = [x / 255 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def shift_hue(h, deg):
    return (h + deg / 360.0) % 1.0


def generate_harmony(rgb, mode="analogous", spread=30):
    h, s, l = rgb_to_hsl(rgb)

    offsets = {
        "analogous": [-spread, 0, spread],
        "complementary": [0, 180],
        "triadic": [0, 120, -120],
        "tetradic": [0, 90, 180, 270],
        "split": [0, 150, -150]
    }.get(mode, [0])

    return [hsl_to_rgb(shift_hue(h, d), s, l) for d in offsets]


# -----------------------------------------------------
# Mood Classification
# -----------------------------------------------------
def mood_from_rgb(rgb):
    r, g, b = [x / 255 for x in rgb]
    h_deg = colorsys.rgb_to_hsv(r, g, b)[0] * 360
    s = colorsys.rgb_to_hsv(r, g, b)[1]

    if s < 0.15:
        return "Neutral"
    if 0 <= h_deg <= 60:
        return "Energetic"
    if 60 < h_deg <= 160:
        return "Fresh"
    if 160 < h_deg <= 260:
        return "Calm"
    return "Neutral"


# -----------------------------------------------------
# WCAG Contrast
# -----------------------------------------------------
def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * srgb_to_linear(r) + \
           0.7152 * srgb_to_linear(g) + \
           0.0722 * srgb_to_linear(b)


def contrast_ratio(fg, bg):
    L1, L2 = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (L1 + 0.05) / (L2 + 0.05)


# -----------------------------------------------------
# Color Blind Sim Simulation
# -----------------------------------------------------
def simulate_cvd(img_rgb, matrix):
    arr = img_rgb.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    sim = flat @ matrix.T
    sim = np.clip(sim, 0, 1)
    return (sim.reshape(h, w, 3) * 255).astype(np.uint8)


PROTAN = np.array([[0.567, 0.433, 0.000],
                   [0.558, 0.442, 0.000],
                   [0.000, 0.242, 0.758]])

DEUTAN = np.array([[0.625, 0.375, 0.000],
                   [0.700, 0.300, 0.000],
                   [0.000, 0.300, 0.700]])

TRITAN = np.array([[0.950, 0.050, 0.000],
                   [0.000, 0.433, 0.567],
                   [0.000, 0.475, 0.525]])


# -----------------------------------------------------
# Palette Extraction
# -----------------------------------------------------
def extract_palette(img_rgb, k=5):
    img = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


# -----------------------------------------------------
# AI Summary Generator
# -----------------------------------------------------
def compose_insight(name, rgb, mood, harmony, white_cr, black_cr, dist):
    tonal_map = {
        "Energetic": "punchy and attention-grabbing",
        "Fresh": "natural and revitalizing",
        "Calm": "soothing and stable",
        "Neutral": "balanced and understated"
    }

    mood_phrase = tonal_map.get(mood, mood.lower())
    hexval = "#{:02X}{:02X}{:02X}".format(*rgb)

    paragraphs = []
    paragraphs.append(f"**Assistant:** You selected **{name}** ({hexval}). Distance from reference: {dist:.1f}.")
    paragraphs.append(f"It reads as **{mood.lower()}** — {mood_phrase}.")
    palette_text = ", ".join([f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in harmony])
    paragraphs.append(f"Harmony palette generated: {palette_text}.")
    acc = []
    acc.append("white text passes" if white_cr >= 4.5 else "white text fails")
    acc.append("black text passes" if black_cr >= 4.5 else "black text fails")
    paragraphs.append("Accessibility: " + ", ".join(acc) + ".")
    return "\n\n".join(paragraphs)


# -----------------------------------------------------
# MAIN UI
# -----------------------------------------------------
def main():

    # Load CSS/JS from ROOT (not static folder)
    root_dir = os.path.dirname(__file__)

    style_path = os.path.join(root_dir, "style.css")
    script_path = os.path.join(root_dir, "script.js")

    if os.path.exists(style_path):
        st.markdown("<style>" + open(style_path).read() + "</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found in repo root")

    if os.path.exists(script_path):
        st.markdown("<script>" + open(script_path).read() + "</script>", unsafe_allow_html=True)
    else:
        st.warning("script.js not found in repo root")

    # Title
    st.header("🎨 ChromoAI — Generative Color Assistant")
    st.caption("Animated UI, chat-style insights, recommendations & accessibility intelligence.")

    # Load dataset
    csv_path = ensure_dataset()
    if not csv_path:
        st.stop()

    df = pd.read_csv(csv_path)
    tree = build_kdtree(df)

    # Layout
    left, right = st.columns([1, 2])

    # -------------------------------------------------
    # LEFT PANEL
    # -------------------------------------------------
    with left:
        st.markdown("### Pick a color")
        color_hex = st.color_picker("Choose a color", "#1976D2")
        rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
        st.write("RGB:", rgb)

        st.markdown("---")

        mode = st.selectbox("Harmony mode", ["analogous", "complementary", "triadic", "tetradic", "split"])

        st.markdown("### Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hello — I'm ChromoAI. Pick a color and press **Talk** to get insights."
            }]

        if st.button("Talk"):
            name, dist = find_nearest_color(rgb, df, tree)
            harmony = generate_harmony(rgb, mode)
            mood = mood_from_rgb(rgb)
            white_cr = contrast_ratio((255, 255, 255), rgb)
            black_cr = contrast_ratio((0, 0, 0), rgb)

            insight = compose_insight(name, rgb, mood, harmony, white_cr, black_cr, dist)

            st.session_state.messages.append({
                "role": "assistant",
                "content": insight
            })

        if st.button("Clear Chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Chat cleared. Pick a color and press **Talk**."
            }]

        st.markdown("---")

        st.markdown("### Upload image for palette extraction")
        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], key="u1")

        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            arr = np.array(pil)
            palette = extract_palette(arr)
            st.session_state.last_palette = palette

    # -------------------------------------------------
    # RIGHT PANEL
    # -------------------------------------------------
    with right:
        name, dist = find_nearest_color(rgb, df, tree)
        st.subheader(f"{name} — {color_hex}")
        st.markdown(f"Perceptual distance: **{dist:.2f}**")

        harmony = generate_harmony(rgb, mode)
        st.markdown("#### Harmony palette")

        cols = st.columns(len(harmony))
        for c, h in zip(cols, harmony):
            c.markdown(f"<div class='swatch' style='background:rgb{h};'></div>", unsafe_allow_html=True)
            c.caption(str(h))

        st.markdown("#### Mood")
        st.metric("Mood", mood_from_rgb(rgb))

        st.markdown("#### Accessibility")
        st.write(f"White contrast: **{contrast_ratio((255,255,255), rgb):.2f}**")
        st.write(f"Black contrast: **{contrast_ratio((0,0,0), rgb):.2f}**")

        if uploaded:
            st.markdown("#### Extracted Palette")
            pcols = st.columns(len(st.session_state.last_palette))
            for c, p in zip(pcols, st.session_state.last_palette):
                c.markdown(f"<div class='swatch' style='background:rgb{p};'></div>", unsafe_allow_html=True)
                c.caption(str(p))

    # Chat rendering
    st.markdown("<div class='chatbox'><div id='messages'></div></div>", unsafe_allow_html=True)

    for m in st.session_state.messages:
        content = m["content"]
        st.markdown(
            f"<div class='bubble assistant'>{content.replace(chr(10),'<br>')}</div>",
            unsafe_allow_html=True
        )

    # Simple Recommendations
    st.markdown("""
    <details>
    <summary><strong>Recommendations</strong></summary>
    <ul>
      <li>Use dark text when contrast_ratio > 4.5 vs black.</li>
      <li>Reserve white text for badges when white contrast > 4.5.</li>
      <li>For CTAs, use the palette's accent color and increase saturation slightly.</li>
    </ul>
    </details>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

