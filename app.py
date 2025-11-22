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
st.set_page_config(page_title="ChromoAI — Color Chatbot", layout="wide")

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
# Color conversion / KDTree
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
# Harmony generator
# -----------------------------------------------------
def rgb_to_hsl(rgb):
    r, g, b = [x / 255 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, s, l


def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def shift_hue(h, deg):
    return (h + deg / 360.0) % 1.0


def generate_harmony(rgb, mode="analogous", spread=30):
    h, s, l = rgb_to_hsl(rgb)

    offsets = {
        "analogous": [-spread, 0, spread],
        "complementary": [0, 180],
        "triadic": [0, 120, -120],
        "tetradic": [0, 90, 180, 270],
        "split": [0, 150, -150],
    }.get(mode, [0])

    return [hsl_to_rgb(shift_hue(h, d), s, l) for d in offsets]


# -----------------------------------------------------
# Mood engine (expanded)
# -----------------------------------------------------
def mood_from_rgb(rgb):
    r, g, b = [x / 255 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_deg = h * 360

    if s < 0.08 and v > 0.85:
        return "Minimal"
    if v < 0.2:
        return "Moody"
    if s > 0.7 and 0 <= h_deg < 60:
        return "Energetic"
    if s > 0.6 and 60 <= h_deg < 150:
        return "Fresh"
    if s > 0.5 and 150 <= h_deg < 260:
        return "Calm"
    if s > 0.6 and 260 <= h_deg < 330:
        return "Playful"
    if s < 0.35 and 0.25 < v < 0.8:
        return "Elegant"
    if s < 0.25 and v > 0.8:
        return "Soft"
    return "Balanced"


# -----------------------------------------------------
# WCAG contrast
# -----------------------------------------------------
def srgb_to_linear(c):
    c = c / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    r, g, b = rgb
    return (
        0.2126 * srgb_to_linear(r)
        + 0.7152 * srgb_to_linear(g)
        + 0.0722 * srgb_to_linear(b)
    )


def contrast_ratio(fg, bg):
    L1, L2 = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (L1 + 0.05) / (L2 + 0.05)


# -----------------------------------------------------
# Color-blindness simulation
# -----------------------------------------------------
PROTAN = np.array(
    [[0.567, 0.433, 0.000],
     [0.558, 0.442, 0.000],
     [0.000, 0.242, 0.758]]
)
DEUTAN = np.array(
    [[0.625, 0.375, 0.000],
     [0.700, 0.300, 0.000],
     [0.000, 0.300, 0.700]]
)
TRITAN = np.array(
    [[0.950, 0.050, 0.000],
     [0.000, 0.433, 0.567],
     [0.000, 0.475, 0.525]]
)


def simulate_cvd(img_rgb, matrix):
    arr = img_rgb.astype(np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    sim = flat @ matrix.T
    sim = np.clip(sim, 0, 1)
    return (sim.reshape(h, w, 3) * 255).astype(np.uint8)


# -----------------------------------------------------
# Palette extraction
# -----------------------------------------------------
def extract_palette(img_rgb, k=5):
    img = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


# -----------------------------------------------------
# AI summary generator (templated)
# -----------------------------------------------------
def compose_insight(prompt, name, rgb, mood, harmony, white_cr, black_cr, dist):
    hexval = "#{:02X}{:02X}{:02X}".format(*rgb)

    mood_expanded = {
        "Energetic": "high-heat, attention grabbing and great for CTAs or warning states.",
        "Fresh": "nature-coded and uplifting — ideal for wellness, sustainability, or outdoors brands.",
        "Calm": "stable, introspective and focused — great for dashboards, finance and productivity tools.",
        "Playful": "vibrant, youthful and expressive — works for entertainment, music and creative apps.",
        "Elegant": "muted and sophisticated — strong candidate for luxury, editorial or portfolio work.",
        "Soft": "pastel and gentle — friendly for onboarding flows, education or mental health.",
        "Minimal": "near-neutral and clean — perfect for modern SaaS layouts.",
        "Moody": "deep and cinematic — good for hero sections, not large text blocks.",
        "Balanced": "versatile — can tilt serious or fun depending on pairing."
    }.get(mood, "versatile and adaptable.")

    harmony_hex = ", ".join(
        [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in harmony]
    )

    acc_bits = []
    if white_cr >= 4.5:
        acc_bits.append("white text passes WCAG AA")
    else:
        acc_bits.append("white text fails AA")

    if black_cr >= 4.5:
        acc_bits.append("black text passes WCAG AA")
    else:
        acc_bits.append("black text fails AA")

    acc_summary = "; ".join(acc_bits)

    # Make the reply feel chatty and contextual to the prompt
    base = [
        f"You're working with **{name}** `{hexval}`.",
        f"Perceptually it's {dist:.1f} units from the nearest reference in the color library.",
        f"Emotionally this reads as **{mood}** — {mood_expanded}",
        f"I generated a {len(harmony)}-color harmony `{harmony_hex}` you can use for backgrounds, accents and borders.",
        f"Accessibility wise: {acc_summary}. Use the higher-contrast option for body text."
    ]

    if prompt:
        base.append(
            f"In the context of your request — _{prompt}_ — I'd lean on this color as a "
            f"primary token and let neutral greys handle most surfaces. Reserve the brightest harmony "
            f"for buttons or highlights so users always know where to look."
        )

    return "\n\n".join(base)


# -----------------------------------------------------
# MAIN APP (Chat-first layout)
# -----------------------------------------------------
def main():
    # 1) Load style.css & script.js from repo root (no /static folder required)
    root_dir = os.path.dirname(__file__)
    style_path = os.path.join(root_dir, "style.css")
    script_path = os.path.join(root_dir, "script.js")

    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            st.markdown("<script>" + f.read() + "</script>", unsafe_allow_html=True)

    # 2) Sidebar: controls + inputs
    st.sidebar.title("🎛 ChromoAI Controls")

    csv_path = ensure_dataset()
    if not csv_path:
        st.stop()

    df = pd.read_csv(csv_path)
    tree = build_kdtree(df)

    color_hex = st.sidebar.color_picker("Base color", "#1976D2")
    rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    st.sidebar.write("RGB:", rgb)

    harmony_mode = st.sidebar.selectbox(
        "Harmony mode",
        ["analogous", "complementary", "triadic", "tetradic", "split"],
        index=0,
    )

    uploaded_img = st.sidebar.file_uploader(
        "Upload image (for palette extraction)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_img:
        pil = Image.open(uploaded_img).convert("RGB")
        arr = np.array(pil)
        palette = extract_palette(arr, k=5)
        st.sidebar.markdown("**Extracted palette**")
        for p in palette:
            st.sidebar.markdown(
                f"<div class='swatch' style='background:rgb{p}; margin-bottom:4px;'></div>",
                unsafe_allow_html=True,
            )
    else:
        palette = None

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: adjust the color, then chat with ChromoAI about themes, branding, or UI ideas.")

    # 3) Main area: chat + visual context
    st.title("🤖 ChromoAI — Color Chatbot")

    # Analyze current color once per run
    name, dist = find_nearest_color(rgb, df, tree)
    harmony = generate_harmony(rgb, harmony_mode)
    mood = mood_from_rgb(rgb)
    white_cr = contrast_ratio((255, 255, 255), rgb)
    black_cr = contrast_ratio((0, 0, 0), rgb)

    # Show a hero swatch + quick stats at top (still chat-focused)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div class='swatch' style='width:140px;height:140px;border-radius:16px;"
            f"background:rgb{rgb};'></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.subheader(f"{name} `{color_hex}`")
        st.write(f"**Mood:** {mood}")
        st.write(f"Contrast vs white: `{white_cr:.2f}`")
        st.write(f"Contrast vs black: `{black_cr:.2f}`")

    st.markdown("---")

    # 4) Chat history
    if "chat_history" not in st.session_state:
        # Initial system-like message
        intro = (
            f"I'm ChromoAI, your color assistant.\n\n"
            f"Right now you're on **{name}** `{color_hex}`, which feels **{mood}**. "
            f"Ask me about branding ideas, UI themes, or how this color behaves in different contexts."
        )
        st.session_state.chat_history = [{"role": "assistant", "content": intro}]

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 5) User input
    user_prompt = st.chat_input("Ask ChromoAI about this color, mood, branding, or design…")

    if user_prompt:
        # show user message
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # generate response
        reply = compose_insight(
            user_prompt,
            name,
            rgb,
            mood,
            harmony,
            white_cr,
            black_cr,
            dist,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # 6) Below chat: color-blindness simulation & harmony visual
    st.markdown("---")
    st.subheader("👁 Color-Blindness Simulation (Selected Color)")

    base = np.zeros((100, 100, 3), dtype=np.uint8)
    base[:] = rgb

    img_pro = simulate_cvd(base, PROTAN)
    img_deu = simulate_cvd(base, DEUTAN)
    img_tri = simulate_cvd(base, TRITAN)

    c1, c2, c3, c4 = st.columns(4)
    c1.image(base, caption="Original", use_column_width=True)
    c2.image(img_pro, caption="Protanopia", use_column_width=True)
    c3.image(img_deu, caption="Deuteranopia", use_column_width=True)
    c4.image(img_tri, caption="Tritanopia", use_column_width=True)

    st.markdown("This shows how your base color collapses or shifts for different color-vision profiles.")

    st.markdown("### Harmony Strip")
    hcols = st.columns(len(harmony))
    for c, h in zip(hcols, harmony):
        c.markdown(
            f"<div class='swatch' style='background:rgb{h};height:80px;'></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()


