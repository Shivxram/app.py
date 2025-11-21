# app.py
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
from io import BytesIO

# ---------------------------
# Configuration / Paths
# ---------------------------
# If your dataset isn't in repo root, we fallback to the uploaded zip from chat session.
ZIP_FALLBACK = "/mnt/data/color set.zip"   # <- local artifact (provided via chat/upload)
CSV_NAME = "color_names.csv"

st.set_page_config(page_title="ChromoAI — Color Intelligence", layout="wide")

# ---------------------------
# Utility functions
# ---------------------------
def ensure_dataset():
    # prefer repo CSV, else try to extract from ZIP_FALLBACK
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
            st.error(f"Failed to extract dataset from zip fallback: {e}")
    st.error("Dataset not found. Place color_names.csv in repo root or upload the zip.")
    return None

def rgb_tuple_from_hex(hexstr):
    hexstr = hexstr.lstrip('#')
    return tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb):
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0,0]
    return tuple(int(v) for v in lab)

def build_kdtree(df):
    df['rgb'] = df.apply(lambda x: (x['Red (8 bit)'], x['Green (8 bit)'], x['Blue (8 bit)']), axis=1)
    df['lab'] = df['rgb'].apply(rgb_to_lab)
    pts = np.array(df['lab'].tolist())
    tree = KDTree(pts)
    return tree

def find_nearest_color(rgb, df, tree):
    lab = np.array(rgb_to_lab(rgb)).reshape(1,-1)
    dist, idx = tree.query(lab, k=1)
    name = df.iloc[int(idx[0][0])]['Name']
    return name, float(dist[0][0])

def rgb_to_hsl(rgb):
    r,g,b = [x/255 for x in rgb]
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    return h,s,l

def hsl_to_rgb(h,s,l):
    r,g,b = colorsys.hls_to_rgb(h,l,s)
    return (int(r*255), int(g*255), int(b*255))

def shift_hue(h, deg):
    return (h + deg/360.0) % 1.0

def generate_harmony(rgb, mode="analogous", spread=30):
    h,s,l = rgb_to_hsl(rgb)
    offsets = {
        "analogous":[-spread,0,spread],
        "complementary":[0,180],
        "triadic":[0,120,-120],
        "tetradic":[0,90,180,270],
        "split":[0,150,-150]
    }.get(mode, [0])
    return [hsl_to_rgb(shift_hue(h,d), s, l) for d in offsets]

def mood_from_rgb(rgb):
    r,g,b = [x/255 for x in rgb]
    h_deg = colorsys.rgb_to_hsv(r,g,b)[0]*360
    s = colorsys.rgb_to_hsv(r,g,b)[1]
    if s < 0.15: return "Neutral"
    if 0 <= h_deg <= 60: return "Energetic"
    if 60 < h_deg <= 160: return "Fresh"
    if 160 < h_deg <= 260: return "Calm"
    return "Neutral"

def srgb_to_linear(c):
    c = c/255
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

def luminance(rgb):
    r,g,b = rgb
    return 0.2126*srgb_to_linear(r) + 0.7152*srgb_to_linear(g) + 0.0722*srgb_to_linear(b)

def contrast_ratio(fg,bg):
    L1, L2 = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (L1 + 0.05) / (L2 + 0.05)

def simulate_cvd(img_rgb, matrix):
    arr = img_rgb.astype(np.float32) / 255.0
    h,w,_ = arr.shape
    flat = arr.reshape(-1,3)
    sim = flat @ matrix.T
    sim = np.clip(sim, 0, 1)
    return (sim.reshape(h,w,3)*255).astype(np.uint8)

def extract_palette(img_rgb, k=5):
    img = img_rgb.reshape(-1,3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]

# CVD matrices (representative)
PROTAN = np.array([[0.567,0.433,0.000],[0.558,0.442,0.000],[0.000,0.242,0.758]])
DEUTAN = np.array([[0.625,0.375,0.000],[0.700,0.300,0.000],[0.000,0.300,0.700]])
TRITAN = np.array([[0.950,0.050,0.000],[0.000,0.433,0.567],[0.000,0.475,0.525]])

# ---------------------------
# Generative Assistant (persona)
# ---------------------------
def compose_insight(name, rgb, mood, harmony, white_cr, black_cr, dist):
    """
    Compose a multi-paragraph generative-style insight about the color.
    This is deterministic template-based generation (no external LLM required).
    """
    # tone phrases:
    tonal_map = {
        "Energetic":"punchy and attention-grabbing",
        "Fresh":"natural and revitalizing",
        "Calm":"soothing and stable",
        "Neutral":"balanced and understated"
    }
    mood_phrase = tonal_map.get(mood, mood.lower())
    hexval = "#{:02X}{:02X}{:02X}".format(*rgb)
    # Build narrative
    paragraphs = []
    paragraphs.append(f"**Assistant:** You selected **{name}** ({hexval}). This color sits at a perceptual distance of {dist:.1f} from a canonical named sample in our color taxonomy.")
    paragraphs.append(f"It reads as *{mood.lower()}* — {mood_phrase}. The hue and saturation imply it pairs well with both muted neutrals and richer accents.")
    # Palette guidance
    palette_text = ", ".join([f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in harmony])
    paragraphs.append(f"I generated a harmony palette: {palette_text}. Consider using the leftmost color as a background and the rightmost as an accent for contrast.")
    # Accessibility
    acc = []
    if white_cr >= 4.5: acc.append("white text passes AA")
    else: acc.append("white text fails AA (consider dark text)")
    if black_cr >= 4.5: acc.append("black text passes AA")
    else: acc.append("black text fails AA")
    paragraphs.append(f"Accessibility check: {', '.join(acc)}.")
    # Actionable recommendations
    recs = []
    if black_cr > white_cr:
        recs.append("Use dark/black typography for body copy and reserve white for small badges.")
    else:
        recs.append("Use white typography for high-contrast CTAs; ensure sufficient padding.")
    recs.append("For hero sections, combine the primary color with desaturated complements to avoid visual fatigue.")
    paragraphs.append("Recommendations: " + " ".join(recs))
    return "\n\n".join(paragraphs)

# ---------------------------
# App layout & logic
# ---------------------------
def main():
    st.header("🎨 ChromoAI — Your Color Assistant")
    st.caption("Generative insights, designer recommendations, accessibility checks, and color-blind previews.")

    csv_path = ensure_dataset()
    if not csv_path:
        st.stop()

    # load dataset
    df = pd.read_csv(csv_path)
    tree = build_kdtree(df)

    # left column: controls
    left, right = st.columns([1,2])
    with left:
        st.markdown("### Pick a color")
        color_hex = st.color_picker("Choose a swatch", "#1976D2")
        rgb = tuple(int(color_hex[i:i+2],16) for i in (1,3,5))
        st.write("RGB:", rgb)
        st.markdown("---")
        st.markdown("### Harmony mode")
        mode = st.selectbox("Mode", ["analogous","complementary","triadic","tetradic","split"])
        st.markdown("### Actions")
        if st.button("Generate Insight"):
            generate = True
        else:
            generate = False

    # right column: outputs
    with right:
        name, dist = find_nearest_color(rgb, df, tree)
        st.subheader(f"Nearest Named Color — {name}")
        st.write(f"Perceptual distance: {dist:.2f}")

        # Harmony
        harmony = generate_harmony(rgb, mode=mode)
        st.markdown("#### Harmony palette")
        cols = st.columns(len(harmony))
        for c,h in zip(cols,harmony):
            c.markdown(f"<div style='width:120px;height:80px;border-radius:8px;background:rgb{h};'></div>", unsafe_allow_html=True)
            c.caption(f"RGB {h}")

        # Mood + simple badges
        mood = mood_from_rgb(rgb)
        st.markdown(f"**Mood:** {mood}")

        # WCAG
        white_cr = contrast_ratio((255,255,255), rgb)
        black_cr = contrast_ratio((0,0,0), rgb)
        st.markdown("#### Accessibility (WCAG)")
        st.write(f"White text contrast: {white_cr:.2f}")
        st.write(f"Black text contrast: {black_cr:.2f}")
        if white_cr >= 7 or black_cr >= 7:
            st.success("Excellent contrast (AAA possible)")
        elif white_cr >= 4.5 or black_cr >= 4.5:
            st.info("Good contrast (AA possible)")
        else:
            st.warning("Contrast issues detected")

        # CVD simulations visual
        st.markdown("#### Color-blindness Preview")
        canvas = np.zeros((120,480,3), dtype=np.uint8)
        canvas[:, :120] = rgb
        canvas[:, 120:240] = simulate_cvd(canvas[:, :120].copy(), PROTAN)
        canvas[:, 240:360] = simulate_cvd(canvas[:, :120].copy(), DEUTAN)
        canvas[:, 360:480] = simulate_cvd(canvas[:, :120].copy(), TRITAN)
        st.image(canvas, caption=["Original | Protanopia | Deutan | Tritan (strip)"], use_column_width=False)

        # Image palette extraction
        st.markdown("#### Upload image to extract palette")
        uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"], key="palette_uploader")
        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            arr = np.array(pil)
            st.image(pil, caption="Uploaded")
            palette = extract_palette(arr, k=5)
            pcols = st.columns(len(palette))
            for c,p in zip(pcols,palette):
                c.markdown(f"<div style='width:120px;height:80px;border-radius:8px;background:rgb{p};'></div>", unsafe_allow_html=True)
                c.caption(str(p))

    # Assistant chat / insight box (bottom full width)
    st.markdown("---")
    st.markdown("## 🧠 ChromoAI Insight")
    if generate:
        insight = compose_insight(name, rgb, mood, harmony, white_cr, black_cr, dist)
        st.markdown(insight, unsafe_allow_html=True)
    else:
        st.info("Click **Generate Insight** to see a generative explanation and recommendations for your selected color.")

    # Quick exports
    st.markdown("---")
    st.markdown("### Quick exports")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export harmony as CSV"):
            rows = [{"r":c[0],"g":c[1],"b":c[2],"hex":"#{:02X}{:02X}{:02X}".format(*c)} for c in harmony]
            csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv_bytes, file_name="harmony.csv", mime="text/csv")
    with col2:
        if st.button("Copy suggestion to clipboard (browser)"):
            # Streamlit can't write to user clipboard server-side; instead show the text to copy
            suggestion = compose_insight(name, rgb, mood, harmony, white_cr, black_cr, dist)
            st.code(suggestion)

    # Footer
    st.markdown("---")
    st.caption("ChromoAI — generative color assistant • analytics-ready • exportable")

if __name__ == "__main__":
    main()

