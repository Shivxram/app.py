import streamlit as st
import numpy as np
import pandas as pd
import colorsys
import cv2
import os
import re
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from PIL import Image

# -----------------------------------------------------
# Streamlit Config
# -----------------------------------------------------
st.set_page_config(page_title="ChromoAI — Color Chatbot", layout="wide")

CSV_NAME = "color_names.csv"

# -----------------------------------------------------
# Color utilities
# -----------------------------------------------------
def rgb_to_lab(rgb):
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return tuple(int(v) for v in lab)

@st.cache_data
def load_color_dataset(csv_name=CSV_NAME):
    if not os.path.exists(csv_name):
        st.error(f"CSV '{csv_name}' not found. Place color_names.csv in repo root.")
        st.stop()
    df = pd.read_csv(csv_name)
    df["rgb"] = df.apply(
        lambda x: (x["Red (8 bit)"], x["Green (8 bit)"], x["Blue (8 bit)"]), axis=1
    )
    df["lab"] = df["rgb"].apply(rgb_to_lab)
    pts = np.array(df["lab"].tolist())
    tree = KDTree(pts)
    return df, tree

def find_nearest_color(rgb, df, tree):
    lab = np.array(rgb_to_lab(rgb)).reshape(1, -1)
    dist, idx = tree.query(lab, k=1)
    name = df.iloc[int(idx[0][0])]["Name"]
    return name, float(dist[0][0])

def rgb_to_hsl(rgb):
    r, g, b = [x / 255 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Return in HSL order
    return h, s, l

def hsl_to_rgb(h, s, l):
    # Convert back from HSL to RGB via HLS
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

# Color-blindness matrices
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

def extract_palette(img_rgb, k=5):
    img = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(img)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]

# -----------------------------------------------------
# Mood / style logic
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

# Style keywords
MOOD_KEYWORDS = {
    "pastel": ("pastel", "soft", "light", "airy"),
    "dark": ("dark", "moody", "deep"),
    "vibrant": ("vibrant", "bright", "punchy", "neon"),
    "muted": ("muted", "dull", "low saturation"),
    "warm": ("warm", "sunset", "orange-ish", "red-ish"),
    "cool": ("cool", "cold", "blue-ish"),
    "calm": ("calm", "relax", "soothing"),
    "energetic": ("energetic", "active", "bold"),
}

def detect_style(prompt: str):
    text = prompt.lower()
    for style, words in MOOD_KEYWORDS.items():
        if any(w in text for w in words):
            return style
    return None

# -----------------------------------------------------
# Mini-LLM style adjustment engine
# -----------------------------------------------------
def infer_adjustment(prompt: str, rgb):
    """
    Understand phrases like:
    - 'make it more red'
    - 'slightly brighter'
    - 'darker blue'
    - 'less saturated', 'muted', 'neon', etc.
    """
    text = prompt.lower()
    r, g, b = rgb

    # Channel nudges
    if "more red" in text:
        r = min(255, r + 40)
    if "more blue" in text:
        b = min(255, b + 40)
    if "more green" in text:
        g = min(255, g + 40)

    if "less red" in text:
        r = max(0, r - 40)
    if "less blue" in text:
        b = max(0, b - 40)
    if "less green" in text:
        g = max(0, g - 40)

    # Brightness
    if any(word in text for word in ["brighter", "lighter", "lighten"]):
        r = min(255, r + 25)
        g = min(255, g + 25)
        b = min(255, b + 25)
    if any(word in text for word in ["darker", "darken"]):
        r = max(0, r - 25)
        g = max(0, g - 25)
        b = max(0, b - 25)

    # Saturation / style
    if any(word in text for word in ["neon", "vibrant", "more saturated", "punchy"]):
        h, s, l = rgb_to_hsl(rgb)
        s = min(1.0, s + 0.25)
        return hsl_to_rgb(h, s, l)

    if any(word in text for word in ["muted", "desaturate", "less saturated", "dull"]):
        h, s, l = rgb_to_hsl(rgb)
        s = max(0.0, s - 0.25)
        return hsl_to_rgb(h, s, l)

    return (int(r), int(g), int(b))

def mini_llm_reply(user_prompt: str, current_color, style_hint: str | None):
    """
    Adds a little fake 'reasoning' flavor without any external LLM.
    """
    import random

    templates = [
        "Reading your prompt, this feels like a **{style}** adjustment request.",
        "Interpreting your message as a push toward a more **{style}** color profile.",
        "From your wording, I'm nudging the color into a **{style}** direction.",
        "Your description hints at a **{style}** mood, so I shifted the tone slightly.",
    ]

    fallback_styles = [
        "softer", "bolder", "deeper", "cleaner",
        "warmer", "cooler", "more playful", "more serious"
    ]

    chosen_style = style_hint if style_hint else random.choice(fallback_styles)
    base_sentence = random.choice(templates).format(style=chosen_style)

    suggestion = f"RGB {current_color}"

    extra = f"{base_sentence}\n\nYou can keep iterating with phrases like *more red*, *brighter*, *more muted*, or another hex code. Current color is {suggestion}."
    return extra

# -----------------------------------------------------
# Prompt parsing (offline NLP-ish)
# -----------------------------------------------------
BASIC_COLORS = {
    "red": (220, 53, 69),
    "green": (46, 204, 113),
    "blue": (52, 152, 219),
    "yellow": (241, 196, 15),
    "orange": (243, 156, 18),
    "purple": (155, 89, 182),
    "violet": (136, 84, 208),
    "pink": (233, 30, 99),
    "teal": (0, 150, 136),
    "cyan": (0, 188, 212),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (120, 120, 120),
    "grey": (120, 120, 120),
    "brown": (121, 85, 72),
}

INTENT_KEYWORDS = {
    "change_color": ["change", "set", "make it", "switch to", "use", "try"],
    "generate_palette": ["palette", "colors", "scheme", "theme"],
    "harmony": ["harmony", "analogous", "complementary", "triadic", "tetradic"],
    "accessibility": ["contrast", "wcag", "accessible", "readable"],
    "explain": ["explain", "describe", "meaning", "vibe", "feel"],
    "compare": ["compare", " vs ", "versus"],
}

def extract_hex(prompt: str):
    match = re.search(r"#([0-9A-Fa-f]{6})", prompt)
    if match:
        hexval = "#" + match.group(1)
        return tuple(int(hexval[i+1:i+3], 16) for i in (0, 2, 4)), hexval.upper()
    return None, None

def extract_rgb_pattern(prompt: str):
    match = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", prompt)
    if match:
        rgb = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return rgb
    return None

def extract_named_color(prompt: str, df: pd.DataFrame):
    text = prompt.lower()
    # dataset names
    for _, row in df.iterrows():
        cname = str(row["Name"]).lower()
        if cname and cname in text:
            r, g, b = row["Red (8 bit)"], row["Green (8 bit)"], row["Blue (8 bit)"]
            return (int(r), int(g), int(b)), row["Name"]
    # basic color words
    for word, rgb in BASIC_COLORS.items():
        if word in text:
            return rgb, word.capitalize()
    return None, None

def detect_intent(prompt: str):
    text = prompt.lower()
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in text for k in keys):
            return intent
    return "chat"

def interpret_prompt(prompt: str, df: pd.DataFrame):
    """
    Returns (intent, target_rgb, target_name, style_hint)
    Now supports 'adjust_color' for relative changes.
    """
    intent = detect_intent(prompt)
    style = detect_style(prompt)

    # Try hex
    rgb, hexstr = extract_hex(prompt)
    if rgb:
        return "change_color", rgb, hexstr, style

    # Try rgb(...)
    rgb_pat = extract_rgb_pattern(prompt)
    if rgb_pat:
        return "change_color", rgb_pat, f"rgb{rgb_pat}", style

    # Try named color from dataset / basic dict
    rgb_name, cname = extract_named_color(prompt, df)
    if rgb_name:
        return "change_color", rgb_name, cname, style

    # Language hints for relative adjustments
    adjustment_words = [
        "more red", "more blue", "more green", "less red", "less blue", "less green",
        "brighter", "lighter", "lighten", "darker", "darken",
        "more saturated", "less saturated", "muted", "neon", "vibrant", "punchy"
    ]
    if any(w in prompt.lower() for w in adjustment_words):
        return "adjust_color", None, None, style

    # No explicit color, just generic chat / commands
    return intent, None, None, style

# -----------------------------------------------------
# Response generation (rule-based "AI")
# -----------------------------------------------------
def describe_style_modifier(style: str | None):
    if style is None:
        return ""
    mapping = {
        "pastel": "Make it soft, light and slightly desaturated.",
        "dark": "Shift towards lower brightness and deeper tones.",
        "vibrant": "Increase saturation and keep brightness high.",
        "muted": "Lower saturation and keep things calm.",
        "warm": "Bias the palette toward reds, oranges and warm yellows.",
        "cool": "Bias the palette toward blues, teals and cool greens.",
        "calm": "Avoid extreme saturation and keep mid-value tones.",
        "energetic": "Use high contrast accents with saturated primaries.",
    }
    return mapping.get(style, "")

def build_text_reply(
    intent,
    user_prompt,
    base_name,
    base_hex,
    active_rgb,
    mood,
    harmony,
    white_cr,
    black_cr,
    style_hint,
):
    mood_explainer = {
        "Energetic": "high energy and attention grabbing — good for CTAs and alerts.",
        "Fresh": "natural and optimistic — nice for wellness or outdoor products.",
        "Calm": "stable and focused — strong for dashboards and productivity tools.",
        "Playful": "loud and expressive — fits music, youth and creative brands.",
        "Elegant": "muted and refined — nice for editorial or luxury surfaces.",
        "Soft": "gentle and friendly — nice for education, onboarding or mental health.",
        "Minimal": "clean and neutral — perfect for modern SaaS layouts.",
        "Moody": "cinematic and deep — better as background than text.",
        "Balanced": "flexible — can tilt playful or serious depending on supporting colors.",
    }
    mood_text = mood_explainer.get(mood, "versatile and adaptable.")

    harmony_hex = ", ".join(
        [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in harmony]
    )

    lines = []
    if intent == "adjust_color":
        lines.append("I treated your message as a **subtle adjustment** to the existing color rather than a full switch.")
    else:
        lines.append(f"You're currently on **{base_name}** `{base_hex}`.")

    lines.append(f"Visually this reads as **{mood}** — {mood_text}")

    if intent in ["change_color", "chat", "explain", "adjust_color"]:
        lines.append(
            f"I built a {len(harmony)}-color harmony around it: `{harmony_hex}` "
            f"which you can use as background / accent / border roles."
        )

    lines.append(
        f"Accessibility: contrast vs white is **{white_cr:.2f}**, vs black is **{black_cr:.2f}**. "
        f"For body text, I'd usually pick the higher contrast option."
    )

    style_mod = describe_style_modifier(style_hint)
    if style_mod:
        lines.append(f"Style hint detected: **{style_hint}**. {style_mod}")

    # intent-specific advice
    if intent == "generate_palette":
        lines.append(
            "Since you asked for a palette, use 1–2 of these colors for primary/secondary, "
            "and let neutrals handle most backgrounds so the UI doesn't feel noisy."
        )
    elif intent == "accessibility":
        lines.append(
            "Because you mentioned contrast/accessibility, test your main text sizes "
            "against both light and dark surfaces. Aim for ≥4.5 contrast for body text."
        )
    elif intent == "harmony":
        lines.append(
            "Since you're focusing on harmony, keep adjacent hues for soft layouts and "
            "use the furthest hue as your accent for emphasis."
        )
    elif intent == "compare":
        lines.append(
            "Right now I'm only analyzing one active color at a time. To compare two, "
            "give me two hex codes like `#FF5733 vs #3498DB` in your next message."
        )

    # Mini-LLM flavored follow-up
    if intent in ["adjust_color", "chat", "change_color"]:
        lines.append(mini_llm_reply(user_prompt, active_rgb, style_hint))

    lines.append(
        f"_Prompt I responded to:_ “{user_prompt}”"
    )
    return "\n\n".join(lines)

# -----------------------------------------------------
# Main app (chat-first UI)
# -----------------------------------------------------
def main():
    df, tree = load_color_dataset()

    # Sidebar controls
    st.sidebar.title("🎛 ChromoAI Controls")
    default_hex = "#1976D2"
    color_hex = st.sidebar.color_picker("Base color", default_hex)
    sidebar_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))

    st.sidebar.markdown("---")
    harmony_mode = st.sidebar.selectbox(
        "Harmony mode",
        ["analogous", "complementary", "triadic", "tetradic", "split"],
        index=0,
    )

    uploaded_img = st.sidebar.file_uploader(
        "Upload image (for palette extraction)",
        type=["jpg", "jpeg", "png"],
    )

    if "active_rgb" not in st.session_state:
        st.session_state.active_rgb = sidebar_rgb

    if uploaded_img:
        pil = Image.open(uploaded_img).convert("RGB")
        arr = np.array(pil)
        palette = extract_palette(arr, k=5)
        st.sidebar.markdown("**Extracted palette**")
        for p in palette:
            st.sidebar.markdown(
                f"<div style='width:60px;height:30px;border-radius:6px;background:rgb{p};margin-bottom:4px;'></div>",
                unsafe_allow_html=True,
            )
    st.sidebar.caption("Hints: try prompts like 'make it pastel green', 'use #FF5733', 'more red', 'make it darker', 'neon blue', 'check contrast', 'generate palette'.")

    # --------- CORE STATE & ANALYTICS (always recomputed) ---------
    active_rgb = st.session_state.active_rgb
    active_hex = "#{:02X}{:02X}{:02X}".format(*active_rgb)
    name, dist = find_nearest_color(active_rgb, df, tree)
    harmony = generate_harmony(active_rgb, harmony_mode)
    mood = mood_from_rgb(active_rgb)
    white_cr = contrast_ratio((255, 255, 255), active_rgb)
    black_cr = contrast_ratio((0, 0, 0), active_rgb)

    # Main header
    st.title("🤖 ChromoAI — Keyword-Driven Color Chatbot (No API)")

    # Hero row
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div style='width:140px;height:140px;border-radius:18px;background:rgb{active_rgb};"
            f"box-shadow:0 10px 30px rgba(0,0,0,0.2);'></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.subheader(f"{name} `{active_hex}`")
        st.write(f"**Mood:** {mood}")
        st.write(f"Nearest dataset color distance: `{dist:.1f}`")
        st.write(f"Contrast vs white: **{white_cr:.2f}**, vs black: **{black_cr:.2f}**")

    # Harmony preview
    st.subheader("🎨 Harmony Preview")
    hcols = st.columns(len(harmony))
    for col, h in zip(hcols, harmony):
        col.markdown(
            f"""
            <div style="
                width:80px;
                height:80px;
                border-radius:12px;
                background:rgb{h};
                box-shadow:0 4px 12px rgba(0,0,0,0.25);
            "></div>
            """,
            unsafe_allow_html=True,
        )
        col.caption(f"rgb{h}")

    # Contrast checker
    st.subheader("🔎 Contrast Checker")
    cc1, cc2 = st.columns(2)

    cc1.markdown(
        f"""
        <div style="
            background:rgb{active_rgb};
            color:white;
            padding:16px;
            border-radius:10px;
            font-weight:600;
            text-align:center;
        ">
            White Text (Contrast {white_cr:.2f})
        </div>
        """,
        unsafe_allow_html=True,
    )

    cc2.markdown(
        f"""
        <div style="
            background:rgb{active_rgb};
            color:black;
            padding:16px;
            border-radius:10px;
            font-weight:600;
            text-align:center;
        ">
            Black Text (Contrast {black_cr:.2f})
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Chat history
    if "chat" not in st.session_state:
        intro = (
            "I'm ChromoAI — I don't use any external LLMs, but I understand color names, hex codes, "
            "rgb(), style words like *pastel / dark / neon*, and commands like "
            "*change, make it darker, more red, generate palette, check contrast, explain*.\n\n"
            "Try things like: `make it pastel green`, `use #FF5733`, `more saturated blue`, "
            "`make it darker`, or `check accessibility for this color`."
        )
        st.session_state.chat = [{"role": "assistant", "content": intro}]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Talk to ChromoAI about colors, vibes, palettes, accessibility…")

    if user_prompt:
        # Add user message
        st.session_state.chat.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Interpret prompt
        intent, target_rgb, target_name, style_hint = interpret_prompt(user_prompt, df)

        # Update active color if user specified one or requested adjustment
        if target_rgb is not None:
            st.session_state.active_rgb = target_rgb
        elif intent == "adjust_color":
            adjusted = infer_adjustment(user_prompt, st.session_state.active_rgb)
            st.session_state.active_rgb = adjusted

        # Recompute analytics after updating state
        active_rgb = st.session_state.active_rgb
        active_hex = "#{:02X}{:02X}{:02X}".format(*active_rgb)
        name, dist = find_nearest_color(active_rgb, df, tree)
        harmony = generate_harmony(active_rgb, harmony_mode)
        mood = mood_from_rgb(active_rgb)
        white_cr = contrast_ratio((255, 255, 255), active_rgb)
        black_cr = contrast_ratio((0, 0, 0), active_rgb)

        # Build reply text
        reply = build_text_reply(
            intent=intent,
            user_prompt=user_prompt,
            base_name=name,
            base_hex=active_hex,
            active_rgb=active_rgb,
            mood=mood,
            harmony=harmony,
            white_cr=white_cr,
            black_cr=black_cr,
            style_hint=style_hint,
        )

        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Color-blindness simulation for active color
    st.markdown("---")
    st.subheader("👁 Color-Blindness Simulation (Current Color)")

    base = np.zeros((80, 80, 3), dtype=np.uint8)
    base[:] = st.session_state.active_rgb

    img_pro = simulate_cvd(base, PROTAN)
    img_deu = simulate_cvd(base, DEUTAN)
    img_tri = simulate_cvd(base, TRITAN)

    c1, c2, c3, c4 = st.columns(4)
    c1.image(base, caption="Original", use_column_width=True)
    c2.image(img_pro, caption="Protanopia", use_column_width=True)
    c3.image(img_deu, caption="Deuteranopia", use_column_width=True)
    c4.image(img_tri, caption="Tritanopia", use_column_width=True)

    st.caption("This shows how your current color shifts for different color-vision profiles.")


if __name__ == "__main__":
    main()
