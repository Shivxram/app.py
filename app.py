import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

from color_engine import (
    load_color_dataset, find_nearest_color, generate_harmony,
    mood_from_rgb, contrast_ratio, simulate_all_cvd
)

from nlp_engine import interpret_prompt
from chat_engine import build_text_reply

st.set_page_config(page_title="ChromoAI — Color Chat", layout="wide")

def main():
    df, tree = load_color_dataset()

    st.sidebar.title("🎛 ChromoAI Controls")
    base_hex = st.sidebar.color_picker("Base Color", "#1976D2")
    sidebar_rgb = tuple(int(base_hex[i:i+2], 16) for i in (1,3,5))

    harmony_mode = st.sidebar.selectbox(
        "Harmony mode",
        ["analogous","complementary","triadic","tetradic","split"]
    )

    uploaded_file = st.sidebar.file_uploader("Upload image", type=["jpg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        arr = np.array(img)
        st.sidebar.image(img, caption="Uploaded Image")

    # --- Chat UI ---
    st.title("🤖 ChromoAI — AI-like Color Assistant (No API)")

    if "active_rgb" not in st.session_state:
        st.session_state.active_rgb = sidebar_rgb

    active_rgb = st.session_state.active_rgb
    active_hex = "#{:02X}{:02X}{:02X}".format(*active_rgb)
    name, dist = find_nearest_color(active_rgb, df, tree)
    mood = mood_from_rgb(active_rgb)
    harmony = generate_harmony(active_rgb, harmony_mode)
    wcag_white = contrast_ratio((255,255,255), active_rgb)
    wcag_black = contrast_ratio((0,0,0), active_rgb)

    # Header card
    st.markdown(f"""
        <div style='
            background:rgb{active_rgb};
            width:140px;height:140px;border-radius:16px;
            box-shadow:0 4px 20px rgba(0,0,0,0.3);
        '></div>
    """, unsafe_allow_html=True)

    st.subheader(f"{name} — {active_hex}")

    # --- Chat History ---
    if "chat" not in st.session_state:
        st.session_state.chat = [{
            "role": "assistant",
            "content": "Hey, I'm ChromoAI. Ask me: *make it pastel green*, *use #FFAA33*, *generate palette*, *check contrast* etc."
        }]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Tell ChromoAI something…")

    if user_msg:
        st.session_state.chat.append({"role":"user","content":user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # --- Interpret text ---
        intent, target_rgb, target_name, style_hint = interpret_prompt(user_msg, df)

        # Change active color
        if target_rgb:
            st.session_state.active_rgb = target_rgb
            active_rgb = target_rgb

        # Recompute everything
        active_hex = "#{:02X}{:02X}{:02X}".format(*active_rgb)
        name, dist = find_nearest_color(active_rgb, df, tree)
        mood = mood_from_rgb(active_rgb)
        harmony = generate_harmony(active_rgb, harmony_mode)
        wcag_white = contrast_ratio((255,255,255), active_rgb)
        wcag_black = contrast_ratio((0,0,0), active_rgb)

        # --- Build AI-like reply ---
        reply = build_text_reply(
            intent, user_msg, name, active_hex, active_rgb, 
            mood, harmony, wcag_white, wcag_black, style_hint
        )

        st.session_state.chat.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # --- Color Blindness Simulation ---
    st.markdown("---")
    st.subheader("👁 Color-Blindness Simulation")

    sim_dict = simulate_all_cvd(active_rgb)

    cols = st.columns(4)
    for idx, (label, img) in enumerate(sim_dict.items()):
        cols[idx].image(img, caption=label)

if __name__ == "__main__":
    main()
