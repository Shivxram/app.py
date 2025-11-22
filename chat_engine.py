def describe_style(style):
    mapping = {
        "pastel": "Soft pastels reduce saturation and brighten tone.",
        "dark": "Darker themes lower brightness for cinematic mood.",
        "vibrant": "Vibrant boosts saturation for punchy visuals.",
        "muted": "Muted tones reduce saturation for calm, serious design.",
    }
    return mapping.get(style, "")

def build_text_reply(
    intent, prompt, name, hexcode, rgb, mood,
    harmony, wcag_white, wcag_black, style
):
    mood_desc = {
        "Energetic":"high-energy and attention grabbing.",
        "Fresh":"natural and optimistic.",
        "Calm":"stable and peaceful.",
        "Playful":"expressive and youthful.",
        "Elegant":"muted and refined.",
        "Soft":"gentle and friendly.",
        "Minimal":"clean and neutral.",
        "Moody":"deep and cinematic.",
        "Balanced":"versatile and adaptive.",
    }.get(mood, "unique and adjustable.")

    harmonious = ", ".join(
        [f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for c in harmony]
    )

    lines = []
    lines.append(f"You're now using **{name}** `{hexcode}` — mood: **{mood}**, which feels {mood_desc}")

    lines.append(f"Harmony suggestions: `{harmonious}`")

    lines.append(
        f"Accessibility → white text contrast: `{wcag_white:.2f}`, black text: `{wcag_black:.2f}`"
    )

    if style:
        lines.append(f"Style detected: **{style}** — {describe_style(style)}")

    if intent == "generate_palette":
        lines.append("Here's your palette: mix 1–2 hues with neutrals for a solid theme.")

    if intent == "accessibility":
        lines.append("Check WCAG ≥4.5 for body text to ensure readability.")

    if intent == "compare":
        lines.append("Provide two hex codes like `#FF5733 vs #3498DB`.")

    lines.append(f"_Your prompt:_ “{prompt}”")
    return "\n\n".join(lines)
