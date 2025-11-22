import re

BASIC_COLORS = {
    "red": (220,53,69), "green": (46,204,113), "blue": (52,152,219),
    "yellow": (241,196,15), "orange": (243,156,18), "purple": (155,89,182),
    "pink": (233,30,99), "teal": (0,150,136), "cyan": (0,188,212),
    "black": (0,0,0), "white": (255,255,255), "grey": (120,120,120)
}

INTENT_KEYWORDS = {
    "change_color": ["change","make it","switch","use","set"],
    "generate_palette": ["palette","scheme","theme"],
    "harmony": ["harmony","analogous","complementary","triadic","tetradic"],
    "accessibility": ["contrast","wcag","accessible"],
    "compare": [" vs ", "compare"],
    "explain": ["explain","describe","meaning"],
}

MOOD_KEYWORDS = {
    "pastel": ["pastel","soft","light","airy"],
    "dark": ["dark","moody","deep"],
    "vibrant": ["vibrant","neon","bright","punchy"],
    "muted": ["muted","dull","low saturation"],
}

def detect_intent(prompt):
    p = prompt.lower()
    for intent, words in INTENT_KEYWORDS.items():
        if any(w in p for w in words):
            return intent
    return "chat"

def detect_style(prompt):
    p = prompt.lower()
    for mood, words in MOOD_KEYWORDS.items():
        if any(w in p for w in words):
            return mood
    return None

def extract_hex(prompt):
    match = re.search(r"#([0-9A-Fa-f]{6})", prompt)
    if match:
        hex_str = "#" + match.group(1)
        rgb = tuple(int(hex_str[i:i+2],16) for i in (1,3,5))
        return rgb, hex_str
    return None, None

def extract_rgb(prompt):
    match = re.search(r"rgb\s*\((\d+),(\d+),(\d+)\)", prompt)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None

def extract_named(prompt, df):
    p = prompt.lower()
    # from dataset
    for _, row in df.iterrows():
        name = row["Name"].lower()
        if name in p:
            return (row["Red (8 bit)"], row["Green (8 bit)"], row["Blue (8 bit)"]), row["Name"]
    # from basic list
    for name, rgb in BASIC_COLORS.items():
        if name in p:
            return rgb, name.capitalize()
    return None, None

def interpret_prompt(prompt, df):
    intent = detect_intent(prompt)
    style = detect_style(prompt)

    # HEX
    rgb, h = extract_hex(prompt)
    if rgb:
        return "change_color", rgb, h, style

    # RGB()
    rgb = extract_rgb(prompt)
    if rgb:
        return "change_color", rgb, f"rgb{rgb}", style

    # Named color
    rgb, name = extract_named(prompt, df)
    if rgb:
        return "change_color", rgb, name, style

    return intent, None, None, style
