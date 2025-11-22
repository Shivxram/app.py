import numpy as np
import pandas as pd
import cv2

def rgb_to_lab(rgb):
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0,0]
    return tuple(int(v) for v in lab)

def load_color_dataset():
    # DIRECT FILE — must be in repo root
    df = pd.read_csv("color_names.csv")

    df["rgb"] = df.apply(lambda x: (x["Red (8 bit)"], x["Green (8 bit)"], x["Blue (8 bit)"]), axis=1)
    df["lab"] = df["rgb"].apply(rgb_to_lab)

    from sklearn.neighbors import KDTree
    pts = np.array(df["lab"].tolist())
    tree = KDTree(pts)

    return df, tree


def find_nearest_color(rgb, df, tree):
    lab = np.array(rgb_to_lab(rgb)).reshape(1,-1)
    dist, idx = tree.query(lab, k=1)
    name = df.iloc[int(idx[0][0])]["Name"]
    return name, float(dist[0][0])

import colorsys

def rgb_to_hsl(rgb):
    r,g,b = [x/255 for x in rgb]
    h,l,s = colorsys.hls_to_rgb(r,g,b)
    return h,s,l

def generate_harmony(rgb, mode, spread=30):
    import colorsys
    r,g,b = [x/255 for x in rgb]
    h,l,s = colorsys.rgb_to_hls(r,g,b)

    def shift(h, deg): return (h + deg/360) % 1.0
    def hsl_to_rgb(h,l,s):
        r,g,b = colorsys.hls_to_rgb(h,l,s)
        return (int(r*255),int(g*255),int(b*255))

    offsets = {
        "analogous":[-spread,0,spread],
        "complementary":[0,180],
        "triadic":[0,120,-120],
        "tetradic":[0,90,180,270],
        "split":[0,150,-150],
    }.get(mode,[0])

    return [hsl_to_rgb(shift(h,d), l, s) for d in offsets]

def mood_from_rgb(rgb):
    r,g,b = [x/255 for x in rgb]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    h_deg = h*360

    if s<0.08 and v>0.85: return "Minimal"
    if v<0.2: return "Moody"
    if s>0.7 and h_deg<60: return "Energetic"
    if s>0.6 and 60<h_deg<150: return "Fresh"
    if s>0.5 and 150<h_deg<260: return "Calm"
    if s>0.6 and 260<h_deg<330: return "Playful"
    if s<0.35 and 0.25<v<0.8: return "Elegant"
    if s<0.25 and v>0.8: return "Soft"
    return "Balanced"


def srgb_to_linear(c):
    c = c/255
    return c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4

def luminance(rgb):
    r,g,b = rgb
    return (
        0.2126*srgb_to_linear(r)
        + 0.7152*srgb_to_linear(g)
        + 0.0722*srgb_to_linear(b)
    )

def contrast_ratio(fg,bg):
    L1, L2 = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (L1+0.05)/(L2+0.05)

PROTAN = np.array([[0.567,0.433,0],[0.558,0.442,0],[0,0.242,0.758]])
DEUTAN = np.array([[0.625,0.375,0],[0.700,0.300,0],[0,0.300,0.700]])
TRITAN = np.array([[0.95,0.05,0],[0,0.433,0.567],[0,0.475,0.525]])

def simulate(matrix, rgb):
    base = np.zeros((80,80,3), dtype=np.uint8)
    base[:] = rgb

    arr = base.astype(np.float32)/255
    flat = arr.reshape(-1,3)
    sim = flat @ matrix.T
    sim = np.clip(sim, 0,1)

    return (sim.reshape(80,80,3)*255).astype(np.uint8)

def simulate_all_cvd(rgb):
    return {
        "Original": simulate(np.eye(3), rgb),
        "Protanopia": simulate(PROTAN, rgb),
        "Deuteranopia": simulate(DEUTAN, rgb),
        "Tritanopia": simulate(TRITAN, rgb),
    }
