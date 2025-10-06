#!/usr/bin/env python3
import os
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────

folders = {
    "adamw": {
        "heat": "/home/jimmy/bcat_main_server/adamw_heatmaps",
        "hist": "/home/jimmy/bcat_main_server/adamw_histograms",
    },
    "adan": {
        "heat": "/home/jimmy/bcat_main_server/adan_heatmaps",
        "hist": "/home/jimmy/bcat_main_server/adan_histograms",
    },
    "muon": {
        "heat": "/home/jimmy/bcat_main_server/muon_heatmaps",
        "hist": "/home/jimmy/bcat_main_server/muon_histograms",
    },
}

# Combine layers 0 through 11
layers = list(range(12))  # [0,1,2,...,11]

# Dimensions for each cell
W_heat, H_heat = 3600, 450
W_hist, H_hist = 3600, 600  # match width to heatmaps

# Grid is len(layers) columns by 2 rows-per-optimizer (heat + hist) × 3 optimizers = 6 rows
cols = len(layers)
canvas_w = cols * W_heat
canvas_h = 3 * H_heat + 3 * H_hist


# ─── Helpers ──────────────────────────────────────────────────────────────────

def capital(opt: str) -> str:
    return "AdamW" if opt == "adamw" else opt.capitalize()

def find_image(path_dir: str, opt: str, layer: int, kind: str) -> str:
    """
    Build the expected stem:
      - heatmaps:    {Capital(opt)}_layer_{layer:02d}_time_heatmaps.png
      - histograms:  {Capital(opt)}_layer_{layer}_distributions.png
    """
    name = capital(opt)
    if kind == "time_heatmaps":
        stem = f"{name}_layer_{layer:02d}_{kind}"
    elif kind == "distributions":
        stem = f"{name}_layer_{layer}_{kind}"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    full = os.path.join(path_dir, stem + ".png")
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Missing expected file:\n  {full}")
    return full


# ─── Stitching ────────────────────────────────────────────────────────────────

def stitch_all():
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    y = 0

    # Heatmaps for each optimizer
    for opt in ("muon", "adamw", "adan"):
        heat_dir = folders[opt]["heat"]
        for col, layer in enumerate(layers):
            img_p = find_image(heat_dir, opt, layer, kind="time_heatmaps")
            im = Image.open(img_p).resize((W_heat, H_heat), Image.LANCZOS)
            canvas.paste(im, (col * W_heat, y))
        y += H_heat

    # Histograms for each optimizer
    for opt in ("muon", "adamw", "adan"):
        hist_dir = folders[opt]["hist"]
        for col, layer in enumerate(layers):
            img_p = find_image(hist_dir, opt, layer, kind="distributions")
            im = Image.open(img_p).resize((W_hist, H_hist), Image.LANCZOS)
            canvas.paste(im, (col * W_heat, y))
        y += H_hist

    out = "combined_heatmaps_histograms.png"
    canvas.save(out)
    print(f"Saved combined image to: {out}")


if __name__ == "__main__":
    stitch_all()