#!/usr/bin/env python3
"""
Compute the Diagonal-Argmax Rate (DAR_w) from time-only attention heatmap PNGs,
for every head in every layer, for each optimizer (AdamW, Adan, Muon).
It parses the images by cropping margins, splitting into 8 head panels, and downsampling
to a T x T grid to approximate the underlying attention matrices.

Usage (example):
    python compute_dar_from_heatmaps.py \
        --root /home/jimmy/bcat_main_server \
        --optimizers AdamW Adan Muon \
        --layers 0 11 \
        --w 1 \
        --grid-size 13 \
        --save-csv dar_metrics.csv \
        --outdir dar_plots

The script expects files like:
    {root}/adamw_heatmaps/AdamW_layer_00_time_heatmaps.png
    {root}/adan_heatmaps/Adan_layer_04_time_heatmaps.png
    {root}/muon_heatmaps/Muon_layer_04_time_heatmaps.png
"""

import os
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def load_head_matrices_from_image(
    image_path: str,
    n_heads: int = 8,
    grid_size: int = 13,
    outer_crop: Tuple[float, float, float, float] = (0.12, 0.18, 0.03, 0.03),  # top, bottom, left, right (fractions)
    inner_crop: Tuple[float, float, float, float] = (0.13, 0.10, 0.08, 0.06),  # within each head panel
) -> List[np.ndarray]:
    """
    Heuristic parser for 'row-of-8-heads' figures.
    1) Outer-crop away title/colorbar/margins
    2) Split horizontally into n_heads equal panels
    3) Inner-crop away axes inside each panel
    4) Convert to luminance and downsample to grid_size x grid_size
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Outer crop to remove title and colorbar/legends and left/right margins
    top_f, bot_f, left_f, right_f = outer_crop
    crop_box = (
        int(left_f * w),
        int(top_f * h),
        w - int(right_f * w),
        h - int(bot_f * h),
    )
    img_c = img.crop(crop_box)
    wc, hc = img_c.size

    # Split into equal-width panels (heads)
    panel_w = wc // n_heads
    heads = []
    for k in range(n_heads):
        panel = img_c.crop((k * panel_w, 0, (k + 1) * panel_w, hc))
        # Inner crop to remove per-axis ticks/labels within each panel
        pw, ph = panel.size
        it, ib, il, ir = inner_crop
        inner_box = (
            int(il * pw),
            int(it * ph),
            pw - int(ir * pw),
            ph - int(ib * ph),
        )
        panel_c = panel.crop(inner_box)

        # Convert to luminance and downsample
        arr = np.asarray(panel_c).astype(np.float32) / 255.0
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        lum_img = Image.fromarray((lum * 255).astype(np.uint8))
        lum_small = lum_img.resize((grid_size, grid_size), Image.BOX)  # area-ish
        mat = np.asarray(lum_small).astype(np.float32) / 255.0
        heads.append(mat)

    return heads


def diagonal_argmax_rate(mat: np.ndarray, w: int = 1) -> float:
    """
    DAR_w: fraction of rows whose brightest column lies within ±w of the diagonal.
    """
    T = mat.shape[0]
    j_star = np.argmax(mat, axis=1)  # per-row peak column
    i = np.arange(T)
    return float(np.mean(np.abs(j_star - i) <= w))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Root folder containing *_heatmaps directories")
    p.add_argument("--optimizers", type=str, nargs="+", default=["AdamW", "Adan", "Muon"],
                   help="Optimizers to include (affects prefixes/dir names)")
    p.add_argument("--layers", type=int, nargs=2, default=[0, 11],
                   help="Layer range inclusive: start end (e.g., 0 11)")
    p.add_argument("--n-heads", type=int, default=8, help="Heads per image")
    p.add_argument("--grid-size", type=int, default=13, help="Downsample target grid (T)")
    p.add_argument("--w", type=int, default=1, help="DAR band half-width (tokens)")
    p.add_argument("--outer-crop", type=float, nargs=4, default=[0.12, 0.18, 0.03, 0.03],
                   help="Outer crop fractions: top bottom left right")
    p.add_argument("--inner-crop", type=float, nargs=4, default=[0.13, 0.10, 0.08, 0.06],
                   help="Inner crop fractions (per head panel): top bottom left right")
    p.add_argument("--save-csv", type=str, default="dar_metrics.csv")
    p.add_argument("--outdir", type=str, default="dar_plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for opt in args.optimizers:
        folder = os.path.join(args.root, f"{opt.lower()}_heatmaps")  # adamw_heatmaps, adan_heatmaps, muon_heatmaps
        for layer in range(args.layers[0], args.layers[1] + 1):
            fname = f"{opt}_layer_{layer:02d}_time_heatmaps.png"
            fpath = os.path.join(folder, fname)
            if not os.path.exists(fpath):
                print(f"[WARN] Missing: {fpath}")
                continue
            try:
                mats = load_head_matrices_from_image(
                    fpath,
                    n_heads=args.n_heads,
                    grid_size=args.grid_size,
                    outer_crop=tuple(args.outer_crop),
                    inner_crop=tuple(args.inner_crop),
                )
                for head_idx, mat in enumerate(mats):
                    dar = diagonal_argmax_rate(mat, w=args.w)
                    rows.append({
                        "optimizer": opt,
                        "layer": layer,
                        "head": head_idx,
                        "DAR_w": args.w,
                        "DAR": dar,
                        "image": fpath,
                    })
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")

    if not rows:
        print("No rows computed. Check your --root and file naming.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.save_csv, index=False)
    print(f"Saved metrics → {args.save_csv}")

    # Aggregate: mean DAR over heads per layer
    df_layer = df.groupby(["optimizer", "layer"])["DAR"].mean().reset_index()

    # 1) Per-optimizer: Mean DAR vs Layer  (one chart per optimizer)
    for opt in df["optimizer"].unique():
        sub = df_layer[df_layer["optimizer"] == opt].sort_values("layer")
        plt.figure(figsize=(6, 4))
        plt.plot(sub["layer"], sub["DAR"], marker="o")
        plt.xlabel("Layer")
        plt.ylabel(f"Mean DAR (w={args.w})")
        plt.title(f"{opt}: Mean Diagonal-Argmax Rate by Layer")
        out_path = os.path.join(args.outdir, f"{opt.lower()}_mean_dar_by_layer.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

    # 2) Per-optimizer: Head×Layer map (DAR values)  (one chart per optimizer)
    for opt in df["optimizer"].unique():
        sub = df[df["optimizer"] == opt]
        if sub.empty:
            continue
        layers_sorted = sorted(sub["layer"].unique())
        heads_sorted = sorted(sub["head"].unique())
        mat = np.zeros((len(heads_sorted), len(layers_sorted)), dtype=np.float32)
        for i, h in enumerate(heads_sorted):
            for j, L in enumerate(layers_sorted):
                val = sub[(sub["head"] == h) & (sub["layer"] == L)]["DAR"].mean()
                mat[i, j] = 0.0 if np.isnan(val) else val

        plt.figure(figsize=(6, 4))
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.colorbar(label=f"DAR (w={args.w})")
        plt.yticks(range(len(heads_sorted)), heads_sorted)
        plt.xticks(range(len(layers_sorted)), layers_sorted)
        plt.xlabel("Layer")
        plt.ylabel("Head")
        plt.title(f"{opt}: DAR Head×Layer")
        out_path = os.path.join(args.outdir, f"{opt.lower()}_dar_head_by_layer.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()