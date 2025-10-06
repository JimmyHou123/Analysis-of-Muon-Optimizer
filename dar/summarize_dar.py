#!/usr/bin/env python3
"""
Summarize DAR_w results into:
  1) Per-optimizer DAR summary
  2) Layer-level mean DAR & fractions (DAR>=thresh, DAR==1)

Usage:
  python summarize_dar.py --csv dar_from_tensors.csv --thresh 0.5 \
      --out-optimizer per_optimizer_summary.csv \
      --out-layer layer_level_summary.csv
"""

import argparse
import pandas as pd

def per_optimizer_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("optimizer")["DAR"]
    out = pd.DataFrame({
        "mean":   g.mean(),
        "std":    g.std(),
        "min":    g.min(),
        "max":    g.max(),
        "count":  g.count(),
        "median": g.median(),
        "p75":    g.quantile(0.75),
        "p90":    g.quantile(0.90),
    }).reset_index()
    return out

def layer_level_summary(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    # mean DAR per (optimizer, layer)
    mean_dar = (df.groupby(["optimizer","layer"])["DAR"]
                  .mean()
                  .reset_index()
                  .rename(columns={"DAR":"mean_DAR"}))

    # fraction of heads with DAR >= thresh per (optimizer, layer)
    frac_local = (df.assign(local = df["DAR"] >= thresh)
                    .groupby(["optimizer","layer"])["local"]
                    .mean()
                    .reset_index()
                    .rename(columns={"local": f"frac_heads_DAR>={thresh}"}))

    # fraction of heads with DAR == 1 per (optimizer, layer)
    frac_ones = (df.assign(one = df["DAR"] == 1.0)
                   .groupby(["optimizer","layer"])["one"]
                   .mean()
                   .reset_index()
                   .rename(columns={"one": "frac_heads_DAR==1"}))

    # join
    out = mean_dar.merge(frac_local, on=["optimizer","layer"]).merge(frac_ones, on=["optimizer","layer"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input DAR CSV (columns: optimizer,layer,head,DAR_w,DAR)")
    ap.add_argument("--thresh", type=float, default=0.5, help="Threshold for 'local' heads (default 0.5)")
    ap.add_argument("--out-optimizer", default="per_optimizer_summary.csv", help="Output CSV for per-optimizer summary")
    ap.add_argument("--out-layer", default="layer_level_summary.csv", help="Output CSV for layer-level summary")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # basic schema check
    needed = {"optimizer","layer","head","DAR_w","DAR"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    opt_summary = per_optimizer_summary(df)
    layer_summary = layer_level_summary(df, thresh=args.thresh)

    # print to console
    print("\n=== Per-optimizer DAR summary ===")
    print(opt_summary.to_string(index=False))
    print("\n=== Layer-level mean DAR & fractions ===")
    print(layer_summary.sort_values(["optimizer","layer"]).to_string(index=False))

    # save to files
    opt_summary.to_csv(args.out_optimizer, index=False)
    layer_summary.to_csv(args.out_layer, index=False)
    print(f"\nSaved: {args.out_optimizer}")
    print(f"Saved: {args.out_layer}")

if __name__ == "__main__":
    main()