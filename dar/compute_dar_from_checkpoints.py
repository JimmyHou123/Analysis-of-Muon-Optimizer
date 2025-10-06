#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relate non-Gaussianity (from checkpoint weights) to performance.

Reads PyTorch checkpoints (.pt/.pth), extracts per-layer weight tensors
(Q/K/V etc. or ANY 2-D weights matching include regex), computes:
  - KL divergence to a *fitted Gaussian* (fit mu/std on the same tensor values)
  - skewness, Pearson kurtosis, std
  - Bimodality Coefficient (BC = (skew^2 + 1) / kurtosis)

Then correlates these stats with validation metrics by step.

USAGE EXAMPLES
--------------
# Multiple optimizers, each with its own experiment directory of checkpoints:
python relate_from_checkpoints.py \
  --exp AdamW:/path/to/adamw_ckpts \
  --exp Adan:/path/to/adan_ckpts \
  --exp Muon:/path/to/muon_ckpts \
  --ckpt-pattern "*.pth" \
  --metrics /path/to/adamw_metrics.csv --metrics-tag AdamW \
  --metrics /path/to/adan_metrics.csv  --metrics-tag Adan \
  --metrics /path/to/muon_metrics.csv  --metrics-tag Muon \
  --perf-col val_l2 \
  --out-dir ./from_ckpt_correlations

# Single combined metrics CSV with an 'optimizer' column:
python relate_from_checkpoints.py \
  --exp AdamW:/path/to/adamw_ckpts \
  --exp Adan:/path/to/adan_ckpts \
  --exp Muon:/path/to/muon_ckpts \
  --ckpt-pattern "checkpoint_step*.pt" \
  --metrics /path/to/all_metrics.csv \
  --metrics-optimizer-col optimizer \
  --perf-col val_l2 \
  --out-dir ./from_ckpt_correlations

NOTES
-----
- Default layer index parsing expects names like "...layers.{L}....".
  Adjust --layer-regex if your model uses a different convention.
- By default we include keys containing: q_proj|k_proj|v_proj|out_proj|attn|fc|linear|mlp|ffn
  and exclude bias/LayerNorm/embedding vectors. You can override with regex flags.
- If your perf is ACCURACY (higher better), change the optimizer-level aggregator
  from min() to max() (see comment near end).

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import matplotlib.pyplot as plt


# ---------- Configurable parsing heuristics ----------

# Extract step from filename: "..._step300000...", "...S300000...", "...300000.pth"
STEP_PATTERNS = [
    r".*?_step(?P<step>\d+).*",
    r".*?_S(?P<step>\d+).*",
    r".*?[_\-\.](?P<step>\d+)\.(?:pt|pth)$",
]

# Default: extract "layer index" from module names like "...layers.3...."
DEFAULT_LAYER_REGEX = r".*?layers\.(?P<layer>\d+).*"

# Default key-inclusion and exclusion regexes
DEFAULT_INCLUDE = r"(q_proj|k_proj|v_proj|out_proj|attn|fc|linear|mlp|ffn)"
DEFAULT_EXCLUDE = r"(bias$|norm|layernorm|ln\d*|embedding|embed|pos_enc|positional|time_embed)"


# ---------- Helpers ----------

def parse_step_from_filename(name: str) -> Optional[int]:
    for pat in STEP_PATTERNS:
        m = re.match(pat, name)
        if m:
            return int(m.group("step"))
    return None

def parse_layer_from_key(key: str, layer_regex: str) -> Optional[int]:
    m = re.match(layer_regex, key)
    if m and "layer" in m.groupdict():
        try:
            return int(m.group("layer"))
        except Exception:
            return None
    return None

def tensor_stats_non_gaussianity(x: torch.Tensor, nbins: int = 256) -> Tuple[float, float, float, float]:
    """
    Compute KL(P||Q) where P is histogram of weights and Q is fitted Gaussian,
    plus skewness, Pearson kurtosis, and std from raw samples.
    """
    x = x.detach().float().reshape(-1).cpu().numpy()
    x = x[np.isfinite(x)]
    if x.size < 32:
        return np.nan, np.nan, np.nan, np.nan

    mu = float(np.mean(x))
    std = float(np.std(x))
    counts, edges = np.histogram(x, bins=nbins)
    p = counts.astype(np.float64) + 1e-12
    p /= p.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    q = stats.norm.pdf(centers, loc=mu, scale=max(std, 1e-12)) + 1e-12
    q /= q.sum()
    kl = float(np.sum(p * (np.log(p) - np.log(q))))

    skew = stats.skew(x, bias=False)
    kurt_pearson = stats.kurtosis(x, fisher=False, bias=False)  # normal -> 3.0
    return kl, skew, kurt_pearson, std

def bimodality_coeff(skew: float, kurt_pearson: float) -> float:
    if not np.isfinite(skew) or not np.isfinite(kurt_pearson) or kurt_pearson <= 0:
        return np.nan
    return (skew**2 + 1.0) / float(kurt_pearson)

def guess_perf_col(df: pd.DataFrame, user: Optional[str]) -> str:
    if user and user in df.columns:
        return user
    for c in ["val_l2", "val_loss", "val_error", "val_rmse", "val_acc", "validation", "valid"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find performance column. Pass --perf-col explicitly.")


# ---------- Main pipeline ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, action="append", default=[],
                    help="OptimizerName:/path/to/ckpts (repeatable). E.g., AdamW:/exp/adamw")
    ap.add_argument("--ckpt-pattern", type=str, default="*.pth", help="Glob to find checkpoints in each exp dir.")
    ap.add_argument("--layer-regex", type=str, default=DEFAULT_LAYER_REGEX,
                    help="Regex to extract layer index from param key; must contain (?P<layer>\\d+).")
    ap.add_argument("--include", type=str, default=DEFAULT_INCLUDE,
                    help="Regex: include param keys that match (defaults target Q/K/V etc).")
    ap.add_argument("--exclude", type=str, default=DEFAULT_EXCLUDE,
                    help="Regex: exclude param keys (bias, norms, embeddings, etc.).")
    ap.add_argument("--nbins", type=int, default=256, help="Bins for histogram.")
    ap.add_argument("--metrics", type=str, action="append", default=[], help="Path to metrics CSV (repeatable).")
    ap.add_argument("--metrics-tag", type=str, action="append", default=[],
                    help="Optimizer tag for each --metrics (same order).")
    ap.add_argument("--metrics-optimizer-col", type=str, default=None,
                    help="If a single CSV contains all optimizers, name of column with optimizer labels.")
    ap.add_argument("--perf-col", type=str, default=None, help="Performance column name, e.g., val_l2.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Parse experiments
    if not args.exp:
        raise ValueError("Provide at least one --exp OptimizerName:/path/to/ckpts")
    exp_map: Dict[str, Path] = {}
    for spec in args.exp:
        if ":" not in spec:
            raise ValueError(f"--exp must be OptimizerName:/path, got: {spec}")
        name, p = spec.split(":", 1)
        exp_map[name.strip()] = Path(p.strip())

    include_re = re.compile(args.include)
    exclude_re = re.compile(args.exclude) if args.exclude else None

    rows = []  # collected stats

    for opt_name, exp_path in exp_map.items():
        if not exp_path.exists():
            print(f"[WARN] Missing exp path: {opt_name} -> {exp_path}")
            continue

        ckpts = sorted(exp_path.rglob(args.ckpt_pattern))
        if not ckpts:
            print(f"[WARN] No checkpoints for {opt_name} under {exp_path} with pattern {args.ckpt-pattern}")
            continue

        for ckpt_path in ckpts:
            step = parse_step_from_filename(ckpt_path.name)
            # Load checkpoint
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            except Exception as e:
                print(f"[WARN] Failed to load {ckpt_path}: {e}")
                continue

            # Try alternative step sources
            if step is None:
                for k in ["global_step", "step", "train_step", "iteration", "it"]:
                    if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], (int, float)):
                        step = int(ckpt[k])
                        break

            sd = ckpt.get("state_dict", None) if isinstance(ckpt, dict) else None
            if sd is None:
                # Some checkpoints store the raw state_dict at top level
                if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    sd = ckpt
                else:
                    print(f"[WARN] No state_dict found in {ckpt_path}")
                    continue

            for key, tensor in sd.items():
                # Only weights that are 2-D (matrices) are interesting here
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    continue
                if not include_re.search(key):
                    continue
                if exclude_re and exclude_re.search(key):
                    continue

                layer = parse_layer_from_key(key, args.layer_regex)
                kl, skew, kurt, std = tensor_stats_non_gaussianity(tensor, nbins=args.nbins)
                bc = bimodality_coeff(skew, kurt)
                rows.append({
                    "optimizer": opt_name,
                    "step": step,
                    "layer": layer,
                    "param_key": key,
                    "file": str(ckpt_path),
                    "kl_to_fitted_gauss": kl,
                    "skewness": skew,
                    "kurtosis_pearson": kurt,
                    "std": std,
                    "bimodality_coeff": bc,
                })

    if not rows:
        raise ValueError("No eligible 2-D weights were processed. Adjust --include/--exclude and --layer-regex.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ckpt_param_stats_raw.csv", index=False)

    # Aggregate to per (optimizer, layer, step)
    agg_cols = ["kl_to_fitted_gauss", "skewness", "kurtosis_pearson", "std", "bimodality_coeff"]
    per_ols = (df.groupby(["optimizer", "layer", "step"], dropna=False)[agg_cols]
                 .mean().reset_index())
    per_ols.to_csv(out_dir / "ckpt_param_stats_by_layerstep.csv", index=False)

    # Also aggregate per (optimizer, layer) across steps
    per_ol = (df.groupby(["optimizer", "layer"], dropna=False)[agg_cols]
                .mean().reset_index())
    per_ol.to_csv(out_dir / "ckpt_param_stats_by_layer.csv", index=False)

    # ---------------- Metrics join & correlations ----------------
    # Load metrics
    metrics_frames: List[pd.DataFrame] = []
    if args.metrics and args.metrics_optimizer_col:
        for m in args.metrics:
            tdf = pd.read_csv(m)
            if "step" not in tdf.columns:
                raise ValueError(f"Metrics file missing 'step': {m}")
            perf_col = guess_perf_col(tdf, args.perf_col)
            if args.metrics_optimizer_col not in tdf.columns:
                raise ValueError(f"Metrics file missing '{args.metrics_optimizer_col}': {m}")
            tdf = tdf[["step", args.metrics_optimizer_col, perf_col]] \
                    .rename(columns={args.metrics_optimizer_col: "optimizer", perf_col: "perf"})
            metrics_frames.append(tdf)
        metrics_df = pd.concat(metrics_frames, ignore_index=True).dropna()
    elif args.metrics:
        if args.metrics_tag and len(args.metrics_tag) != len(args.metrics):
            raise ValueError("Provide exactly one --metrics-tag per --metrics.")
        for i, m in enumerate(args.metrics):
            tdf = pd.read_csv(m)
            if "step" not in tdf.columns:
                raise ValueError(f"Metrics file missing 'step': {m}")
            perf_col = guess_perf_col(tdf, args.perf_col)
            tag = args.metrics_tag[i] if args.metrics_tag else f"run{i+1}"
            tdf = tdf[["step", perf_col]].rename(columns={perf_col: "perf"})
            tdf["optimizer"] = tag
            metrics_frames.append(tdf)
        metrics_df = pd.concat(metrics_frames, ignore_index=True).dropna()
    else:
        metrics_df = pd.DataFrame(columns=["optimizer", "step", "perf"])

    if not metrics_df.empty:
        metrics_df["optimizer"] = metrics_df["optimizer"].astype(str).str.strip()

    corr_rows = []

    # Per (optimizer, step) join (averaging across layers)
    if not metrics_df.empty and per_ols["step"].notna().any():
        by_os = (per_ols.groupby(["optimizer", "step"], dropna=False)[agg_cols]
                       .mean().reset_index())
        joined = pd.merge(by_os, metrics_df, on=["optimizer", "step"], how="inner")
        joined.to_csv(out_dir / "joined_per_opt_step.csv", index=False)

        # Global correlations
        for c in agg_cols:
            s, p = joined[c], joined["perf"]
            pear = s.corr(p, method="pearson") if s.notna().sum() >= 3 else np.nan
            spear = s.corr(p, method="spearman") if s.notna().sum() >= 3 else np.nan
            corr_rows.append({"scope": "per_opt_step_all", "stat": c, "pearson": pear, "spearman": spear})

        # Per-optimizer correlations
        for opt_name, sub in joined.groupby("optimizer"):
            for c in agg_cols:
                s, p = sub[c], sub["perf"]
                pear = s.corr(p, method="pearson") if s.notna().sum() >= 3 else np.nan
                spear = s.corr(p, method="spearman") if s.notna().sum() >= 3 else np.nan
                corr_rows.append({"scope": f"per_opt_step:{opt_name}", "stat": c, "pearson": pear, "spearman": spear})

        # Plots: perf vs stat, colored by optimizer
        for c in agg_cols:
            plt.figure(figsize=(6,5))
            for opt_name, sub in joined.groupby("optimizer"):
                plt.scatter(sub[c], sub["perf"], alpha=0.85, label=opt_name)
            plt.xlabel(c)
            plt.ylabel("performance (lower is better if this is loss/error)")
            plt.legend()
            plt.title(f"Performance vs {c} (by optimizer)")
            plt.tight_layout()
            plt.savefig(plots_dir / f"perf_vs_{c}.png", dpi=150)
            plt.close()

    # Optimizer-level (mean stats vs best perf)
    if not metrics_df.empty and metrics_df["optimizer"].nunique() >= 2:
        opt_stats = per_ols.groupby("optimizer")[agg_cols].mean().reset_index()
        # If perf is ACCURACY (higher better), change min() → max() here:
        perf_agg = metrics_df.groupby("optimizer")["perf"].min().reset_index().rename(columns={"perf": "perf_best"})
        j2 = pd.merge(opt_stats, perf_agg, on="optimizer", how="inner")
        j2.to_csv(out_dir / "optimizer_agg_join.csv", index=False)

        for c in agg_cols:
            s, p = j2[c], j2["perf_best"]
            pear = s.corr(p, method="pearson") if s.notna().sum() >= 3 else np.nan
            spear = s.corr(p, method="spearman") if s.notna().sum() >= 3 else np.nan
            corr_rows.append({"scope": "optimizer_agg", "stat": c, "pearson": pear, "spearman": spear})

        # Plot optimizer-level
        for c in agg_cols:
            plt.figure(figsize=(6,5))
            for _, row in j2.iterrows():
                plt.scatter(row[c], row["perf_best"])
                plt.annotate(row["optimizer"], (row[c], row["perf_best"]), xytext=(5,5), textcoords="offset points")
            plt.xlabel(c)
            plt.ylabel("best performance (min over steps)")
            plt.title(f"Optimizer-level: best perf vs {c}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"optimizer_bestperf_vs_{c}.png", dpi=150)
            plt.close()

    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(out_dir / "correlations_summary.csv", index=False)

    # Layer summaries by optimizer
    for opt_name, sub in per_ol.groupby("optimizer"):
        s2 = sub.sort_values("layer")
        for c in ["kl_to_fitted_gauss", "bimodality_coeff", "skewness", "kurtosis_pearson"]:
            plt.figure(figsize=(8,4))
            plt.bar(s2["layer"].astype(float), s2[c])
            plt.xlabel("Layer")
            plt.ylabel(c)
            plt.title(f"{opt_name}: {c} by layer (weights from checkpoints)")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{opt_name}_layer_{c}.png", dpi=150)
            plt.close()

    print(f"[OK] Finished. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
