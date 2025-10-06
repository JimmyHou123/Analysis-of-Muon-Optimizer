#!/usr/bin/env python3
"""
Compute Diagonal-Argmax Rate (DAR_w) directly from raw attention tensors,
using the exact loading & preprocessing flow from heatmap.py.

This:
  • loads Hydra config like heatmap.py
  • builds BCAT the same way
  • loads a checkpoint
  • loads a sample from PDEArena (same H5 flow as heatmap.py)
  • runs one forward pass model("fwd", data=..., times=..., input_len=...)
  • for each layer: grabs self_attn.last_attn, reshapes to time-only attention A_time
  • computes DAR_w per head/layer (w=1 by default)
  • writes dar_metrics.csv and some quick plots

USAGE EXAMPLE (three optimizers):
  python compute_dar_from_checkpoints.py \
    --names Muon Adan AdamW \
    --ckpts /abs/path/to/muon.pth /abs/path/to/adan.pth /abs/path/to/adamw.pth \
    --layers 0-11 \
    --w 1 \
    --out_csv dar_from_tensors.csv \
    --out_dir dar_tensor_plots
"""

import os
import sys
import glob
import h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# ---------- keep these exactly like heatmap.py ----------
from hydra import initialize, compose

# ensure src/ is on PYTHONPATH (same as your heatmap.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.bcat import BCAT
from utils.muon import Muon
# --------------------------------------------------------


# ========== COPY of config/model/data helpers from heatmap.py ==========

def load_config():
    # identical to your heatmap.py
    with initialize(config_path="src/configs", version_base=None):
        return compose(
            config_name="main",
            overrides=[
                "data=fluids_arena",
                "data.incom_ns_arena_u.folder=/data/shared/dataset/pdearena/NavierStokes-2D/",
                "optim=muon",
                "model.n_layer=12",
                "model.dim_emb=512",
                "model.dim_ffn=2048",
                "model.n_head=8",
                "model.kv_cache=false",
                "use_wandb=1",
                "batch_size=1",
                "n_steps_per_epoch=1000",
                "max_epoch=20",
            ],
        )


def build_model(cfg, device, ckpt_path):
    # identical structure to your heatmap.py (build BCAT + construct Muon optimizer)
    model = BCAT(
        cfg.model,
        x_num=cfg.data.x_num,
        max_output_dim=cfg.data.max_output_dimension
    ).to(device)

    # split params for optimizer (as in heatmap.py)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    muon_params, adamw_params = [], []
    for n, p in named:
        if p.ndim >= 2 and ".self_attn." in n and "embed" not in n:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optim = Muon(
        lr=cfg.optim.lr,
        wd=cfg.optim.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
        adamw_betas=(0.9, cfg.optim.get("beta2", 0.95)),
        adamw_eps=cfg.optim.get("eps", 1e-8),
    )

    # load checkpoint from user-provided path (same as heatmap.py)
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    model.eval()
    return model


def load_sample(cfg):
    # identical to heatmap.py
    data_folder = cfg.data.incom_ns_arena_u.folder
    fp = sorted(glob.glob(os.path.join(data_folder, "*.h5")))[0]
    with h5py.File(fp, "r") as f:
        grp = f["test"]
        u, vx, vy, t = grp["u"][:], grp["vx"][:], grp["vy"][:], grp["t"][:]
    u0, vx0, vy0, t0 = u[0], vx[0], vy[0], t[0]
    T, H, W = u0.shape
    u0 = u0[..., None]; vx0 = vx0[..., None]; vy0 = vy0[..., None]
    zeros = np.zeros((T, H, W, 1), dtype=u0.dtype)
    sample = np.concatenate([u0, vx0, vy0, zeros], axis=-1).astype("float32")
    return sample, t0.astype("float32")


def to_tensors(sample, t0, device):
    # identical to heatmap.py
    data = torch.from_numpy(sample)[None].to(device)
    times = torch.from_numpy(t0)[None,:,None].to(device)
    return data, times

# =======================================================================


# --------------- DAR metric (per head) ---------------

@torch.no_grad()
def diagonal_argmax_rate(attn_2d: torch.Tensor, w: int = 1) -> float:
    """
    attn_2d: [T, T] attention probabilities for one head (rows ~sum to 1).
    DAR_w = fraction of rows whose argmax lies within ±w of the diagonal.
    """
    assert attn_2d.dim() == 2, f"Expect [T,T], got {tuple(attn_2d.shape)}"
    Tq, Tk = attn_2d.shape
    j_star = torch.argmax(attn_2d, dim=1)             # [T]
    i = torch.arange(Tq, device=attn_2d.device)       # [T]
    near = (j_star - i).abs() <= w                    # [T] boolean
    return float(near.float().mean().item())


def compute_dar_for_layer_time_heads(A_time: torch.Tensor, w: int) -> List[float]:
    """
    A_time: [H, T, T] (time-only attention per head for a given layer)
    returns list length H of DAR_w per head.
    """
    assert A_time.dim() == 3, f"Expect [H,T,T], got {tuple(A_time.shape)}"
    H = A_time.shape[0]
    vals = []
    for h in range(H):
        vals.append(diagonal_argmax_rate(A_time[h], w=w))
    return vals


# --------------- Collect time-only attention from the model ---------------

def collect_time_only_attention(model, cfg) -> Dict[int, torch.Tensor]:
    """
    After a forward pass, read each layer's attention like heatmap.py:
       A = layer.self_attn.last_attn[0]            # [H, S, S]
       reshape -> [H, T_attn, spatial, T_attn, spatial]
       average over spatial -> A_time [H, T_attn, T_attn]
    Returns dict: layer_idx -> A_time [H, T, T]
    """
    out: Dict[int, torch.Tensor] = {}
    n_layers = len(model.transformer.layers)
    for L in range(n_layers):
        layer = model.transformer.layers[L]
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "last_attn"):
            continue
        A = layer.self_attn.last_attn[0]             # [H, S, S]
        n_head, S, _ = A.shape

        patch_num = cfg.model.patch_num
        spatial   = patch_num * patch_num
        T_attn    = S // spatial
        assert T_attn * spatial == S, f"S={S} not = T_attn*spatial={T_attn*spatial}"

        A = A.view(n_head, T_attn, spatial, T_attn, spatial)  # [H, T, spatial, T, spatial]
        A_time = A.mean(dim=(2, 4))                           # -> [H, T, T]
        out[L] = A_time.detach().cpu()
    return out


# --------------- Main ---------------

def parse_layers(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        lo, hi = map(int, spec.split("-", 1))
        return list(range(lo, hi + 1))
    else:
        return [int(x) for x in spec.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True, help="Optimizer names, e.g. Muon Adan AdamW")
    ap.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths, same order as --names")
    ap.add_argument("--layers", type=str, default="0-11", help="Layer index or range, e.g. '4' or '2-5' or '0-11'")
    ap.add_argument("--w", type=int, default=1, help="DAR band half-width (±w around diagonal)")
    ap.add_argument("--out_csv", type=str, default="dar_from_tensors.csv")
    ap.add_argument("--out_dir", type=str, default="dar_tensor_plots")
    args = ap.parse_args()

    assert len(args.names) == len(args.ckpts), "Provide one checkpoint per name."

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load config (same as heatmap.py)
    cfg = load_config()

    # 2) load one sample & preprocess (same as heatmap.py)
    sample, t0  = load_sample(cfg)
    data, times = to_tensors(sample, t0, device)

    rows = []

    # 3) loop over (name, ckpt)
    for name, ckpt_path in zip(args.names, args.ckpts):
        print(f"\n=== {name}: {ckpt_path}")
        model = build_model(cfg, device, ckpt_path)

        # one forward pass exactly like heatmap.py
        with torch.no_grad():
            _ = model("fwd", data=data, times=times, input_len=data.size(1))

        # gather time-only attention per layer
        time_attn_by_layer = collect_time_only_attention(model, cfg)

        target_layers = set(parse_layers(args.layers))
        for L, A_time in time_attn_by_layer.items():
            if L not in target_layers:
                continue
            # A_time: [H, T, T]
            dar_vals = compute_dar_for_layer_time_heads(A_time, w=args.w)
            for h, v in enumerate(dar_vals):
                rows.append({
                    "optimizer": name,
                    "layer": L,
                    "head": h,
                    "DAR_w": args.w,
                    "DAR": float(v),
                })

    if not rows:
        print("No DAR rows computed — check --layers range and checkpoint contents.")
        return

    # 4) save csv
    df = pd.DataFrame(rows)
    df = df.sort_values(["optimizer", "layer", "head"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved DAR metrics -> {args.out_csv}")

    # 5) quick plots (mean DAR vs layer, head×layer map) per optimizer
    df_layer = df.groupby(["optimizer", "layer"])["DAR"].mean().reset_index()

    # line plots
    for opt in df["optimizer"].unique():
        sub = df_layer[df_layer["optimizer"] == opt].sort_values("layer")
        plt.figure(figsize=(6, 4))
        plt.plot(sub["layer"], sub["DAR"], marker="o")
        plt.xlabel("Layer")
        plt.ylabel(f"Mean DAR (w={args.w})")
        plt.title(f"{opt}: Mean Diagonal-Argmax Rate by Layer")
        out_path = os.path.join(args.out_dir, f"{opt.lower()}_mean_dar_by_layer.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"Saved {out_path}")

    # head×layer maps
    for opt in df["optimizer"].unique():
        sub = df[df["optimizer"] == opt]
        layers_sorted = sorted(sub["layer"].unique())
        heads_sorted  = sorted(sub["head"].unique())
        M = np.zeros((len(heads_sorted), len(layers_sorted)), dtype=np.float32)
        for i, h in enumerate(heads_sorted):
            for j, L in enumerate(layers_sorted):
                val = sub[(sub["head"] == h) & (sub["layer"] == L)]["DAR"].mean()
                M[i, j] = 0.0 if np.isnan(val) else val
        plt.figure(figsize=(6, 4))
        plt.imshow(M, aspect="auto", origin="lower")
        plt.colorbar(label=f"DAR (w={args.w})")
        plt.yticks(range(len(heads_sorted)), heads_sorted)
        plt.xticks(range(len(layers_sorted)), layers_sorted)
        plt.xlabel("Layer")
        plt.ylabel("Head")
        plt.title(f"{opt}: DAR Head×Layer")
        out_path = os.path.join(args.out_dir, f"{opt.lower()}_dar_head_by_layer.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()