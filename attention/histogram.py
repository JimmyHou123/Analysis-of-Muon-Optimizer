import os
import glob
import h5py
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose

# ─── make sure src/ is importable ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.bcat import BCAT
from utils.muon import Muon


def load_config():
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
    model = BCAT(
        cfg.model,
        x_num=cfg.data.x_num,
        max_output_dim=cfg.data.max_output_dimension
    ).to(device)

    # split parameters for Muon vs AdamW
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

    # load the checkpoint at user-provided path
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    model.eval()
    return model


def load_sample(cfg):
    folder = cfg.data.incom_ns_arena_u.folder
    path   = sorted(glob.glob(os.path.join(folder, "*.h5")))[0]
    with h5py.File(path, "r") as f:
        grp = f["test"]
        u, vx, vy, t = grp["u"][:], grp["vx"][:], grp["vy"][:], grp["t"][:]
    u0, vx0, vy0, t0 = u[0], vx[0], vy[0], t[0]
    T, H, W = u0.shape
    u0 = u0[..., None]; vx0 = vx0[..., None]; vy0 = vy0[..., None]
    zeros = np.zeros((T, H, W, 1), dtype=u0.dtype)
    sample = np.concatenate([u0, vx0, vy0, zeros], axis=-1).astype("float32")
    return sample, t0.astype("float32")


def to_tensors(sample, t0, device):
    data  = torch.from_numpy(sample)[None].to(device)      # [1,T,H,W,4]
    times = torch.from_numpy(t0)[None, :, None].to(device) # [1,T,1]
    return data, times


def plot_distributions(model, data, times, layer_idx, out_dir, opt_name):
    # run forward
    with torch.no_grad():
        _ = model("fwd", data=data, times=times, input_len=data.size(1))

    layer     = model.transformer.layers[layer_idx].self_attn
    scores    = layer.last_scores   .flatten().numpy()
    weights   = layer.last_attn     .flatten().numpy()
    out_proj  = layer.last_out_proj .flatten().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # raw Q·K
    axes[0].hist(scores, bins=100, alpha=0.7)
    axes[0].set_title(f"{opt_name} Layer {layer_idx} Raw Q⋅K/√dₕ")
    axes[0].set_xlabel("score"); axes[0].set_ylabel("count")

    # attention weights (nonzero)
    nonzero = weights[weights > 0]
    # plot only the bulk of the distribution up to its 99th percentile
    p99 = np.percentile(nonzero, 99)
    axes[1].hist(nonzero, bins=100, range=(0, p99), alpha=0.7)
    axes[1].set_title(f"{opt_name} Layer {layer_idx} Attention Weights (nonzero)")
    axes[1].set_xlabel("weight"); axes[1].set_ylabel("count")

    # output projection
    axes[2].hist(out_proj, bins=100, alpha=0.7)
    axes[2].set_title(f"{opt_name} Layer {layer_idx} Out_Proj")
    axes[2].set_xlabel("activation"); axes[2].set_ylabel("count")

    plt.suptitle(f"{opt_name} Distributions for Layer {layer_idx}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{opt_name}_layer_{layer_idx}_distributions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # 0) ask for optimizer name
    opt_name   = input("Enter optimizer name (e.g. 'Muon'): ").strip() or "Optimizer"

    # 1) ask for checkpoint path
    ckpt_path  = input("Enter absolute path to checkpoint.pth: ").strip()
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: File not found at {ckpt_path}")
        sys.exit(1)

    # 2) ask for layers
    layers_str = input("Enter layer index or range (e.g. '4' or '2-5'): ").strip()
    if "-" in layers_str:
        lo, hi  = map(int, layers_str.split("-", 1))
        layers  = list(range(lo, hi + 1))
    else:
        layers  = [int(layers_str)]

    # 3) ask for output folder
    out_dir    = input("Enter output folder (default 'histograms'): ").strip() or "histograms"

    # load everything
    cfg       = load_config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(cfg, device, ckpt_path)

    sample, t0 = load_sample(cfg)
    data, times = to_tensors(sample, t0, device)

    # plot per layer
    for L in layers:
        print(f"Plotting distributions for layer {L} → saving into '{out_dir}'")
        plot_distributions(model, data, times, layer_idx=L, out_dir=out_dir, opt_name=opt_name)

    print("Done! Saved histograms for layers:", layers, "in folder:", out_dir)
