import os
import glob
import h5py
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose

# ensure src/ is on PYTHONPATH
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


def build_model(cfg, device):
    model = BCAT(cfg.model,
                 x_num=cfg.data.x_num,
                 max_output_dim=cfg.data.max_output_dimension).to(device)
    # split params for optimizer
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
    ckpt = torch.load(
        "/home/jimmy/bcat_main_server/checkpoint/bcat/muon_ns_attention_52M/checkpoint.pth",
        map_location="cpu",
    )
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    model.eval()
    return model


def load_sample(cfg):
    data_folder = cfg.data.incom_ns_arena_u.folder
    fp = sorted(glob.glob(os.path.join(data_folder, "*.h5")))[0]
    with h5py.File(fp, "r") as f:
        grp = f["test"]
        u, vx, vy, t = grp["u"][:], grp["vx"][:], grp["vy"][:], grp["t"][:]
    # pick first trajectory
    u0, vx0, vy0, t0 = u[0], vx[0], vy[0], t[0]
    T, H, W = u0.shape
    # build 4-channel sample
    u0 = u0[..., None]; vx0 = vx0[..., None]; vy0 = vy0[..., None]
    zeros = np.zeros((T, H, W, 1), dtype=u0.dtype)
    sample = np.concatenate([u0, vx0, vy0, zeros], axis=-1).astype("float32")
    return sample, t0.astype("float32")


def to_tensors(sample, t0, device):
    data = torch.from_numpy(sample)[None].to(device)        # [1,T,H,W,4]
    times = torch.from_numpy(t0)[None,:,None].to(device)    # [1,T,1]
    return data, times


def plot_layer_attention(model, cfg, data, times, layer_idx, out_dir):
    # forward
    with torch.no_grad():
        _ = model("fwd", data=data, times=times, input_len=data.size(1))

    A = model.transformer.layers[layer_idx].self_attn.last_attn[0]  # [n_head, S, S]
    n_head, S, _ = A.shape

    patch_num = cfg.model.patch_num
    spatial   = patch_num * patch_num
    T_attn    = S // spatial
    assert T_attn * spatial == S, f"S={S} not = T_attn*spatial={T_attn*spatial}"

    # reshape & average
    A = A.view(n_head, T_attn, spatial, T_attn, spatial)
    A_time = A.mean(dim=(2,4))

    # plot
    fig, axes = plt.subplots(1, n_head, figsize=(3*n_head,3), constrained_layout=True)
    vmax = A_time.max().item()
    for h, ax in enumerate(axes):
        im = ax.imshow(A_time[h].cpu(),
                       vmin=0, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"L{layer_idx} H{h}")
        ax.set_xlabel("t"); ax.set_ylabel("t")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("attn weight")
    fig.suptitle(f"Layer {layer_idx} Time‐Only Attention")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/layer_{layer_idx:02d}_time_heatmaps.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # prompt the user directly
    layers_str = input("​Enter layer index or range (e.g. '4' or '2-5'): ").strip()
    out_dir    = input("​Enter output folder (default ‘heatmaps’): ").strip() or "heatmaps"

    # parse layers_str
    if "-" in layers_str:
        lo, hi = map(int, layers_str.split("-", 1))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(layers_str)]

    # load model & data as before...
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg, device)
    sample, t0 = load_sample(cfg)
    data, times = to_tensors(sample, t0, device)

    # generate and save one heatmap per requested layer
    for L in layers:
        print(f"→ plotting layer {L}")
        plot_layer_attention(model, cfg, data, times, layer_idx=L, out_dir=out_dir)

    print(f"Done! Saved heatmaps for layers {layers} into “{out_dir}/”.")
