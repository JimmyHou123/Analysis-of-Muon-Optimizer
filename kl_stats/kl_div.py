import os
import glob
import h5py
import sys
import csv
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra import initialize, compose

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
                # The cfg's 'optim' doesn't matter for inference, but we keep it to match your original flow
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

    # split parameters for Muon vs AdamW (not needed for forward pass)
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
    if "optimizer" in ckpt:
        try:
            optim.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    model.eval()
    return model


def load_sample(cfg):
    folder = cfg.data.incom_ns_arena_u.folder
    paths  = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    path   = paths[0]
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


def _histogram(x: np.ndarray, bins: int = 256, range_: Tuple[float, float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized histogram probs and bin edges."""
    probs, edges = np.histogram(x, bins=bins, range=range_, density=False)
    probs = probs.astype(np.float64)
    total = probs.sum()
    if total == 0:
        # Degenerate (e.g., all NaNs) – return uniform-ish to avoid blowups
        probs = np.ones_like(probs) / len(probs)
    else:
        probs /= total
    return probs, edges


def _gaussian_bin_probs(mu: float, sigma: float, edges: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Gaussian bin probabilities by CDF at edges."""

    from scipy.stats import norm
    
    if not np.isfinite(sigma) or sigma <= 0:
        # fallback to near-delta at mu: place mass to closest bin center
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = int(np.argmin(np.abs(centers - mu)))
        q = np.full_like(centers, eps, dtype=np.float64)
        q[idx] = 1.0 - eps * (len(centers) - 1)
        return q

    def _cdf(z, mu, sigma):
        return norm.cdf(z, loc=mu, scale=sigma)

    cdf_vals = _cdf(edges, mu, sigma)
    q = np.diff(cdf_vals)
    q = np.clip(q, eps, 1.0)  # ensure strictly positive
    q /= q.sum()
    return q


def kl_divergence_to_fitted_gaussian(x: np.ndarray, bins: int = 256, clip_pct: float | None = None) -> float:
    """KL(P || Q) where P = empirical histogram of x, Q = Gaussian fitted to x (mu, sigma)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")

    if clip_pct is not None:
        lo = np.nanpercentile(x, 100 - clip_pct)
        hi = np.nanpercentile(x, clip_pct)
        if lo < hi:
            x = x[(x >= lo) & (x <= hi)]

    mu  = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    if std > 0:
        lo, hi = mu - 4.0 * std, mu + 4.0 * std
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5

    P, edges = _histogram(x, bins=bins, range_=(lo, hi))
    Q = _gaussian_bin_probs(mu, std, edges)

    # KL(P||Q) = sum P_i log(P_i / Q_i)
    kl = float(np.sum(P * (np.log(P + 1e-12) - np.log(Q + 1e-12))))
    return kl


def kl_for_weights_with_logit(weights: np.ndarray, bins: int = 256, eps: float = 1e-7) -> float:
    """
    Map weights in [0,1] to R via logit after clamping (to avoid infinities),
    then compute KL to fitted Gaussian on the transformed domain.
    """
    w = weights[np.isfinite(weights)]
    if w.size == 0:
        return float("nan")
    # keep ONLY (0,1), discard exact zeros from sparse attention
    w = w[(w > 0.0) & (w < 1.0)]
    if w.size == 0:
        return float("nan")
    w = np.clip(w, eps, 1.0 - eps)
    z = np.log(w) - np.log(1.0 - w)  # logit
    return kl_divergence_to_fitted_gaussian(z, bins=bins, clip_pct=None)


def extract_layer_arrays(model: BCAT, data: torch.Tensor, times: torch.Tensor, layer_idx: int) -> Dict[str, np.ndarray]:
    """
    Runs a forward pas, then extracts arrays needed for statistics.
    Assumes your attention layer stores: last_scores, last_attn, last_out_proj.
    """
    with torch.no_grad():
        _ = model("fwd", data=data, times=times, input_len=data.size(1))

    layer = model.transformer.layers[layer_idx].self_attn

    # The following attributes are assumed by your original script
    scores   = getattr(layer, "last_scores", None)
    attn     = getattr(layer, "last_attn", None)
    out_proj = getattr(layer, "last_out_proj", None)

    arrays = {}
    if scores is not None:
        arrays["scores"] = scores.flatten().detach().cpu().numpy()
    if attn is not None:
        arrays["weights"] = attn.flatten().detach().cpu().numpy()
    if out_proj is not None:
        arrays["out_proj"] = out_proj.flatten().detach().cpu().numpy()
    return arrays


def compute_kls_for_layer(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute KL divergences to fitted Gaussian for each array type.
    - 'scores' and 'out_proj' are on R already → direct KL to Gaussian.
    - 'weights' is on [0,1] → logit transform then KL to Gaussian.
    """
    out: Dict[str, float] = {}
    if "scores" in arrays:
        # Clip tails a bit to reduce plot-driven outliers; 99.5% keeps meaningful deviations
        out["KL_scores"] = kl_divergence_to_fitted_gaussian(arrays["scores"], bins=256, clip_pct=99.5)
    if "weights" in arrays:
        out["KL_weights_logit"] = kl_for_weights_with_logit(arrays["weights"], bins=256)
    if "out_proj" in arrays:
        out["KL_out_proj"] = kl_divergence_to_fitted_gaussian(arrays["out_proj"], bins=256, clip_pct=99.5)
    return out


def plot_grouped_bars(per_layer_results: Dict[str, Dict[int, Dict[str, float]]],
                      out_dir: str,
                      stat_key: str,
                      title_suffix: str):
    """
    per_layer_results: {optimizer_name: {layer_idx: {KL_scores:val, ...}}}
    """
    optimizers = list(per_layer_results.keys())
    if not optimizers:
        return
    layers = sorted({L for opt in optimizers for L in per_layer_results[opt].keys()})
    if not layers:
        return

    # data matrix: rows=layers, cols=optimizers
    vals = []
    for L in layers:
        row = []
        for opt in optimizers:
            v = per_layer_results[opt].get(L, {}).get(stat_key, np.nan)
            row.append(v)
        vals.append(row)
    vals = np.array(vals, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(layers)), 5))
    x = np.arange(len(layers))
    width = 0.7 / max(1, len(optimizers))
    for j, opt in enumerate(optimizers):
        ax.bar(x + j * width, vals[:, j], width, label=opt)

    ax.set_xticks(x + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels([str(L) for L in layers], rotation=0)
    ax.set_ylabel("KL(P || fitted Gaussian)")
    ax.set_title(f"{stat_key} per layer {title_suffix}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{stat_key}_grouped_bar.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    opt_names = ["AdamW", "Adan", "Muon"]

    ckpt_paths = [
        "/home/jimmy/bcat_main_server/checkpoint/bcat/adamw_ns_52M/checkpoint.pth",
        "/home/jimmy/bcat_main_server/checkpoint/bcat/adan_ns_52M/checkpoint.pth",
        "/home/jimmy/bcat_main_server/checkpoint/bcat/muon_ns_attention_52M/checkpoint.pth",
    ]

    # Layers 0-11 inclusive
    layers = list(range(0, 12))

    # Output folder
    out_dir = "kl_stats"
    os.makedirs(out_dir, exist_ok=True)

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample, t0 = load_sample(cfg)
    data, times = to_tensors(sample, t0, device)

    # Results: {optimizer_name: {layer_idx: {KL_scores:..., KL_weights_logit:..., KL_out_proj:...}}}
    per_layer_results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for opt_name, ckpt_path in zip(opt_names, ckpt_paths):
        print(f"\n===== Processing {opt_name} =====")
        model = build_model(cfg, device, ckpt_path)

        per_layer_results[opt_name] = {}
        for L in layers:
            print(f"  Computing KLs for layer {L}...")
            arrays = extract_layer_arrays(model, data, times, layer_idx=L)
            kls = compute_kls_for_layer(arrays)
            per_layer_results[opt_name][L] = kls

    # Save CSV
    csv_path = os.path.join(out_dir, "kl_divergences.csv")
    fieldnames = ["optimizer", "layer", "KL_scores", "KL_weights_logit", "KL_out_proj"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for opt_name in opt_names:
            layer_map = per_layer_results.get(opt_name, {})
            for L in sorted(layer_map.keys()):
                row = {"optimizer": opt_name, "layer": L}
                row.update({k: layer_map[L].get(k, np.nan) for k in fieldnames if k.startswith("KL_")})
                writer.writerow(row)

    print(f"\nSaved KL divergences to: {csv_path}")

    # Plot grouped bars per stat across layers
    for stat_key in ["KL_scores", "KL_weights_logit", "KL_out_proj"]:
        plot_grouped_bars(per_layer_results, out_dir, stat_key, title_suffix=f"(3 optimizers)")

    print("Done! Generated CSV and bar plots in:", out_dir)
