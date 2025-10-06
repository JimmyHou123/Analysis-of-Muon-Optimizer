import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch
from hydra import initialize, compose
from models.bcat import BCAT
from utils.muon import Muon
from omegaconf import OmegaConf

# ──────────────────────────────────────────────────────────────────────────────
# 1) cd into repo root so Hydra and imports work
# ──────────────────────────────────────────────────────────────────────────────
# os.chdir("/content/drive/MyDrive/bcat-main")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Build the exact same Hydra config you used in training
# ──────────────────────────────────────────────────────────────────────────────
with initialize(config_path="src/configs", version_base=None):
    cfg = compose(
        config_name="main",
        overrides=[
            "data=fluids_arena",
            "data.incom_ns_arena_u.folder=/tmp/pde_h5_files",
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

# ──────────────────────────────────────────────────────────────────────────────
# 3) Instantiate BCAT + Muon optimizer exactly as in training
# ──────────────────────────────────────────────────────────────────────────────
model = BCAT(
    cfg.model,
    x_num          = cfg.data.x_num,
    max_output_dim = cfg.data.max_output_dimension
)

# split parameters into Muon vs AdamW
named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
adam_keys    = ["embed"]
muon_params, adamw_params = [], []
for name, p in named:
    if p.ndim >= 2 and ".self_attn." in name and not any(k in name for k in adam_keys):
        muon_params.append(p)
    else:
        adamw_params.append(p)

optimizer = Muon(
    lr           = cfg.optim.lr,
    wd           = cfg.optim.weight_decay,
    muon_params  = muon_params,
    adamw_params = adamw_params,
    adamw_betas  = (0.9, cfg.optim.get("beta2", 0.95)),
    adamw_eps    = cfg.optim.get("eps", 1e-8),
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Load your checkpoint
# ──────────────────────────────────────────────────────────────────────────────
ckpt_path = "/home/jimmy/bcat_main_server/checkpoint/bcat/on0ad3jawp/checkpoint.pth"
ckpt = torch.load(
    ckpt_path,
    map_location="cpu")

model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])

start_epoch  = ckpt.get("epoch", 0)
best_metrics = ckpt.get("best_metrics", {})
print(f"✅  Resumed from epoch {start_epoch} | best_metrics={best_metrics}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Switch to eval, move to GPU, and grab dtype/device
# ──────────────────────────────────────────────────────────────────────────────
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

param = next(model.parameters())
dtype, device = param.dtype, param.device
print(f"Model is on {device} and dtype={dtype}")
print("Model parameter dtype:", param.dtype)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Print the learned attention weight matrices
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Attention weight matrices ===")
for name, param in model.named_parameters():
    if ".self_attn." in name:
        print(f"{name:60s}  {tuple(param.shape)}")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Build dummy inputs in the same dtype/device, forward once
# ──────────────────────────────────────────────────────────────────────────────
bs        = 1
t_num     = cfg.data.t_num
input_len = cfg.input_len
x_num     = cfg.data.x_num
dim       = cfg.data.max_output_dimension

data  = torch.randn(bs, t_num, x_num, x_num, dim, dtype=dtype, device=device)
times = torch.linspace(0, 10, t_num, dtype=dtype, device=device)[None, :, None]
print("data dtype:", data.dtype)
print("times dtype:", times.dtype)

with torch.no_grad():
    _ = model("fwd", data=data, times=times, input_len=input_len)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Now each layer.self_attn.last_attn is populated—print their shapes
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Attention scores from last forward ===")
for i, layer in enumerate(model.transformer.layers):
    scores = layer.self_attn.last_attn  # (bs, n_head, seq_len, seq_len)
    print(f"Layer {i:2d}  scores shape = {tuple(scores.shape)}")
    # optionally: print(scores)