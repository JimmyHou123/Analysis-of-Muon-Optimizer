import os
import sys
import torch
from hydra import initialize, compose
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.bcat import BCAT
from utils.muon import Muon
import matplotlib.pyplot as plt

# 1) Build your Hydra config (must match training)
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

# 2) Instantiate the BCAT model
model = BCAT(
    cfg.model,
    x_num          = cfg.data.x_num,
    max_output_dim = cfg.data.max_output_dimension
)

# 3) Split parameters and build the Muon optimizer
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
adam_keys, muon_params, adamw_params = ["embed"], [], []
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
    adamw_betas  = (0.9, cfg.optim.get("beta2",0.95)),
    adamw_eps    = cfg.optim.get("eps",1e-8),
)

# 4) Load checkpoint
ckpt = torch.load(
    "/home/jimmy/bcat_main_server/checkpoint/bcat/muon_ns_attention_52M/checkpoint.pth",
    map_location="cpu"
)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5) Dummy input
bs        = 1
t_num     = cfg.data.t_num
x_num     = cfg.data.x_num
dim       = cfg.data.max_output_dimension
input_len = cfg.input_len

data  = torch.randn(bs, t_num, x_num, x_num, dim, device=device)
times = torch.linspace(0, 10, t_num, device=device)[None, :, None]

with torch.no_grad():
    _ = model("fwd", data=data, times=times, input_len=input_len)

# 6) Collect distributions
all_raw_vs    = []
all_out_projs = []

for layer in model.transformer.layers:
    attn = layer.self_attn

    # raw V before mixing
    raw_v = attn.last_raw_v       # [1, heads, L, head_dim]
    all_raw_vs.append(raw_v.flatten())

    # final out_proj activations
    out_proj = attn.last_out_proj  # [1, L, embed_dim]
    all_out_projs.append(out_proj.flatten())

all_raw_vs    = torch.cat(all_raw_vs)   .numpy()
all_out_projs = torch.cat(all_out_projs).numpy()

# 7) Plot raw V distribution
plt.figure(figsize=(6,4))
plt.hist(all_raw_vs, bins=50)
plt.title("Histogram of Raw V Values")
plt.xlabel("V component value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_raw_v.png", dpi=150)

# 8) Plot out_proj activations distribution
plt.figure(figsize=(6,4))
plt.hist(all_out_projs, bins=50)
plt.title("Histogram of out_proj Activations")
plt.xlabel("out_proj output value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_out_proj.png", dpi=150)