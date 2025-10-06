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
named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
adam_keys, muon_params, adamw_params = ["embed"], [], []
for name,p in named:
    if p.ndim>=2 and ".self_attn." in name and not any(k in name for k in adam_keys):
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

ckpt = torch.load("/home/jimmy/bcat_main_server/checkpoint/bcat/muon_ns_attention_52M/checkpoint.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# create dummy input matching your training shapes
bs        = 1
t_num     = cfg.data.t_num
x_num     = cfg.data.x_num
dim       = cfg.data.max_output_dimension
input_len = cfg.input_len
data  = torch.randn(bs, t_num, x_num, x_num, dim, device=device)
times = torch.linspace(0,10,t_num,device=device)[None,:,None]

with torch.no_grad():
    _ = model("fwd", data=data, times=times, input_len=input_len)

all_scores  = []
all_weights = []

for layer in model.transformer.layers:
    scores  = layer.self_attn.last_scores   # on CPU
    weights = layer.self_attn.last_attn     # on CPU
    all_scores .append(scores .flatten())
    all_weights.append(weights.flatten())

all_scores  = torch.cat(all_scores).numpy()
all_weights = torch.cat(all_weights).numpy()

# Separate plots: QK scores vs. attention weights
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Raw QK scores before softmax
ax1.hist(all_scores, bins=50)
ax1.set_title("Histogram of Raw QK Scores")
ax1.set_xlabel("Q⋅Kᵀ / √dₕ")
ax1.set_ylabel("Count")

# Post-softmax attention weights
ax2.hist(all_weights, bins=50, range=(0,1))
ax2.set_title("Histogram of QK-Based Attention Weights")
ax2.set_xlabel("softmax(QKᵀ / √dₕ)")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("qk_scores_and_weights.png", dpi=150)