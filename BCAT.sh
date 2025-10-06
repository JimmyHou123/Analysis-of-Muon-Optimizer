#!/usr/bin/env bash
python /home/jimmy/bcat_main_server/src/main.py \
  data=fluids_arena \
  data.incom_ns_arena_u.folder=/data/shared/dataset/pdearena/NavierStokes-2D/ \
  optim=adan \
  model.n_layer=12 \
  model.dim_emb=512 \
  model.dim_ffn=2048 \
  model.n_head=8 \
  use_wandb=0 \
  batch_size=16 \
  n_steps_per_epoch=1000 \
  max_epoch=20