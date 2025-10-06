python compute_dar_from_checkpoints.py \
  --names Muon Adan AdamW \
  --ckpts /home/jimmy/bcat_main_server/checkpoint/bcat/adamw_ns_52M/checkpoint.pth \
          /home/jimmy/bcat_main_server/checkpoint/bcat/adan_ns_52M/checkpoint.pth \
          /home/jimmy/bcat_main_server/checkpoint/bcat/muon_ns_attention_52M/checkpoint.pth \
  --layers 0-11 \
  --w 1 \
  --out_csv dar_from_tensors.csv \
  --out_dir dar_tensor_plots