CUDA_VISIBLE_DEVICES=0 python -m runners.main rl \
  --model /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --data artifacts/rl/rl_vindr.jsonl \
  --steps 600 \
  --K 3 \
  --lr 5e-6 \
  --kl 0.02 \
  --ent 0.005 \
  --max_steps_per_traj 3
