# # ZS（纯语言）
# python runners/eval_baselines.py --dataset artifacts/rl/rl_vindr.jsonl \
#   --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct --mode zs --n_eval 200 --max_steps 3

# # ZS+Tool（允许CHECK）
# python runners/eval_baselines.py --dataset artifacts/rl/rl_vindr.jsonl \
#   --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct --mode zstool --n_eval 200 --max_steps 3

# # Policy(init)（允许CHECK）
# python runners/eval_baselines.py --dataset artifacts/rl/rl_vindr.jsonl \
#   --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct --mode policy_init --n_eval 200 --max_steps 3

# Policy(+LoRA)
python runners/eval_baselines.py --dataset artifacts/rl/rl_vindr.jsonl \
  --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --lora_path artifacts/ckpt_rl/best \
  --mode policy_init --n_eval 200 --max_steps 3
