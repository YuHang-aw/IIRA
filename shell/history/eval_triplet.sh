# #!/usr/bin/env bash
# # shell/eval_triplet.sh
# set -e

# DATA=${1:-"artifacts/rl/rl_vindr.jsonl"}    # 你的评测集
# MODEL=${2:-"/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"}    # 模型名称/路径
# LORA=${3:-""}                       # Policy(init) 用的 LoRA（可留空）

# export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
# # 可选：静音可选 CUDA 扩展
# export XFORMERS_DISABLED=1
# export FLASH_ATTENTION_DISABLE=1


# python -m runners.main eval3 \
#   --data "$DATA" \
#   --model "$MODEL" \
#   --max_steps 3
# ZS
python runners/eval_baselines.py \
  --dataset artifacts/rl/rl_vindr.jsonl \
  --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --mode zs \
  --n_eval 200 --max_steps 3

# ZS+QwenPrior
python runners/eval_baselines.py \
  --dataset artifacts/rl/rl_vindr.jsonl \
  --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --mode zstool \
  --n_eval 200 --max_steps 3

# Policy(init) —— 不带 RL LoRA（严格按你定义）
python runners/eval_baselines.py \
  --dataset artifacts/rl/rl_vindr.jsonl \
  --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --mode policy_init \
  --n_eval 200 --max_steps 3

# # 我们的方法：Policy(init)+LoRA
# python runners/eval_baselines.py \
#   --dataset artifacts/rl/vindr_200.jsonl \
#   --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
#   --mode policy_init --n_eval 200 --max_steps 3