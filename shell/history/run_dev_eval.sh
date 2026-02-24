# # ==== 一次性依赖（装过可跳过）====
# pip install -U "bitsandbytes>=0.44" transformers peft accelerate
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m runners.main rl \
  --model /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --data artifacts/rl/rl_vindr.jsonl \
  --steps 200 \
  --K 3 \
  --max_steps_per_traj 2 \
  --lr 1e-4 \
  --kl 0.02 \
  --ent 0.005 \
  --batch_size 1 \
  --refresh_every 200 \
  --es_patience 100 \
  --es_delta 1e-4 \
  --use_qwen_prior \
  2>&1 | tee -a artifacts/logs/rl_safe_$(date +%Y%m%d-%H%M%S).log

# python runners/eval_baselines.py \
#   --dataset artifacts/rl/rl_vindr.jsonl \
#   --model_name /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
#   --lora_path artifacts/ckpt_rl/best \
#   --mode policy_init \
#   --n_eval 200 --max_steps 3
python -m runners.main eval3 \
  --data artifacts/rl/vindr_200.jsonl \
  --model /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
  --lora artifacts/ckpt_rl/best \
  --max_steps 3 --n_eval 200