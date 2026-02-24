DATA200="artifacts/rl/vindr_200.jsonl"
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
LORA="artifacts/ckpt_rl/best"
MAX_STEPS=3
N_EVAL=200
OUT_ROOT="artifacts/eval_ext"
CLAIM_GAMMA=1.0

#!/usr/bin/env bash
set -euo pipefail

DATA200="artifacts/rl/vindr_200.jsonl"
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
MAX_STEPS=3
N_EVAL=200
OUT_ROOT="artifacts/eval_ext"
CLAIM_GAMMA=1.0

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT}"

echo "[E1] Policy(init)+Prior-mix — tool:prior, mix, WITH TRAJ"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init \
  --tool_mode prior \
  --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} \
  --claim_gamma ${CLAIM_GAMMA} \
  --debug_traj \
  --out_dir "${OUT}/Policy-init__prior-mix"

# 这一行是关键：把 SimpleEnv 写出的 rl_traj.jsonl 收集到 prior-mix 目录里
# 注意 Policy(init) 这一层目录是 eval_baselines 里加的 tag
mv artifacts/debug/rl_traj.jsonl \
   "${OUT}/Policy-init__prior-mix/Policy(init)/traj.jsonl"

echo
echo "Prior-mix metrics : ${OUT}/Policy-init__prior-mix/Policy(init)/metrics.json"
echo "Prior-mix preds   : ${OUT}/Policy-init__prior-mix/Policy(init)/preds.csv"
echo "Prior-mix traj    : ${OUT}/Policy-init__prior-mix/Policy(init)/traj.jsonl"
