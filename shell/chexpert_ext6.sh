#!/usr/bin/env bash
# Extended 6 ablations on the cross-dataset split (CheXpert -> VinDr6 50p/50n)
# 和你原来在 rl/vindr_200.jsonl 上的 ext-6 是同一套路，只是把数据集、out_dir 换成 transfer 版本
set -euo pipefail

# ============= configurable =============
DATA="artifacts/transfer/chexpert_vindr6_50p50n.jsonl"
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
LORA="artifacts/ckpt_rl/best"
MAX_STEPS=3
N_EVAL=999999       # 和你上面 eval3 / eval_baselines 保持一致，跑满
OUT_ROOT="artifacts/eval_ext_transfer"
CLAIM_GAMMA=1.0
ETA=0.5             # 跟原来扩展6组保持
# =======================================

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT}"

{
  echo "=== EXT-6 (TRANSFER) | $(date) ==="
  echo "DATA=${DATA}"
  echo "MODEL=${MODEL}"
  echo "LORA=${LORA}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "N_EVAL=${N_EVAL}"
  echo "CLAIM_GAMMA=${CLAIM_GAMMA}"
  echo "ETA=${ETA}"
} | tee -a "${OUT}/MANIFEST.txt"

# 1) Policy(init) + Prior-mix  (tool:prior, gate:mix)
echo "[T1] Policy(init)+Prior-mix — tool:prior, gate:mix, eta=${ETA}"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" \
  --mode policy_init \
  --tool_mode prior \
  --gate mix --eta "${ETA}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__prior-mix"

# 2) Policy(+LoRA) + Prior-mix
echo "[T2] Policy(lora)+Prior-mix — tool:prior, gate:mix, eta=${ETA}"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init \
  --tool_mode prior \
  --gate mix --eta "${ETA}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__prior-mix"

# 3) Policy(init) + KBTool-mix  (tool:kbcs, calibrate, gate:mix)
echo "[T3] Policy(init)+KBTool-mix — tool:kbcs, calibrate, gate:mix, eta=${ETA}"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" \
  --mode policy_init \
  --tool_mode kbcs --calibrate_tool \
  --gate mix --eta "${ETA}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__kbcs-mix"

# 4) Policy(+LoRA) + KBTool-mix
echo "[T4] Policy(lora)+KBTool-mix — tool:kbcs, calibrate, gate:mix, eta=${ETA}"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init \
  --tool_mode kbcs --calibrate_tool \
  --gate mix --eta "${ETA}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__kbcs-mix"

# 5) Policy(init) — noCHECK  (tool:none, disable_check)
echo "[T5] Policy(init)-noCHECK — tool:none, disable_check"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" \
  --mode policy_init \
  --tool_mode none \
  --gate mix --eta "${ETA}" \
  --no_check \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__noCHECK"

# 6) Policy(+LoRA) — noCHECK
echo "[T6] Policy(lora)-noCHECK — tool:none, disable_check"
python runners/eval_baselines.py \
  --dataset "${DATA}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init \
  --tool_mode none \
  --gate mix --eta "${ETA}" \
  --no_check \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" \
  --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__noCHECK"

# index
{
  echo
  echo "=== OUTPUT INDEX (TRANSFER) ==="
  echo "Policy(init)__prior-mix:   ${OUT}/Policy-init__prior-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__prior-mix:   ${OUT}/Policy-lora__prior-mix/Policy(init)/metrics.json"
  echo "Policy(init)__kbcs-mix:    ${OUT}/Policy-init__kbcs-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__kbcs-mix:    ${OUT}/Policy-lora__kbcs-mix/Policy(init)/metrics.json"
  echo "Policy(init)__noCHECK:     ${OUT}/Policy-init__noCHECK/Policy(init)/metrics.json"
  echo "Policy(lora)__noCHECK:     ${OUT}/Policy-lora__noCHECK/Policy(init)/metrics.json"
} | tee -a "${OUT}/MANIFEST.txt"

echo
echo "[DONE] ext-6 (transfer) finished. Batch dir: ${OUT}"
