#!/usr/bin/env bash
# Extended 6 ablations (prior-mix / kbcs-mix / noCHECK) for init & LoRA
set -euo pipefail

# ============= configurable =============
DATA200="artifacts/rl/vindr_200.jsonl"
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
LORA="artifacts/ckpt_rl/best"
MAX_STEPS=3
N_EVAL=200
OUT_ROOT="artifacts/eval_ext"
CLAIM_GAMMA=1.0
# =======================================

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT}"

{
  echo "=== EXT-6 | $(date) ==="
  echo "DATA200=${DATA200}"
  echo "MODEL=${MODEL}"
  echo "LORA=${LORA}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "N_EVAL=${N_EVAL}"
  echo "CLAIM_GAMMA=${CLAIM_GAMMA}"
} | tee -a "${OUT}/MANIFEST.txt"

# 7) Policy(init)+Prior-mix
echo "[E1] Policy(init)+Prior-mix — tool:prior, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode prior --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-init__prior-mix"

# 8) Policy(+LoRA)+Prior-mix
echo "[E2] Policy(lora)+Prior-mix — tool:prior, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode prior --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-lora__prior-mix"

# 9) Policy(init)+KBTool-mix（不用门控）
echo "[E3] Policy(init)+KBTool-mix — tool:kbcs, calibrate, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-init__kbcs-mix"

# 10) Policy(+LoRA)+KBTool-mix（不用门控）
echo "[E4] Policy(lora)+KBTool-mix — tool:kbcs, calibrate, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-lora__kbcs-mix"

# 11) Policy(init) — noCHECK
echo "[E5] Policy(init)-noCHECK — tool:none, disable_check"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode none --gate mix --eta 0.5 \
  --no_check \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-init__noCHECK"

# 12) Policy(+LoRA) — noCHECK
echo "[E6] Policy(lora)-noCHECK — tool:none, disable_check"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode none --gate mix --eta 0.5 \
  --no_check \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-lora__noCHECK"

# index
{
  echo
  echo "=== OUTPUT INDEX ==="
  echo "Policy(init)__prior-mix:   ${OUT}/Policy-init__prior-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__prior-mix:   ${OUT}/Policy-lora__prior-mix/Policy(init)/metrics.json"
  echo "Policy(init)__kbcs-mix:    ${OUT}/Policy-init__kbcs-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__kbcs-mix:    ${OUT}/Policy-lora__kbcs-mix/Policy(init)/metrics.json"
  echo "Policy(init)__noCHECK:     ${OUT}/Policy-init__noCHECK/Policy(init)/metrics.json"
  echo "Policy(lora)__noCHECK:     ${OUT}/Policy-lora__noCHECK/Policy(init)/metrics.json"
} | tee -a "${OUT}/MANIFEST.txt"

echo
echo "[DONE] ext-6 finished. Batch dir: ${OUT}"
