#!/usr/bin/env bash
# Core 6 experiments + dual-env copy with robust release dir detection
set -euo pipefail

# ============= configurable =============
DATA200="artifacts/rl/vindr_200.jsonl"
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
LORA="artifacts/ckpt_rl/best"
MAX_STEPS=3
N_EVAL=200
OUT_ROOT="artifacts/eval_core"
CLAIM_GAMMA=1.0    # evaluation-time sharpening; 1.0 means no sharpening
# =======================================

# ---- helpers: robustly find latest dual-env release dir ----
get_release_candidates() {
  # 1) read from configs/pipeline.yaml if present
  if [ -f "configs/pipeline.yaml" ]; then
    base=$(python - <<'PY'
import yaml, sys
try:
    y = yaml.safe_load(open("configs/pipeline.yaml","r"))
    print(y.get("PIPELINE",{}).get("RELEASE_DIR",""))
except Exception:
    print("")
PY
)
    if [ -n "${base:-}" ]; then echo "$base"; fi
  fi
  # 2) common fallbacks
  echo "artifacts/releases"
  echo "artifacts/release"
}
latest_release_dir() {
  # pick newest time-stamped subdir under candidate bases
  for base in $(get_release_candidates); do
    if [ -d "$base" ]; then
      d=$(ls -1dt "$base"/* 2>/dev/null | head -n1 || true)
      if [ -n "$d" ] && [ -d "$d" ]; then
        echo "$d"
        return
      fi
    fi
  done
  # ultimate fallback: find newest release.json anywhere under artifacts/
  d=$(find artifacts -type f -name "release.json" -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}')
  if [ -n "$d" ] && [ -d "$d" ]; then
    echo "$d"
    return
  fi
  echo ""
}

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${OUT_ROOT}/${STAMP}"
mkdir -p "${OUT}"

{
  echo "=== CORE-6 | $(date) ==="
  echo "DATA200=${DATA200}"
  echo "MODEL=${MODEL}"
  echo "LORA=${LORA}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "N_EVAL=${N_EVAL}"
  echo "CLAIM_GAMMA=${CLAIM_GAMMA}"
} | tee -a "${OUT}/MANIFEST.txt"

# A) dual-env: ZS & ZS+Tool (true tool + gate)
echo "[A] dual-env ZS / ZS+Tool(gated,true) ..."
python -m runners.main eval --data "${DATA200}"

REL_DIR="$(latest_release_dir)"
if [ -z "${REL_DIR}" ] || [ ! -d "${REL_DIR}" ]; then
  echo "[ERR] 找不到 release 目录；请检查 configs/pipeline.yaml 的 PIPELINE.RELEASE_DIR 或 artifacts/*" >&2
  exit 2
fi
echo "REL_DIR=${REL_DIR}" | tee -a "${OUT}/MANIFEST.txt"

# copy dual-env release snapshot into this batch
mkdir -p "${OUT}/dual_env_release"
cp -f "${REL_DIR}/release.json"   "${OUT}/dual_env_release/release.json"
cp -f "${REL_DIR}/eval_log.jsonl" "${OUT}/dual_env_release/eval_log.jsonl" 2>/dev/null || true
echo "[OK] copied dual-env to ${OUT}/dual_env_release" | tee -a "${OUT}/MANIFEST.txt"

# B) policy-only: 4 core runs
echo "[B1] Policy(init) — tool:none"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode none --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-init__none"

echo "[B2] Policy(init)+KBTool+Gate — tool:kbcs, calibrate, gate"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate gate --tau 0.12 --gamma_gate 0.35 --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-init__kbcs-gate"

echo "[B3] Policy(+LoRA/RL) — tool:none"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode none --gate mix --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-lora__none"

echo "[B4] Policy(+LoRA/RL)+KBTool+Gate — tool:kbcs, calibrate, gate"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate gate --tau 0.12 --gamma_gate 0.35 --eta 0.5 \
  --n_eval ${N_EVAL} --max_steps ${MAX_STEPS} --claim_gamma ${CLAIM_GAMMA} \
  --out_dir "${OUT}/Policy-lora__kbcs-gate"

# index
{
  echo
  echo "=== OUTPUT INDEX ==="
  echo "dual-env: ${OUT}/dual_env_release/{release.json, eval_log.jsonl}"
  echo "Policy(init)__none:      ${OUT}/Policy-init__none/Policy(init)/metrics.json"
  echo "Policy(init)__kbcs-gate: ${OUT}/Policy-init__kbcs-gate/Policy(init)/metrics.json"
  echo "Policy(lora)__none:      ${OUT}/Policy-lora__none/Policy(init)/metrics.json"
  echo "Policy(lora)__kbcs-gate: ${OUT}/Policy-lora__kbcs-gate/Policy(init)/metrics.json"
} | tee -a "${OUT}/MANIFEST.txt"

echo
echo "[DONE] core-6 finished. Batch dir: ${OUT}"
