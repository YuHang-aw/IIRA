#!/usr/bin/env bash
# 8B FULL: dual-env snapshot + CORE-6 + EXT-6
# - Robust release dir detection (copied from 3B core)
# - Core-6: (init/lora) x (noP&G baseline, kbcs-gate)
# - Ext-6:  (init/lora) x (prior-mix, kbcs-mix) + noP&G baseline (already in core but we also index it)
set -euo pipefail

# ===================== configurable =====================
DATA200="artifacts/rl/vindr_200.jsonl"

MODEL="/home/neutron/sdc/MODEL/Qwen3-VL-8B-Instruct"
LORA="artifacts/ckpt_rl_vindr_auto_k3_se328_g3/chunk_10_from_1800/step_200"

MAX_STEPS=3
N_EVAL=200
OUT_ROOT="artifacts/eval_core_8b"

# evaluation-time sharpening; 1.0 means no sharpening
CLAIM_GAMMA=1.0

# tool/gate hyperparams (keep consistent with your paper defaults)
ETA_MIX=0.5
TAU_GATE=0.12
GAMMA_GATE=0.35
# =======================================================

# ---- helpers: robustly find latest dual-env release dir ----
get_release_candidates() {
  # 1) read from configs/pipeline.yaml if present
  if [ -f "configs/pipeline.yaml" ]; then
    base=$(python - <<'PY'
import sys
try:
    import yaml
    y = yaml.safe_load(open("configs/pipeline.yaml","r"))
    print(y.get("PIPELINE",{}).get("RELEASE_DIR","") or "")
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
  echo "=== FULL-8B | $(date) ==="
  echo "DATA200=${DATA200}"
  echo "MODEL=${MODEL}"
  echo "LORA=${LORA}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "N_EVAL=${N_EVAL}"
  echo "CLAIM_GAMMA=${CLAIM_GAMMA}"
  echo "ETA_MIX=${ETA_MIX}"
  echo "TAU_GATE=${TAU_GATE}"
  echo "GAMMA_GATE=${GAMMA_GATE}"
} | tee -a "${OUT}/MANIFEST.txt"

# =========================================================
# A) dual-env snapshot (true tool + gate) via runners.main
# =========================================================
echo "[A] dual-env ZS / ZS+Tool(gated,true) ..."
python -m runners.main eval --data "${DATA200}"

REL_DIR="$(latest_release_dir)"
if [ -z "${REL_DIR}" ] || [ ! -d "${REL_DIR}" ]; then
  echo "[ERR] 找不到 release 目录；请检查 configs/pipeline.yaml 的 PIPELINE.RELEASE_DIR 或 artifacts/*" >&2
  exit 2
fi
echo "REL_DIR=${REL_DIR}" | tee -a "${OUT}/MANIFEST.txt"

mkdir -p "${OUT}/dual_env_release"
cp -f "${REL_DIR}/release.json"   "${OUT}/dual_env_release/release.json"
cp -f "${REL_DIR}/eval_log.jsonl" "${OUT}/dual_env_release/eval_log.jsonl" 2>/dev/null || true
echo "[OK] copied dual-env to ${OUT}/dual_env_release" | tee -a "${OUT}/MANIFEST.txt"

# =========================================================
# B) CORE-6 (policy-only, reproducible)
#    noP&G baseline MUST be --no_check
# =========================================================
echo
echo "================ CORE-6 ================" | tee -a "${OUT}/MANIFEST.txt"

# B1) Policy(init) — noP&G baseline
echo "[B1] Policy(init) — noP&G baseline (--no_check)"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode none --gate mix --eta "${ETA_MIX}" \
  --no_check \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__noPG"

# B2) Policy(init)+KBTool+Gate
echo "[B2] Policy(init)+KBTool+Gate — tool:kbcs, calibrate, gate"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate gate --tau "${TAU_GATE}" --gamma_gate "${GAMMA_GATE}" --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__kbcs-gate"

# B3) Policy(+LoRA/RL) — noP&G baseline
echo "[B3] Policy(lora) — noP&G baseline (--no_check)"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode none --gate mix --eta "${ETA_MIX}" \
  --no_check \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__noPG"

# B4) Policy(+LoRA/RL)+KBTool+Gate
echo "[B4] Policy(lora)+KBTool+Gate — tool:kbcs, calibrate, gate"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate gate --tau "${TAU_GATE}" --gamma_gate "${GAMMA_GATE}" --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__kbcs-gate"

# =========================================================
# C) EXT-6 (aligned with your 8B ext script)
# =========================================================
echo
echo "================ EXT-6 ================" | tee -a "${OUT}/MANIFEST.txt"

# C1) Policy(init)+Prior-mix
echo "[C1] Policy(init)+Prior-mix — tool:prior, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode prior --gate mix --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__prior-mix"

# C2) Policy(lora)+Prior-mix
echo "[C2] Policy(lora)+Prior-mix — tool:prior, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode prior --gate mix --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__prior-mix"

# C3) Policy(init)+KBCS-mix (calibrate)
echo "[C3] Policy(init)+KBCS-mix — tool:kbcs, calibrate, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate mix --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-init__kbcs-mix"

# C4) Policy(lora)+KBCS-mix (calibrate)
echo "[C4] Policy(lora)+KBCS-mix — tool:kbcs, calibrate, mix"
python runners/eval_baselines.py \
  --dataset "${DATA200}" --model_name "${MODEL}" --lora_path "${LORA}" \
  --mode policy_init --tool_mode kbcs --calibrate_tool \
  --gate mix --eta "${ETA_MIX}" \
  --n_eval "${N_EVAL}" --max_steps "${MAX_STEPS}" --claim_gamma "${CLAIM_GAMMA}" \
  --out_dir "${OUT}/Policy-lora__kbcs-mix"

# =========================================================
# D) index
# =========================================================
{
  echo
  echo "=== OUTPUT INDEX ==="
  echo "dual-env: ${OUT}/dual_env_release/{release.json, eval_log.jsonl}"
  echo
  echo "[CORE]"
  echo "Policy(init)__noPG:       ${OUT}/Policy-init__noPG/Policy(init)/metrics.json"
  echo "Policy(init)__kbcs-gate:  ${OUT}/Policy-init__kbcs-gate/Policy(init)/metrics.json"
  echo "Policy(lora)__noPG:       ${OUT}/Policy-lora__noPG/Policy(init)/metrics.json"
  echo "Policy(lora)__kbcs-gate:  ${OUT}/Policy-lora__kbcs-gate/Policy(init)/metrics.json"
  echo
  echo "[EXT]"
  echo "Policy(init)__prior-mix:  ${OUT}/Policy-init__prior-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__prior-mix:  ${OUT}/Policy-lora__prior-mix/Policy(init)/metrics.json"
  echo "Policy(init)__kbcs-mix:   ${OUT}/Policy-init__kbcs-mix/Policy(init)/metrics.json"
  echo "Policy(lora)__kbcs-mix:   ${OUT}/Policy-lora__kbcs-mix/Policy(init)/metrics.json"
} | tee -a "${OUT}/MANIFEST.txt"

echo
echo "[DONE] FULL-8B finished. Batch dir: ${OUT}"
