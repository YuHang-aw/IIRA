#!/usr/bin/env bash
set -euo pipefail

# ========= 可配 =========
DATASET="artifacts/transfer/chexpert_vindr6_50p50n.jsonl"
CALIB="artifacts/transfer/vindr6_calib_20per.jsonl"   # 目标域小校准集（若不存在会自动生成）
MODEL="/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct"
LORA="artifacts/ckpt_rl/best"
METHOD="temp"       # temp | isotonic
OUT_CALIB="artifacts/calib"
OUT_EVAL="artifacts/eval_transfer_calib"
ETA=0.5
SEED=123
# =======================

STAMP=$(date +%Y%m%d-%H%M%S)
echo "=== TRANSFER CALIB | $STAMP ==="

# 0) 若校准集不存在，先从 DATASET 里采样生成（每概念 20 正/20 负）
if [ ! -f "${CALIB}" ]; then
  echo "[info] calib jsonl not found: ${CALIB}"
  echo "[info] auto-generating from pool: ${DATASET}"
  python scripts/calib/make_target_calib_split.py \
    --pool "${DATASET}" \
    --per_concept 20 \
    --out "${CALIB}" \
    --seed ${SEED} \
    --include_regex "(?i)vindr" || true  # 若匹配不到 vindr 也无妨
fi

# 1) 拟合工具校准+阈值
python scripts/calib/fit_kbcs_profile.py \
  --calib_jsonl "${CALIB}" \
  --model "${MODEL}" \
  --lora "${LORA}" \
  --method "${METHOD}" \
  --per_concept \
  --limit_per_concept 200 \
  --out_dir "${OUT_CALIB}"

# 找最新 profile
PROFILE="$(ls -dt ${OUT_CALIB}/*/ | head -n1)kbcs_profile.json"
echo "[info] profile = ${PROFILE}"

# 2) 用 profile 跑 transfer 对照组
python scripts/calib/eval_transfer_with_profile.py \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --lora "${LORA}" \
  --profile "${PROFILE}" \
  --n_eval 999999 \
  --out_root "${OUT_EVAL}" \
  --eta_when_adopt ${ETA}

echo "[DONE] Transfer-calibrated finished."
