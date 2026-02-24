#!/usr/bin/env bash
set -euo pipefail

# ===== 默认参数（可用环境变量覆盖）=====
TR="${TR:-/home/neutron/sdc/Interactive-Image-Reasoning-Agent/newProject/artifacts/releases/20251030-002125/eval_policy_init_no_lora_log.jsonl}"
ROI_TR="${ROI_TR:-/home/neutron/sdc/Interactive-Image-Reasoning-Agent/newProject/artifacts/eval_core/20251029-151645/dual_env_release/eval_log.jsonl}"
DS="${DS:-artifacts/rl/vindr_200.jsonl}"
MODEL="${MODEL:-/home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct}"

ADOPT_STRATEGY="${ADOPT_STRATEGY:-tool_delta}"  # final_change / tool_delta / any
ADOPT_EPS="${ADOPT_EPS:-0.01}"                  # |p1_calib - p0| 或 |p_after - p_before| 的阈值
MARGIN_THR="${MARGIN_THR:-0.5}"                 # |margin| 阈值（log-odds），仅 tool_delta/any 用
DILATE="${DILATE:-12}"
MASK_VALUE="${MASK_VALUE:-zero}"
FALLBACK_FULL_ROI="${FALLBACK_FULL_ROI:-0}"     # 1 则缺 ROI 时遮全图

DO_KBCS="${DO_KBCS:-0}"                         # 1=也跑 KBCS 对照
OUT_BASE="${OUT_BASE:-artifacts/eval/interventional/combined_$(date +%Y%m%d-%H%M%S)}"

# ===== 基本检查 =====
[[ -f "runners/eval_baselines.py" ]] || { echo "[ERR] 请在项目根目录 newProject/ 下运行"; exit 2; }
[[ -f "$TR" ]] || { echo "[ERR] 主轨迹不存在：$TR"; exit 2; }
[[ -f "$ROI_TR" ]] || { echo "[ERR] ROI 轨迹不存在：$ROI_TR"; exit 2; }
[[ -f "$DS" ]] || { echo "[ERR] 评测集不存在：$DS"; exit 2; }
echo "[INFO] 主轨迹: $TR"
echo "[INFO] ROI轨迹: $ROI_TR"
echo "[INFO] 数据集: $DS"
echo "[INFO] 模型: $MODEL"
mkdir -p "$OUT_BASE"

# ===== 生成遮挡数据集 =====
echo "[STEP] 生成遮挡数据集 → $OUT_BASE"
python eval/eval_interventional.py \
  --traj "$TR" \
  --roi_traj "$ROI_TR" \
  --base_dataset "$DS" \
  --out_dir "$OUT_BASE" \
  --mask_value "$MASK_VALUE" \
  --dilate "$DILATE" \
  --emit_non_adopted \
  --also_export_original \
  --adopt_strategy "$ADOPT_STRATEGY" \
  --adopt_eps "$ADOPT_EPS" \
  --margin_thr "$MARGIN_THR" \
  $( [[ "$FALLBACK_FULL_ROI" == "1" ]] && echo "--fallback_full_roi" )

MANIFEST="$OUT_BASE/manifest.json"
[[ -f "$MANIFEST" ]] || { echo "[ERR] manifest 不存在：$MANIFEST"; exit 4; }
ADOPTED_N=$(python - "$MANIFEST" <<'PY'
import sys,json
m=json.load(open(sys.argv[1])); print(m.get("n_adopted",0))
PY
)
echo "[INFO] n_adopted = $ADOPTED_N"
if [[ "$ADOPTED_N" -eq 0 ]]; then
  echo "[ERR] 没有采纳样本。建议：1) 降低 ADOPT_EPS（如 0.005/0.0）；2) 降低 MARGIN_THR（如 0.0）; 3) 打开 FALLBACK_FULL_ROI=1 试试。"
  exit 3
fi

# ===== PRIOR+MIX: before / after =====
PRIOR_BEFORE="$OUT_BASE/prior_mix_before"
PRIOR_AFTER="$OUT_BASE/prior_mix_after"
mkdir -p "$PRIOR_BEFORE" "$PRIOR_AFTER"

echo "[STEP] PRIOR+MIX — before（原图 adopted 子集）"
python runners/eval_baselines.py \
  --dataset "$OUT_BASE/adopted_original.jsonl" \
  --model_name "$MODEL" \
  --mode policy_init --tool_mode prior --gate mix --eta 0.5 \
  --n_eval 999999 --max_steps 3 \
  --out_dir "$PRIOR_BEFORE"

echo "[STEP] PRIOR+MIX — after（遮挡后 adopted 子集）"
python runners/eval_baselines.py \
  --dataset "$OUT_BASE/adopted.jsonl" \
  --model_name "$MODEL" \
  --mode policy_init --tool_mode prior --gate mix --eta 0.5 \
  --n_eval 999999 --max_steps 3 \
  --out_dir "$PRIOR_AFTER"

python - "$PRIOR_BEFORE/Policy(init)/metrics.json" "$PRIOR_AFTER/Policy(init)/metrics.json" <<'PY'
import sys,json,os
b,a=sys.argv[1],sys.argv[2]
def load(p): 
    return json.load(open(p)) if os.path.exists(p) else None
B,A=load(b),load(a)
if not (B and A):
    print("[ERR] 缺少 metrics.json"); sys.exit(5)
db=A["Brier(mean)"]-B["Brier(mean)"]
de=A["ECE(10bins)"]-B["ECE(10bins)"]
print(f"[RESULT][PRIOR+MIX]  ΔBrier(mean) = {db:.6f}")
print(f"[RESULT][PRIOR+MIX]  ΔECE(10bins) = {de:.6f}")
PY

# ===== KBCS（可选）=====
if [[ "$DO_KBCS" == "1" ]]; then
  KBCS_BEFORE="$OUT_BASE/kbcs_gate_before"
  KBCS_AFTER="$OUT_BASE/kbcs_gate_after"
  mkdir -p "$KBCS_BEFORE" "$KBCS_AFTER"

  echo "[STEP] KBCS-gate — before"
  python runners/eval_baselines.py \
    --dataset "$OUT_BASE/adopted_original.jsonl" \
    --model_name "$MODEL" \
    --mode policy_init --tool_mode kbcs --calibrate_tool --gate gate --tau 0.02 \
    --n_eval 999999 --max_steps 3 \
    --out_dir "$KBCS_BEFORE"

  echo "[STEP] KBCS-gate — after"
  python runners/eval_baselines.py \
    --dataset "$OUT_BASE/adopted.jsonl" \
    --model_name "$MODEL" \
    --mode policy_init --tool_mode kbcs --calibrate_tool --gate gate --tau 0.02 \
    --n_eval 999999 --max_steps 3 \
    --out_dir "$KBCS_AFTER"

  python - "$KBCS_BEFORE/Policy(init)/metrics.json" "$KBCS_AFTER/Policy(init)/metrics.json" <<'PY'
import sys,json,os
b,a=sys.argv[1],sys.argv[2]
def load(p): 
    return json.load(open(p)) if os.path.exists(p) else None
B,A=load(b),load(a)
if not (B and A):
    print("[ERR] 缺少 metrics.json"); sys.exit(6)
db=A["Brier(mean)"]-B["Brier(mean)"]
de=A["ECE(10bins)"]-B["ECE(10bins)"]
print(f"[RESULT][KBCS-gate] ΔBrier(mean) = {db:.6f}")
print(f"[RESULT][KBCS-gate] ΔECE(10bins) = {de:.6f}")
PY
fi

echo "[DONE] 产物目录：$OUT_BASE"
