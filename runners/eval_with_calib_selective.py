#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval (ENV-ALIGNED, FINAL).
- 评测严格复用 SimpleEnv：工具调用、温标/保序校准、门控/混合、是否采纳，全部走环境；
- 不再在脚本里做“第二次”校准/门控/选择性采纳，避免跨域偏差；
- 目标域校准：请用 configs/calib_kbcs.json（由你的内环 fast path 产出）；
- 工具运行参数：请用 configs/tool_runtime.json（如 external_calib/device/short_edge 等）；
- 本脚本仅做：装策略→跑环境→汇总指标→落盘。
"""
from __future__ import annotations
import os, json, argparse, random, math, time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

# 项目依赖
from training.policy import Policy
from training.rl_env import SimpleEnv, EnvCfg
from training.rewards import brier_reward

# -------------------- 基本工具 --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if limit and len(data) >= limit:
                    break
    return data

def ece_score(probs: List[float], labels: List[float], n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (probs >= b0) & (probs < b1) if b1 < 1.0 else (probs >= b0) & (probs <= b1)
        if not np.any(mask): continue
        conf = probs[mask].mean(); acc = labels[mask].mean(); w = mask.mean()
        ece += w * abs(conf - acc)
    return float(ece)

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) == 0:
        return {"n": 0}
    ps = [r["p"] for r in rows]
    gs = [r["g"] for r in rows]
    steps = [r["steps"] for r in rows]
    briers = [(p - g) ** 2 for p, g in zip(ps, gs)]
    ece = ece_score(ps, gs, n_bins=10)

    def _is_act(r, name):
        a = str(r.get("final_action", "")).upper().strip()
        name = name.upper().strip("<>")
        return (a == f"<{name}>") or (a == name)

    claim_rows = [r for r in rows if _is_act(r, "CLAIM")]
    cons_A = float(np.mean([r.get("ever_checked", False) for r in claim_rows])) if claim_rows else None

    abst_rows = [r for r in rows if _is_act(r, "ABSTAIN")]
    mid = [0.4 <= r["p"] <= 0.6 for r in abst_rows]
    cons_B = float(np.mean(mid)) if abst_rows else None

    avg_tool_ms = float(np.mean([r.get("tool_ms", 0) for r in rows]))
    avg_wall_ms = float(np.mean([r.get("wall_ms", 0) for r in rows]))
    check_rate = float(np.mean([r.get("ever_checked", False) for r in rows])) if rows else None
    adopt_rate = float(np.mean([r.get("adopted", False) for r in rows])) if rows else None

    return {
        "n": len(rows),
        "Brier(mean)": float(np.mean(briers)),
        "Brier(std)": float(np.std(briers)),
        "ECE(10bins)": float(ece),
        "AvgSteps": float(np.mean(steps)),
        "CheckRate": check_rate,
        "ConsistencyA(Claim→hadCHECK)": cons_A,
        "ConsistencyB(Abstain@0.4-0.6)": cons_B,
        "AvgToolMS": avg_tool_ms,
        "AvgWallMS": avg_wall_ms,
        "AdoptRate": adopt_rate,
    }

def save_table(rows: List[Dict[str, Any]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = [
        "image","concept",
        "id","p","g","steps",
        "reward_brier","ever_checked","adopted",
        "final_action","tool_ms","wall_ms"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in keys}
            w.writerow(row)

# -------------------- 环境对齐评测 --------------------
def run_env_once(policy: Policy, env: SimpleEnv, ex: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    p_final, steps, traj = env.rollout_once(policy, ex)
    wall_ms = int((time.time() - t0) * 1000)

    # 提取 ever_checked / adopted / final_action / tool_ms
    def _get_action(step: Dict[str, Any]) -> str | None:
        for k in ("action", "a", "act", "token", "action_token"):
            v = step.get(k)
            if isinstance(v, str) and v: return v
        return None

    ever_checked, final_action, tool_ms, adopted = False, "", 0, False
    if isinstance(traj, (list, tuple)):
        for st in traj:
            if not isinstance(st, dict): continue
            a = _get_action(st)
            if a and a.upper().strip("<>") == "CHECK": ever_checked = True
            if "tool_time_ms" in st:
                try: tool_ms += int(st["tool_time_ms"])
                except: pass
            if "adopted" in st:
                try: adopted = adopted or bool(int(st["adopted"]))
                except: adopted = adopted or bool(st["adopted"])
        for st in reversed(traj):
            a = _get_action(st)
            if a: final_action = a; break

    p_final = float(np.clip(float(p_final), 0.0, 1.0))
    g = float(np.clip(float(ex.get("soft_label", ex.get("label", 0))), 0.0, 1.0))
    img_base = Path(ex.get("image","")).name if ex.get("image","") else ""
    concept = ex.get("concept", ex.get("finding", ex.get("label_name", "")))

    return {
        "image": img_base, "concept": concept, "id": ex.get("id", None),
        "p": p_final, "g": g, "steps": int(steps),
        "reward_brier": float(brier_reward(p_final, g)),
        "ever_checked": bool(ever_checked), "adopted": bool(adopted),
        "final_action": str(final_action) if final_action else "",
        "tool_ms": int(tool_ms), "wall_ms": int(wall_ms),
    }

# -------------------- 主程序 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--lora_path", type=str, default=None)
    ap.add_argument("--mode", type=str, required=True, choices=["zs","zstool","policy_init"])
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=str, default="artifacts/eval")

    # 环境相关（全部交给 EnvCfg）
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--tool_mode", choices=["none","prior","kbcs"], default="none")
    ap.add_argument("--gate", choices=["none","mix","gate"], default="mix")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--gamma_gate", type=float, default=0.35)
    ap.add_argument("--calibrate_tool", action="store_true",
                    help="让环境对 kbcs 的 p_raw 做温标/保序（读取 configs/calib_kbcs.json）")
    ap.add_argument("--claim_gamma", type=float, default=1.0)
    ap.add_argument("--debug_traj", action="store_true")
    ap.add_argument("--check_alpha", type=float, default=0.90)
    ap.add_argument("--max_checks_per_case", type=int, default=1)

    # 关键：强制对齐环境（默认 True）
    ap.add_argument("--align_env", action="store_true", default=True,
                    help="默认开启。评测只走 SimpleEnv，不做任何脚本内的二次校准/选择性采纳。")

    args = ap.parse_args()
    set_seed(args.seed)

    # 提醒：align_env=True 时，忽略一切“脚本内校准”的历史参数（本脚本已移除这些参数）
    if not args.align_env:
        print("[warn] --align_env 未开启，本脚本仍将只走环境；脚本内二次校准已被移除。")

    # Policy & Env
    pol = Policy(
        model_name=args.model_name,
        lora_path=args.lora_path,
        load_4bit=True,
        behav_on_cpu=True,
        grad_ckpt=False
    )
    use_qwen_prior = (args.mode == "zstool")
    env = SimpleEnv(EnvCfg(
        max_steps=args.max_steps,
        disable_check=bool(args.no_check),
        check_source = args.tool_mode,     # "none" | "prior" | "kbcs"
        fuse_mode   = args.gate,           # "none" | "mix" | "gate"
        eta=args.eta, tau=args.tau, gamma_gate=args.gamma_gate,
        calibrate_tool=bool(args.calibrate_tool),
        use_qwen_prior=use_qwen_prior,
        use_baseline_when_no_check=True,
        check_alpha=args.check_alpha,
        claim_gamma=args.claim_gamma,
        debug_traj=bool(args.debug_traj),
    ))

    # 数据
    data = load_jsonl(args.dataset, limit=args.n_eval)

    # 评测：严格走环境
    rows: List[Dict[str, Any]] = []
    for ex in data:
        try:
            rows.append(run_env_once(pol, env, ex))
        except Exception as e:
            rows.append({
                "image":"", "concept":ex.get("concept",""), "id": ex.get("id", None),
                "p": float("nan"), "g": float(ex.get("soft_label", ex.get("label", 0))),
                "steps": -1, "reward_brier": float("nan"),
                "ever_checked": False, "adopted": False,
                "final_action":"", "tool_ms":0, "wall_ms":0
            })
            print(f"[warn] sample failed: {ex.get('id', '<no-id>')} -> {e}")

    # 清洗 + 汇总
    clean = [r for r in rows if (isinstance(r.get("p"), float) and math.isfinite(r["p"]) and r.get("steps",0)>=0)]
    stats = summarize(clean)

    # 输出
    tag = {"zs":"ZS","zstool":"ZS+Tool","policy_init":"Policy(init)"}[args.mode]
    if args.no_check: tag += "-noCHECK"
    if args.tool_mode == "kbcs":
        tag += "-KBCS"
        tag += f"-{args.gate}"
        if args.calibrate_tool: tag += "+calib"
    elif args.tool_mode == "prior":
        tag += "-PRIOR"
        tag += f"-{args.gate}"

    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(clean, out_dir / "preds.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[done] {tag} metrics:", json.dumps(stats, ensure_ascii=False))
    print(f"[save] {out_dir}")

if __name__ == "__main__":
    main()
