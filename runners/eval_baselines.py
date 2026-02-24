from __future__ import annotations
import os, json, argparse, random, math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

# 复用你项目里的组件
from training.policy import Policy
from training.rl_env import SimpleEnv, EnvCfg
from training.rewards import brier_reward

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data

def ece_score(probs: List[float], labels: List[float], n_bins: int = 15) -> float:
    """经典 ECE：标量二分类标签（0/1 或 soft label 0~1）"""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (probs >= b0) & (probs < b1) if b1 < 1.0 else (probs >= b0) & (probs <= b1)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        w    = mask.mean()
        ece += w * abs(conf - acc)
    return float(ece)

import time

def run_eval_once(policy: Policy, env: SimpleEnv, ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用 env 跑一条轨迹（eval 模式）。SimpleEnv.rollout_once 返回 (p_final, steps, traj)。
    这里额外记录：
      - ever_checked: 轨迹中是否出现过 <CHECK>
      - final_action: 最后一个动作（<CLAIM>/<CHECK>/<ABSTAIN>/<STOP>）
      - tool_ms: 轨迹内工具调用耗时总和（若 traj 记录了 tool_time_ms；否则为 0）
      - wall_ms: 整条样本的端到端耗时（毫秒）
    """
    t0 = time.time()
    p_final, steps, traj = env.rollout_once(policy, ex)
    wall_ms = int((time.time() - t0) * 1000)

    def _get_action(step: Dict[str, Any]) -> str | None:
        for k in ("action", "a", "act", "token", "action_token"):
            v = step.get(k)
            if isinstance(v, str) and v:
                return v
        return None

    ever_checked = False
    final_action = None
    tool_ms = 0
    adopted = False
    if isinstance(traj, (list, tuple)):
        for st in traj:
            if not isinstance(st, dict):
                continue
            a = _get_action(st)
            if a and a.upper().strip("<>") == "CHECK":
                ever_checked = True
            if "tool_time_ms" in st:
                try:
                    tool_ms += int(st["tool_time_ms"])
                except Exception:
                    pass
            if "adopted" in st:
                try:
                    adopted = adopted or bool(int(st["adopted"]))
                except Exception:
                    adopted = adopted or bool(st["adopted"])
        # 找最后一个有动作名的 step
        for st in reversed(traj):
            a = _get_action(st)
            if a:
                final_action = a
                break

    p_final = float(max(0.0, min(1.0, float(p_final))))
    g = float(ex.get("soft_label", ex.get("label", 0)))
    g = max(0.0, min(1.0, g))

    img_raw = ex.get("image", "")
    img_base = Path(img_raw).name if img_raw else ""
    concept = ex.get("concept", ex.get("finding", ex.get("label_name", "")))

    return {
        # 新增的元信息
        "image": img_base,
        "concept": concept,

        # 下面都是你原来已经填的
        "id": ex.get("id", None),
        "p": p_final,
        "g": g,
        "steps": int(steps),
        "reward_brier": float(brier_reward(p_final, g)),
        "ever_checked": bool(ever_checked),
        "adopted": bool(adopted),
        "final_action": str(final_action) if final_action else "",
        "tool_ms": int(tool_ms),
        "wall_ms": int(wall_ms),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) == 0:
        return {"n": 0}
    ps = [r["p"] for r in rows]
    gs = [r["g"] for r in rows]
    steps = [r["steps"] for r in rows]
    briers_pos = [(p - g) ** 2 for p, g in zip(ps, gs)]
    ece = ece_score(ps, gs, n_bins=10)

    # Consistency-A：最终 CLAIM 的样本里，有多少在此之前出现过 CHECK
    def _is_act(r, name):
        a = str(r.get("final_action", "")).upper().strip()
        name = name.upper().strip("<>")
        return (a == f"<{name}>") or (a == name)
    claim_rows = [r for r in rows if _is_act(r, "CLAIM")]
    cons_A = float(np.mean([r.get("ever_checked", False) for r in claim_rows])) if claim_rows else None

    # Consistency-B：最终 ABSTAIN 的样本，其 p 是否集中在中间区间（默认 0.4~0.6）
    abst_rows = [r for r in rows if _is_act(r, "ABSTAIN")]
    mid = [0.4 <= r["p"] <= 0.6 for r in abst_rows]
    cons_B = float(np.mean(mid)) if abst_rows else None

    # 耗时统计
    avg_tool_ms = float(np.mean([r.get("tool_ms", 0) for r in rows]))
    avg_wall_ms = float(np.mean([r.get("wall_ms", 0) for r in rows]))
    check_rate = float(np.mean([r.get("ever_checked", False) for r in rows])) if rows else None
    adopt_rate = float(np.mean([r.get("adopted", False) for r in rows])) if rows else None
    return {
        "n": len(rows),
        "Brier(mean)": float(np.mean(briers_pos)),
        "Brier(std)": float(np.std(briers_pos)),
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
    "image", "concept",   
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="评测集 JSONL；每行至少包含 image/prompt/label 或 soft_label")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--lora_path", type=str, default=None, help="LoRA 路径（Policy(init) 可留空仅挂 LoRA 结构）")
    ap.add_argument("--mode", type=str, required=True, choices=["zs","zstool","policy_init"], help="三种评测模式")
    ap.add_argument("--n_eval", type=int, default=200, help="最多评测多少条（可先小样本快速看走势）")
    ap.add_argument("--max_steps", type=int, default=3, help="每条轨迹最多交互步数")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=str, default="artifacts/eval")
    # --- 新增：Ablation 开关（禁用 <CHECK>） ---
    ap.add_argument("--no_check", action="store_true", help="禁用 <CHECK> 动作（Ablation: noCHECK）")
    ap.add_argument("--tool_mode", choices=["none","prior","kbcs"], default="none",
                    help="CHECK 的证据源：无/先验/真工具(KBCS)")
    ap.add_argument("--gate", choices=["none","mix","gate"], default="mix",
                    help="融合：none 覆盖 / mix 线性混合 / gate 门控融合")
    ap.add_argument("--eta", type=float, default=0.5, help="mix 融合权重")
    ap.add_argument("--tau", type=float, default=0.12, help="门控阈值 τ")
    ap.add_argument("--gamma_gate", type=float, default=0.35, help="门控收益权重 γ")
    ap.add_argument("--calibrate_tool", action="store_true", help="对工具概率做温标/保序校准")
    ap.add_argument("--claim_gamma", type=float, default=1.0, help="评测期 claim 锐化 γ_eval（建议=1.0）")
    ap.add_argument("--debug_traj", action="store_true", help="把轨迹写入 artifacts/debug/rl_traj.jsonl，便于定位 CHECK/工具使用")
    ap.add_argument("--check_alpha", type=float, default=0.90,
                help="触发 CHECK 的置信门槛（越低越容易触发，CheckRate 越高）")
    ap.add_argument("--max_checks_per_case", type=int, default=1,
                help="每个样本最多发起 CHECK 的次数上限")

    args = ap.parse_args()

    set_seed(args.seed)

    # 构建 Policy（会自动在 lora_path 存在时恢复 action_rows）
    pol = Policy(model_name=args.model_name, lora_path=args.lora_path, load_4bit=True, behav_on_cpu=True, grad_ckpt=False)

    # 三种模式仅在 Env 上有差异；禁用 CHECK 通过 EnvCfg.disable_check 控制
    use_qwen_prior = (args.mode == "zstool")
    env = SimpleEnv(EnvCfg(
        max_steps=args.max_steps,
        disable_check=bool(args.no_check),
        check_source = args.tool_mode,
        fuse_mode = args.gate,
        eta=args.eta, tau=args.tau, gamma_gate=args.gamma_gate,
        calibrate_tool=bool(args.calibrate_tool),
        use_qwen_prior=use_qwen_prior,
        use_baseline_when_no_check=True,

        check_alpha=args.check_alpha,
        claim_gamma=args.claim_gamma,
        debug_traj=bool(args.debug_traj),
    ))

    data = load_jsonl(args.dataset, limit=args.n_eval)
    rows: List[Dict[str, Any]] = []
    for ex in data:
        try:
            rows.append(run_eval_once(pol, env, ex))
        except Exception as e:
            # 某个样本失败不致命，记录一下继续
            rows.append({"id": ex.get("id", None), "p": float("nan"), "g": float(ex.get("soft_label", ex.get("label", 0))), "steps": -1, "reward_brier": float("nan")})
            print(f"[warn] sample failed: {ex.get('id', '<no-id>')} -> {e}")

    # 过滤 NaN
    clean = [r for r in rows if (isinstance(r.get("p"), float) and math.isfinite(r["p"]) and r.get("steps",0)>=0)]
    stats = summarize(clean)

    # 另存（noCHECK 后缀，避免和正常版本混淆）
    tag = {"zs":"ZS","zstool":"ZS+Tool","policy_init":"Policy(init)"}[args.mode]
    if args.no_check:
        tag += "-noCHECK"
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(clean, out_dir / "preds.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 若是 ZS+Tool，顺便算下 Evidence-lift（相对 ZS）
    if args.mode == "zstool":
        zs_metrics = Path(args.out_dir) / "ZS" / "metrics.json"
        if zs_metrics.exists():
            try:
                base = json.loads(zs_metrics.read_text(encoding="utf-8"))
                lift = base["Brier(mean)"] - stats["Brier(mean)"]   # Brier 越小越好：ZS - ZS+Tool
                stats["EvidenceLift(Brier↓)"] = float(lift)
                with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    print(f"[done] {tag} metrics:", json.dumps(stats, ensure_ascii=False))
    print(f"[save] {out_dir}")

if __name__ == "__main__":
    main()
