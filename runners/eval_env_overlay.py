#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval with ENV alignment + runtime overlays (no file edits).
- 默认：严格复用 SimpleEnv（工具→校准→门控/混合→采纳）以保证和旧实验一致；
- 可选：以“覆盖层”在内存里注入目标域温标(T/b)与门控(tau/gamma/eta)，不改任何 json 文件；
- 可选：屏蔽 gate 中的 margin 项（drop_margin_term），缓解不同后端 margin 语义差异；
- 可选：从一小份目标域校准集( jsonl )即刻拟合每概念 T（b=0），仅本次评测生效。
"""
from __future__ import annotations
import os, json, argparse, random, math, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

# 依赖：你的工程内模块（不修改）
from training.policy import Policy
from training.rl_env import SimpleEnv, EnvCfg
from training.rewards import brier_reward

# -------------------- 工具函数 --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit: break
    return rows

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
    if not rows: return {"n": 0}
    ps = [r["p"] for r in rows]; gs = [r["g"] for r in rows]
    briers = [(p-g)**2 for p,g in zip(ps, gs)]
    ece = ece_score(ps, gs, 10)
    steps = [r["steps"] for r in rows]
    check_rate = float(np.mean([r.get("ever_checked", False) for r in rows]))
    adopt_rate = float(np.mean([r.get("adopted", False) for r in rows]))
    avg_tool = float(np.mean([r.get("tool_ms", 0) for r in rows]))
    avg_wall = float(np.mean([r.get("wall_ms", 0) for r in rows]))
    # 论文里两个一致性指标
    def _is_act(r, name):
        a = str(r.get("final_action","")).upper().strip("<>")
        return a == name.upper()
    claim_rows = [r for r in rows if _is_act(r,"CLAIM")]
    consA = float(np.mean([r.get("ever_checked", False) for r in claim_rows])) if claim_rows else None
    abst_rows = [r for r in rows if _is_act(r,"ABSTAIN")]
    consB = float(np.mean([0.4<=r["p"]<=0.6 for r in abst_rows])) if abst_rows else None
    return {
        "n": len(rows),
        "Brier(mean)": float(np.mean(briers)),
        "Brier(std)": float(np.std(briers)),
        "ECE(10bins)": float(ece),
        "AvgSteps": float(np.mean(steps)),
        "CheckRate": check_rate, "AdoptRate": adopt_rate,
        "AvgToolMS": avg_tool, "AvgWallMS": avg_wall,
        "ConsistencyA(Claim→hadCHECK)": consA,
        "ConsistencyB(Abstain@0.4-0.6)": consB,
    }

def save_table(rows: List[Dict[str, Any]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = ["image","concept","id","p","g","steps","reward_brier",
            "ever_checked","adopted","final_action","tool_ms","wall_ms"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

# -------------------- 覆盖层：目标域温标(T)拟合（b=0） --------------------
def fit_temperature(ps: np.ndarray, ys: np.ndarray) -> float:
    ps = np.clip(ps, 1e-6, 1-1e-6)
    logits = np.log(ps/(1-ps))
    grid = np.linspace(0.25, 4.0, 60)
    best_T, best = 1.0, 1e9
    for T in grid:
        q = 1/(1+np.exp(-(logits/T)))
        mse = float(np.mean((q-ys)**2))
        if mse < best: best, best_T = mse, T
    return float(best_T)

def build_overlay_from_jsonl(calib_jsonl: str,
                             min_n: int = 12) -> Dict[str, Any]:
    """
    即时从目标域小集拟合每概念 T（b=0），返回 {'default':{}, concept:{'T','b'}}。
    不改磁盘文件。
    """
    # 用工具后端拿 p_raw（保持与线上一致）
    from adapters.kbcs_check import score as kbcs_score
    from adapters.runtime_cfg import load_all as load_rt
    rt = load_rt()

    data = load_jsonl(calib_jsonl)
    by_c: Dict[str, List[Tuple[float,float]]] = {}
    hit = 0
    for ex in data:
        img = ex.get("image"); cpt = ex.get("concept") or ex.get("label_name") or ex.get("finding")
        if not img or not cpt: continue
        y = float(ex.get("soft_label", ex.get("label", 0)))
        try:
            out = kbcs_score(img, cpt, rt)
            p_raw = float(out.get("p_raw", out.get("p", 0.5)))
            by_c.setdefault(cpt.strip().lower(), []).append((p_raw, y))
            hit += 1
        except Exception:
            pass
    print(f"[overlay] calib probe hit {hit}/{len(data)}")

    overlay = {"default": {"T": 1.0, "b": 0.0}}  # tau/gamma/eta 由调用处决定是否另行覆盖
    for c, arr in sorted(by_c.items()):
        ps = np.array([p for p,_ in arr], dtype=np.float64)
        ys = np.array([y for _,y in arr], dtype=np.float64)
        if len(ps) < min_n:
            print(f"[overlay] {c:18s} n={len(ps):3d} → skip(T=1.0,b=0)")
            overlay[c] = {"T": 1.0, "b": 0.0}
        else:
            T = fit_temperature(ps, ys)
            overlay[c] = {"T": float(T), "b": 0.0}
            print(f"[overlay] {c:18s} n={len(ps):3d} → T={T:.3f}, b=0.0")
    return overlay

def merge_calib_cfg(base: Optional[dict], overlay: Optional[dict],
                    default_tau: Optional[float], default_gamma: Optional[float],
                    default_eta: Optional[float]) -> dict:
    """
    生成“内存态”的 calib 配置：以 base 为底，叠加 overlay 的 T/b，
    同时可覆写 default 的 tau/gamma/eta（不写盘）。
    """
    base = dict(base or {})
    over = dict(overlay or {})
    out = dict(base)

    # 覆写 per-concept T/b
    for k,v in over.items():
        if k == "default":
            out.setdefault("default", {})
            out["default"]["T"] = v.get("T", out["default"].get("T", 1.0))
            out["default"]["b"] = v.get("b", out["default"].get("b", 0.0))
        else:
            out.setdefault(k, {})
            if "T" in v: out[k]["T"] = v["T"]
            if "b" in v: out[k]["b"] = v["b"]

    # 覆写 default 的 tau/gamma/eta（如指定）
    out.setdefault("default", {})
    if default_tau  is not None: out["default"]["tau"]  = float(default_tau)
    if default_gamma is not None: out["default"]["gamma"]= float(default_gamma)
    if default_eta  is not None: out["default"]["eta"]  = float(default_eta)
    return out

def patch_drop_margin(env: SimpleEnv):
    """
    以猴补丁的方式屏蔽 gate 中的 margin 影响（不改源码/文件）。
    """
    orig = env._gate_fuse
    def _gate_fuse_wo_margin(concept: str, p0: float, p1: float, margin: float | None):
        return orig(concept, p0, p1, None)
    env._gate_fuse = _gate_fuse_wo_margin  # type: ignore
    print("[overlay] gate margin term disabled (drop_margin_term=True)")

# -------------------- 一次评测（严格走环境） --------------------
def run_env_once(policy: Policy, env: SimpleEnv, ex: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    p_final, steps, traj = env.rollout_once(policy, ex)
    wall_ms = int((time.time() - t0) * 1000)

    def _act_name(step: Dict[str, Any]) -> Optional[str]:
        for k in ("action","a","act","token","action_token"):
            v = step.get(k)
            if isinstance(v,str) and v: return v
        return None

    ever_checked, adopted, tool_ms, final_action = False, False, 0, ""
    if isinstance(traj, (list, tuple)):
        for st in traj:
            if not isinstance(st, dict): continue
            a = _act_name(st)
            if a and a.upper().strip("<>")=="CHECK": ever_checked = True
            if "adopted" in st:
                try: adopted = adopted or bool(int(st["adopted"]))
                except: adopted = adopted or bool(st["adopted"])
            if "tool_time_ms" in st:
                try: tool_ms += int(st["tool_time_ms"])
                except: pass
        for st in reversed(traj):
            a = _act_name(st)
            if a: final_action = a; break

    p_final = float(np.clip(float(p_final), 0.0, 1.0))
    g = float(np.clip(float(ex.get("soft_label", ex.get("label", 0))), 0.0, 1.0))
    img_base = Path(ex.get("image","")).name if ex.get("image","") else ""
    concept = ex.get("concept", ex.get("finding", ex.get("label_name","")))

    return {
        "image": img_base, "concept": concept, "id": ex.get("id", None),
        "p": p_final, "g": g, "steps": int(steps),
        "reward_brier": float(brier_reward(p_final, g)),
        "ever_checked": bool(ever_checked), "adopted": bool(adopted),
        "final_action": final_action, "tool_ms": int(tool_ms), "wall_ms": wall_ms,
    }

# -------------------- 主程序 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--mode", choices=["zs","zstool","policy_init"], required=True)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", default="artifacts/eval_overlay")

    # 环境参数（与旧实验一致）
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--tool_mode", choices=["none","prior","kbcs","grid","self"], default="none")
    ap.add_argument("--grid_n", type=int, default=3, help="grid/self: split image into NxN and probe each cell")
    ap.add_argument("--grid_allow_repeat", action="store_true", help="grid/self: allow selecting same cell across multiple CHECK steps")
    ap.add_argument("--self_mode", choices=["grid","bbox"], default="grid", help="self: probe mode (bbox reserved, grid recommended)")
    ap.add_argument("--gate", choices=["none","mix","gate"], default="mix")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--gamma_gate", type=float, default=0.35)
    ap.add_argument("--calibrate_tool", action="store_true")
    ap.add_argument("--claim_gamma", type=float, default=1.0)
    ap.add_argument("--debug_traj", action="store_true")
    ap.add_argument("--check_alpha", type=float, default=0.90)

    # 运行期覆盖层（不改文件）
    ap.add_argument("--overlay_calib_json", type=str, default=None,
                    help="提供一份 per-concept T/b 的 json（仅内存生效）")
    ap.add_argument("--fit_calib_from_jsonl", type=str, default=None,
                    help="从目标域小集(jsonl)即时拟合每概念 T(b=0)，仅本次评测生效")
    ap.add_argument("--override_default_tau", type=float, default=None)
    ap.add_argument("--override_default_gamma", type=float, default=None)
    ap.add_argument("--override_default_eta", type=float, default=None)
    ap.add_argument("--drop_margin_term", action="store_true",
                    help="gate 时忽略 margin 增益项（仅内存猴补丁，不改源码）")
    ap.add_argument("--use_qwen_prior", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    # 1) 构建策略与环境（严格与旧链路一致）
    pol = Policy(args.model_name, lora_path=args.lora_path,
                 load_4bit=True, behav_on_cpu=True, grad_ckpt=False)
    use_qwen_prior = args.use_qwen_prior or (args.mode == "zstool")
    env = SimpleEnv(EnvCfg(
        max_steps=args.max_steps,
        disable_check=bool(args.no_check),
        check_source=args.tool_mode,          # "none"/"prior"/"kbcs"/"grid"/"self"
        self_mode=args.self_mode,
        grid_n=args.grid_n,
        grid_allow_repeat=bool(args.grid_allow_repeat),
        fuse_mode=args.gate,                  # "none"/"mix"/"gate"
        eta=args.eta, tau=args.tau, gamma_gate=args.gamma_gate,
        calibrate_tool=bool(args.calibrate_tool),
        use_qwen_prior=use_qwen_prior,
        use_baseline_when_no_check=True,
        check_alpha=args.check_alpha,
        claim_gamma=args.claim_gamma,
        debug_traj=bool(args.debug_traj),
    ))

    # 2) 构造“内存覆盖层”的 calib 配置（可来自 json 或即时拟合）
    overlay: Optional[dict] = None
    if args.fit_calib_from_jsonl:
        overlay = build_overlay_from_jsonl(args.fit_calib_from_jsonl)
    elif args.overlay_calib_json:
        try:
            overlay = json.loads(Path(args.overlay_calib_json).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[overlay][warn] load overlay_calib_json failed: {e}")
            overlay = None

    # 3) 把覆盖层注入到环境（仅内存，不改磁盘）
    if overlay is not None:
        merged = merge_calib_cfg(
            base=getattr(env, "_calib_cfg", None),
            overlay=overlay,
            default_tau=args.override_default_tau,
            default_gamma=args.override_default_gamma,
            default_eta=args.override_default_eta,
        )
        env._calib_cfg = merged  # type: ignore
        print("[overlay] env._calib_cfg updated (in-memory only)")
    else:
        # 即便没有 overlay，也可仅覆写 default 的 tau/gamma/eta
        if any(v is not None for v in [args.override_default_tau, args.override_default_gamma, args.override_default_eta]):
            merged = merge_calib_cfg(
                base=getattr(env, "_calib_cfg", None),
                overlay=None,
                default_tau=args.override_default_tau,
                default_gamma=args.override_default_gamma,
                default_eta=args.override_default_eta,
            )
            env._calib_cfg = merged  # type: ignore
            print("[overlay] env._calib_cfg default gate params overridden (in-memory only)")

    # 4) 可选屏蔽 gate 的 margin 项
    if args.drop_margin_term and hasattr(env, "_gate_fuse"):
        patch_drop_margin(env)

    # 5) 跑评测（严格走环境）
    data = load_jsonl(args.dataset, args.n_eval)
    rows: List[Dict[str, Any]] = []
    for ex in data:
        try:
            rows.append(run_env_once(pol, env, ex))
        except Exception as e:
            rows.append({
                "image":"", "concept":ex.get("concept",""), "id": ex.get("id", None),
                "p": float("nan"), "g": float(ex.get("soft_label", ex.get("label",0))),
                "steps": -1, "reward_brier": float("nan"),
                "ever_checked": False, "adopted": False,
                "final_action":"", "tool_ms":0, "wall_ms":0
            })
            print(f"[warn] sample failed: {ex.get('id','<no-id>')} -> {e}")

    clean = [r for r in rows if isinstance(r.get("p"), float) and math.isfinite(r["p"]) and r.get("steps",0)>=0]
    stats = summarize(clean)

    # 6) 落盘
    tag = {"zs":"ZS","zstool":"ZS+Tool","policy_init":"Policy(init)"}[args.mode]
    if args.tool_mode == "kbcs":
        tag += "-KBCS"
        tag += f"-{args.gate}"
        if args.calibrate_tool: tag += "+calib"
        if overlay is not None: tag += "+overlayT"
        if args.drop_margin_term: tag += "-noMargin"
    elif args.tool_mode == "prior":
        tag += "-PRIOR-" + args.gate
    if args.no_check: tag += "-noCHECK"

    out_dir = Path(args.out_dir)/tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(clean, out_dir/"preds.csv")
    (out_dir/"metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {tag} metrics:", json.dumps(stats, ensure_ascii=False))
    print(f"[save] {out_dir}")

if __name__ == "__main__":
    main()