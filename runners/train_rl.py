# runners/train_rl.py
from __future__ import annotations

import argparse, json, pathlib, random, time
from typing import Dict, Any, List, Optional

import numpy as np

from training.rl_trainer import RLTrainer, RLCfg

# 复用你已有的评测入口：直接调用 Policy + Env
from training.policy import Policy
from training.rl_env import SimpleEnv, EnvCfg


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in _iter_jsonl(path):
        out.append(ex)
        if limit is not None and len(out) >= limit:
            break
    return out


def _brier(p: float, g: float) -> float:
    return float((p - g) ** 2)


def _ece_10bins(ps: List[float], gs: List[float], n_bins: int = 10) -> float:
    ps = np.asarray(ps, dtype=np.float64)
    gs = np.asarray(gs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (ps >= b0) & (ps < b1) if b1 < 1.0 else (ps >= b0) & (ps <= b1)
        if not np.any(mask):
            continue
        conf = ps[mask].mean()
        acc = gs[mask].mean()
        w = mask.mean()
        ece += w * abs(conf - acc)
    return float(ece)


def _eval_dataset(
    model_name: str,
    lora_path: Optional[str],
    data: List[Dict[str, Any]],
    *,
    max_steps: int,
    check_source: str,
    self_mode: str,
    grid_n: int,
    grid_allow_repeat: bool,
    fuse_mode: str,
    eta: float,
    tau: float,
    gamma_gate: float,
    calibrate_tool: bool,
    claim_gamma: float,
    check_alpha: float,
    short_edge: int,
    load_4bit: bool,
    behav_on_cpu: bool,
    grad_ckpt: bool,
    seed: int,
    debug_traj: bool = False,
    use_qwen_prior: bool = False,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    pol = Policy(
        model_name,
        lora_path=lora_path,
        short_edge=short_edge,
        load_4bit=load_4bit,
        behav_on_cpu=behav_on_cpu,
        grad_ckpt=grad_ckpt,
    )

    # ✅ 确保 eval 时 action_head 参数也能恢复
    # Policy.__init__ 会自动尝试从 <lora_path>/action_rows.safetensors 加载；
    # 这里再做一层兜底，避免外部传入的是目录路径而不是具体文件。
    if lora_path and hasattr(pol, "load_action_rows"):
        try:
            p = pathlib.Path(lora_path)
            f = p / "action_rows.safetensors" if p.is_dir() else p
            if f.exists():
                pol.load_action_rows(str(f))
        except Exception:
            pass

    env = SimpleEnv(
        EnvCfg(
            max_steps=max_steps,
            check_source=check_source,
            self_mode=self_mode,
            grid_n=grid_n,
            grid_allow_repeat=bool(grid_allow_repeat),
            fuse_mode=fuse_mode,
            debug_traj=bool(debug_traj),
            eta=float(eta),
            tau=float(tau),
            gamma_gate=float(gamma_gate),
            calibrate_tool=bool(calibrate_tool),
            claim_gamma=float(claim_gamma),
            check_alpha=float(check_alpha),
            use_qwen_prior=bool(use_qwen_prior),
        )
    )

    ps: List[float] = []
    gs: List[float] = []
    steps_list: List[int] = []
    check_any: List[bool] = []
    adopt_any: List[bool] = []
    tool_ms_list: List[int] = []
    wall_ms_list: List[int] = []

    for ex in data:
        t0 = time.time()
        p_final, steps, traj = env.rollout_once(pol, ex)
        wall_ms = int((time.time() - t0) * 1000)

        g = float(ex.get("soft_label", ex.get("label", 0.0)))
        g = float(np.clip(g, 0.0, 1.0))
        p_final = float(np.clip(float(p_final), 0.0, 1.0))

        ever_checked = False
        adopted = False
        tool_ms = 0

        if isinstance(traj, list):
            for st in traj:
                if not isinstance(st, dict):
                    continue
                if st.get("action") == "<CHECK>":
                    ever_checked = True

                # adopted 推断：p_after != p_before
                pb = st.get("p_before", None)
                pa = st.get("p_after", None)
                if pb is not None and pa is not None:
                    try:
                        if abs(float(pa) - float(pb)) > 1e-6:
                            adopted = True
                    except Exception:
                        pass

                if "adopted" in st:
                    try:
                        adopted = adopted or bool(int(st["adopted"]))
                    except Exception:
                        adopted = adopted or bool(st["adopted"])

                if "tool_time_ms" in st:
                    try:
                        tool_ms += int(st["tool_time_ms"])
                    except Exception:
                        pass

        ps.append(p_final)
        gs.append(g)
        steps_list.append(int(steps))
        check_any.append(bool(ever_checked))
        adopt_any.append(bool(adopted))
        tool_ms_list.append(int(tool_ms))
        wall_ms_list.append(int(wall_ms))

    briers = [_brier(p, g) for p, g in zip(ps, gs)]
    return {
        "n": len(ps),
        "Brier(mean)": float(np.mean(briers)) if briers else None,
        "ECE(10bins)": _ece_10bins(ps, gs, 10) if briers else None,
        "AvgSteps": float(np.mean(steps_list)) if steps_list else None,
        "CheckRate": float(np.mean(check_any)) if check_any else None,
        "AdoptRate": float(np.mean(adopt_any)) if adopt_any else None,
        "AvgToolMS": float(np.mean(tool_ms_list)) if tool_ms_list else None,
        "AvgWallMS": float(np.mean(wall_ms_list)) if wall_ms_list else None,
    }


def main():
    ap = argparse.ArgumentParser()

    # data/model
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None, help="optional validation jsonl (recommended)")
    ap.add_argument("--init_lora", default=None)

    # RL core
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--ent", type=float, default=0.001)
    ap.add_argument("--is_clip", type=float, default=5.0)
    ap.add_argument("--adv_scale", type=float, default=5.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--refresh_every", type=int, default=50)
    ap.add_argument("--warmup", dest="warmup", action="store_true")
    ap.add_argument("--no_warmup", dest="warmup", action="store_false")
    ap.set_defaults(warmup=True)
    ap.add_argument("--max_steps_per_traj", type=int, default=3)
    ap.add_argument("--disable_check", action="store_true")
    ap.add_argument("--use_baseline_when_no_check", dest="use_baseline_when_no_check", action="store_true")
    ap.add_argument("--no_use_baseline_when_no_check", dest="use_baseline_when_no_check", action="store_false")
    ap.set_defaults(use_baseline_when_no_check=True)

    # env / tool
    ap.add_argument("--check_source", choices=["none", "prior", "kbcs", "grid", "self"], default="self")
    ap.add_argument("--self_mode", choices=["grid", "bbox"], default="grid")
    ap.add_argument("--grid_n", type=int, default=3)
    ap.add_argument("--grid_allow_repeat", action="store_true")

    # 兼容旧脚本：--gate 是 --fuse_mode 的别名
    ap.add_argument("--gate", dest="fuse_mode", choices=["none", "mix", "gate"], default=None)
    ap.add_argument("--fuse_mode", dest="fuse_mode", choices=["none", "mix", "gate"], default="gate")

    ap.add_argument("--eta", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--gamma_gate", type=float, default=0.35)
    ap.add_argument("--calibrate_tool", action="store_true")
    ap.add_argument("--check_alpha", type=float, default=0.9)
    ap.add_argument("--claim_gamma", type=float, default=1.0)

    # reward shaping (optional)
    ap.add_argument("--step_penalty", type=float, default=0.0, help="penalty per step (>=0)")
    ap.add_argument("--check_cost", type=float, default=0.0, help="extra cost for each <CHECK>")

    # ✅ optional: image-conditioned prior p0
    ap.add_argument("--use_qwen_prior", action="store_true", help="use Qwen prior for p0 instead of concept prevalence")

    # policy load (memory/speed knobs)
    ap.add_argument("--behav_on_cpu", action="store_true")
    ap.add_argument("--load_4bit", dest="load_4bit", action="store_true")
    ap.add_argument("--no_load_4bit", dest="load_4bit", action="store_false")
    ap.set_defaults(load_4bit=True)
    ap.add_argument("--short_edge", type=int, default=384)
    ap.add_argument("--grad_ckpt", dest="grad_ckpt", action="store_true")
    ap.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    ap.set_defaults(grad_ckpt=True)

    # ckpt/output
    ap.add_argument("--ckpt_dir", default="artifacts/ckpt_rl")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--es_patience", type=int, default=400)
    ap.add_argument("--es_delta", type=float, default=1e-4)
    ap.add_argument("--save_lora_dir", default="artifacts/lora/rl-cispo-v1")

    # eval loop
    ap.add_argument("--eval_every", type=int, default=200, help="if --val provided, run eval every N steps")
    ap.add_argument("--eval_n", type=int, default=400, help="subsample size for val eval (speed)")
    ap.add_argument("--select_metric", choices=["brier", "ece"], default="brier")

    args = ap.parse_args()

    # Load datasets
    train_ds = _load_jsonl(args.data)
    val_ds = _load_jsonl(args.val, limit=args.eval_n) if args.val else None

    cfg = RLCfg(
        model_name=args.model,
        init_lora=args.init_lora,
        load_4bit=bool(args.load_4bit),
        short_edge=int(args.short_edge),
        behav_on_cpu=bool(args.behav_on_cpu),
        grad_ckpt=bool(args.grad_ckpt),
        lr=args.lr,
        kl_coef=args.kl,
        ent_coef=args.ent,
        is_clip=args.is_clip,
        K=args.K,
        steps=args.steps,
        seed=args.seed,
        max_steps_per_traj=args.max_steps_per_traj,
        check_source=args.check_source,
        self_mode=args.self_mode,
        grid_n=args.grid_n,
        grid_allow_repeat=bool(args.grid_allow_repeat),
        batch_size=args.batch_size,
        refresh_every=args.refresh_every,
        warmup=bool(args.warmup),
        es_patience=args.es_patience,
        es_delta=args.es_delta,
        ckpt_every=args.ckpt_every,
        ckpt_dir=args.ckpt_dir,
        grad_clip=args.grad_clip,
        adv_scale=args.adv_scale,
        fuse_mode=args.fuse_mode,
        eta=args.eta,
        tau=args.tau,
        gamma_gate=args.gamma_gate,
        calibrate_tool=bool(args.calibrate_tool),
        check_alpha=args.check_alpha,
        claim_gamma=args.claim_gamma,
        use_baseline_when_no_check=bool(args.use_baseline_when_no_check),
        disable_check=bool(args.disable_check),
        step_penalty=float(args.step_penalty),
        check_cost=float(args.check_cost),
    )

    # ✅ 这里把 use_qwen_prior 透传给 RLTrainer（EnvCfg 内会用）
    trainer = RLTrainer(cfg, use_qwen_prior=bool(args.use_qwen_prior), behav_on_cpu=bool(args.behav_on_cpu))

    # Optional: track best on val
    best_score = float("inf")
    best_step = 0

    # Train step-by-step so we can interleave eval
    for step in range(1, cfg.steps + 1):
        batch = random.sample(train_ds, k=min(cfg.batch_size, len(train_ds)))
        _ = trainer.step_batch(batch)

        # periodic ckpt
        if cfg.ckpt_every and (step % cfg.ckpt_every == 0):
            pathlib.Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
            sd = pathlib.Path(cfg.ckpt_dir) / f"step_{step}"
            trainer.policy.model.save_pretrained(str(sd))
            trainer._save_action_rows(str(sd))

        if cfg.refresh_every and (step % cfg.refresh_every == 0):
            trainer.policy.refresh_behavior(show_progress=False)

        # periodic eval
        if val_ds and (args.eval_every and (step % args.eval_every == 0)):
            tmp_dir = pathlib.Path(cfg.ckpt_dir) / "_tmp_eval"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # ✅ 一定要把 action_rows.safetensors 一起写到 tmp_dir，否则 eval 的动作头会丢
            trainer.policy.model.save_pretrained(str(tmp_dir))
            trainer._save_action_rows(str(tmp_dir))

            ev = _eval_dataset(
                model_name=args.model,
                lora_path=str(tmp_dir),
                data=val_ds,
                max_steps=cfg.max_steps_per_traj,
                check_source=cfg.check_source,
                self_mode=cfg.self_mode,
                grid_n=cfg.grid_n,
                grid_allow_repeat=cfg.grid_allow_repeat,
                fuse_mode=cfg.fuse_mode,
                eta=cfg.eta,
                tau=cfg.tau,
                gamma_gate=cfg.gamma_gate,
                calibrate_tool=cfg.calibrate_tool,
                claim_gamma=cfg.claim_gamma,
                check_alpha=cfg.check_alpha,
                short_edge=args.short_edge,
                load_4bit=args.load_4bit,
                behav_on_cpu=True,   # eval 不需要 GPU behav
                grad_ckpt=False,     # eval 关掉 checkpoint 更快
                seed=args.seed,
                debug_traj=False,
                use_qwen_prior=bool(args.use_qwen_prior),
            )

            score = ev["Brier(mean)"] if args.select_metric == "brier" else ev["ECE(10bins)"]
            print(f"[val@{step}] {json.dumps(ev, ensure_ascii=False)}  select={args.select_metric}:{score}")

            if score is not None and (score + cfg.es_delta < best_score):
                best_score = float(score)
                best_step = int(step)
                best_dir = pathlib.Path(cfg.ckpt_dir) / "best_by_val"
                best_dir.mkdir(parents=True, exist_ok=True)
                trainer.policy.model.save_pretrained(str(best_dir))
                trainer._save_action_rows(str(best_dir))
                print(f"[best] update best_by_val at step={step} score={best_score}")

    # Final save
    out = pathlib.Path(args.save_lora_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.policy.model.save_pretrained(str(out))
    trainer._save_action_rows(str(out))
    print(f"[OK] saved RL LoRA + action_rows to {out}")
    if best_step > 0:
        print(f"[OK] best_by_val: step={best_step}, score={best_score}")


if __name__ == "__main__":
    main()
