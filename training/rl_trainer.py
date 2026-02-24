# training/rl_trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import random
import math
import json

import numpy as np
import torch

from training.policy import Policy
from training.rl_env import SimpleEnv, EnvCfg
from training.rewards import brier_reward  # R = - (p - g)^2


# -------------------------
# metrics helpers
# -------------------------
def _ece_score(probs: List[float], labels: List[float], n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        if b1 < 1.0:
            mask = (probs >= b0) & (probs < b1)
        else:
            mask = (probs >= b0) & (probs <= b1)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        w = mask.mean()
        ece += w * abs(conf - acc)
    return float(ece)


def _summarize_preds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    ps = [float(r["p"]) for r in rows if math.isfinite(float(r["p"]))]
    gs = [float(r["g"]) for r in rows if math.isfinite(float(r["p"]))]

    briers = [(p - g) ** 2 for p, g in zip(ps, gs)]
    ece = _ece_score(ps, gs, n_bins=10)

    steps = [int(r.get("steps", 0)) for r in rows if math.isfinite(float(r["p"]))]
    check_rate = float(np.mean([bool(r.get("ever_checked", False)) for r in rows])) if rows else 0.0
    adopt_rate = float(np.mean([bool(r.get("adopted", False)) for r in rows])) if rows else 0.0

    return {
        "n": len(rows),
        "Brier(mean)": float(np.mean(briers)) if briers else None,
        "Brier(std)": float(np.std(briers)) if briers else None,
        "ECE(10bins)": float(ece),
        "AvgSteps": float(np.mean(steps)) if steps else None,
        "CheckRate": float(check_rate),
        "AdoptRate": float(adopt_rate),
    }


# -------------------------
# config
# -------------------------
@dataclass
class RLCfg:
    """外环 RL (CISPO-like) 的可调超参（单卡 24G 友好）"""
    # model / init
    model_name: str
    init_lora: str | None = None
    load_4bit: bool = True
    short_edge: int = 384
    behav_on_cpu: bool = True          # ✅ 推荐：行为策略放 CPU，省显存
    grad_ckpt: bool = True

    # optimizer
    lr: float = 1e-4                   # LoRA-only 通常 1e-4~3e-4
    grad_clip: float = 1.0             # 0 表示不裁剪

    # RL objective terms
    kl_coef: float = 0.02
    ent_coef: float = 0.001
    is_clip: float = 5.0               # 截断系数 c
    is_log_clip: float = 20.0          # ✅ 防 exp 溢出：clip(log_w) to [-20,20]
    adv_scale: float = 5.0             # ✅ 优势标准化后整体放大（更有力）

    # sampling
    K: int = 3                         # 组内采样条数（>=2 才有方差）
    batch_size: int = 1
    steps: int = 600
    seed: int = 123
    max_steps_per_traj: int = 3

    # env settings during training
    check_source: str = "prior"        # "none"|"prior"|"kbcs"|"grid"|"self"
    self_mode: str = "grid"
    grid_n: int = 3
    grid_allow_repeat: bool = False

    fuse_mode: str = "mix"             # "none"|"mix"|"gate"
    eta: float = 0.5
    tau: float = 0.12
    gamma_gate: float = 0.35
    calibrate_tool: bool = False
    check_alpha: float = 0.9
    claim_gamma: float = 1.0
    disable_check: bool = False
    use_baseline_when_no_check: bool = True

    # ✅ reward shaping（默认 0，不改变旧行为）
    step_penalty: float = 0.0          # 每走一步扣多少
    check_cost: float = 0.0            # 每次 CHECK 额外扣多少（可用于逼停/控制开销）

    # behavior refresh / logging / ckpt
    refresh_every: int = 50
    warmup: bool = True
    use_tqdm: bool = True
    log_precision: int = 6

    ckpt_every: int = 100
    ckpt_dir: str = "artifacts/ckpt_rl"
    save_action_rows: bool = True

    # ✅ 上限关键：验证集驱动的 early stop + best 保存
    val_path: Optional[str] = None     # jsonl（与训练同格式）
    eval_every: int = 100              # 每隔多少 step 做一次 val 评测
    eval_n: int = 200                  # val 抽样多少条
    best_metric: str = "brier"         # "brier" or "ece"
    es_patience: int = 10              # 连续多少次没提升就停
    es_delta: float = 1e-4             # 认为提升的最小幅度


# -------------------------
# trainer
# -------------------------
class RLTrainer:
    """
    稳定版 CISPO 外环（LoRA-only）：
      - ✅ 行为策略初始化同步（behav=cur），避免 IS/KL 噪声
      - ✅ 用整条轨迹做信用分配（logp/H/KL 累积），CHECK 才学得到
      - ✅ early stop / best ckpt 用验证集指标（Brier/ECE），不是 RL loss

    关键修复（配合新的 Policy.action_head）：
      - ✅ optimizer 必须包含 policy.action_head 的参数，否则动作分布会一直卡在均匀
      - ✅ _save_action_rows 必须保存/恢复 action_head（而不是 lm_head 的 4 行）
    """

    def __init__(self, cfg: RLCfg, use_qwen_prior: bool = False, behav_on_cpu: bool = False):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.policy = Policy(
            cfg.model_name,
            lora_path=cfg.init_lora,
            short_edge=cfg.short_edge,
            load_4bit=cfg.load_4bit,
            behav_on_cpu=cfg.behav_on_cpu,     # legacy flag still supported
            grad_ckpt=cfg.grad_ckpt,
        )

        self.env = SimpleEnv(EnvCfg(
            max_steps=cfg.max_steps_per_traj,
            disable_check=cfg.disable_check,
            use_baseline_when_no_check=cfg.use_baseline_when_no_check,

            check_source=cfg.check_source,
            self_mode=cfg.self_mode,
            grid_n=cfg.grid_n,
            grid_allow_repeat=cfg.grid_allow_repeat,

            fuse_mode=cfg.fuse_mode,
            eta=cfg.eta,
            tau=cfg.tau,
            gamma_gate=cfg.gamma_gate,
            calibrate_tool=cfg.calibrate_tool,

            use_qwen_prior=use_qwen_prior,
            check_alpha=cfg.check_alpha,
            claim_gamma=cfg.claim_gamma,
        ))

        # ✅ 关键：behav 初始必须与当前策略一致（否则 IS/KL = 噪声）
        if cfg.behav_on_cpu:
            try:
                self.policy.refresh_behavior(show_progress=False)
                print("[rl] initial behav sync done")
            except Exception as e:
                print(f"[rl][warn] initial behav sync failed: {e}")

        # -------------------------
        # optimizer params
        # -------------------------
        # 只训练 LoRA 参数 + ✅ action_head
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]

        # ✅ action_head 一定要训：不加入 optimizer 就会长期均匀 (H≈ln4, logp≈ln(1/4))
        if hasattr(self.policy, "action_head"):
            ah_params = list(self.policy.action_head.parameters())
            for p in ah_params:
                p.requires_grad_(True)
            trainable_params.extend(ah_params)

        if not trainable_params:
            raise RuntimeError("没有可训练参数。请确认 Policy 中 LoRA/action_head 已正确挂载并 set requires_grad=True。")

        self.opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
        self._running_baseline: Optional[float] = None
        self._step_i = 0

        # val dataset
        self._val_data: Optional[List[Dict[str, Any]]] = None
        if cfg.val_path:
            self._val_data = self._read_jsonl(cfg.val_path)
            # 给没有 id 的样本补 id
            for i, ex in enumerate(self._val_data):
                ex.setdefault("id", f"val-{i:06d}")

    @staticmethod
    def _read_jsonl(p: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _device(self) -> torch.device:
        # Prefer model device
        for p in self.policy.model.parameters():
            if getattr(p, "device", None) is not None and p.device.type != "meta":
                return p.device
        # fallback: action_head device
        if hasattr(self.policy, "action_head"):
            for p in self.policy.action_head.parameters():
                return p.device
        return torch.device("cpu")

    def _save_action_rows(self, save_dir: str):
        """
        ✅ 保存动作头参数（policy.action_head）到 save_dir/action_rows.safetensors
        供评测/恢复时一致使用。

        NOTE: 旧版是从 lm_head 抠 4 行保存，这会导致动作分布卡死在均匀（尤其是新加 special tokens）。
        """
        if not self.cfg.save_action_rows:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)

            # New policy API: save_action_rows()
            if hasattr(self.policy, "save_action_rows"):
                self.policy.save_action_rows(save_dir)
                print(f"[rl][ckpt] saved action_rows to {save_dir}")
                return

            # Fallback: keep old behavior if needed
            from safetensors.torch import save_file

            lm = self.policy.model.get_output_embeddings()
            action_ids = torch.tensor(
                [self.policy.action_ids[a] for a in ["<CLAIM>", "<CHECK>", "<ABSTAIN>", "<STOP>"]],
                device=lm.weight.device, dtype=torch.long
            )
            rows = lm.weight.detach().index_select(0, action_ids).cpu()  # [4, hidden_dim]
            save_file({"action_rows": rows}, os.path.join(save_dir, "action_rows.safetensors"))

            with open(os.path.join(save_dir, "action_tokens.json"), "w", encoding="utf-8") as f:
                json.dump({"tokens": ["<CLAIM>", "<CHECK>", "<ABSTAIN>", "<STOP>"]}, f, ensure_ascii=False)

            print(f"[rl][ckpt] saved action_rows to {save_dir} (fallback lm_head rows)")
        except Exception as e:
            print(f"[rl][ckpt][warn] save action_rows failed: {e}")

    def _episode_reward(self, p_final: float, g: float, traj: List[Dict[str, Any]], steps: int) -> float:
        """终局 Brier + 可选 shaping（默认 0 不影响）"""
        r = float(brier_reward(p_final, g))
        if self.cfg.step_penalty and steps > 0:
            r -= float(self.cfg.step_penalty) * float(steps)
        if self.cfg.check_cost:
            n_check = 0
            for st in traj:
                a = str(st.get("action", ""))
                if a.upper().strip("<>") == "CHECK":
                    n_check += 1
            r -= float(self.cfg.check_cost) * float(n_check)
        return float(r)

    def _rollout_train(self, ex: Dict[str, Any]) -> Tuple[float, int, List[Dict[str, Any]]]:
        # env.rollout_once 内部会调用 policy.sample_action_train（可回传梯度）
        return self.env.rollout_once(self.policy, ex)

    @torch.no_grad()
    def _rollout_eval(self, ex: Dict[str, Any]) -> Tuple[float, int, List[Dict[str, Any]]]:
        """
        为了少侵入：仍走 env.rollout_once，但把 model 切 eval 并用 no_grad 包裹。
        （env 内部用 sample_action_train 取动作；在 no_grad 下仍可用，只是 logp_t 不需要 grad）
        """
        was_train = self.policy.model.training
        self.policy.model.eval()
        try:
            p_final, steps, traj = self.env.rollout_once(self.policy, ex)
        finally:
            if was_train:
                self.policy.model.train()
        return p_final, steps, traj

    def eval_on_val(self, n: int = 200, seed: int = 123) -> Dict[str, Any]:
        if not self._val_data:
            return {"n": 0}

        rng = random.Random(seed)
        data = self._val_data
        k = min(int(n), len(data))
        batch = rng.sample(data, k=k)

        rows: List[Dict[str, Any]] = []
        for ex in batch:
            try:
                p_final, steps, traj = self._rollout_eval(ex)
                p_final = float(np.clip(float(p_final), 0.0, 1.0))
                g = float(np.clip(float(ex.get("soft_label", ex.get("label", 0.0))), 0.0, 1.0))

                ever_checked = any(str(st.get("action", "")).upper().strip("<>") == "CHECK" for st in traj)

                adopted = False
                for st in traj:
                    if "adopted" in st:
                        try:
                            adopted = adopted or bool(int(st.get("adopted", 0)))
                        except Exception:
                            adopted = adopted or bool(st.get("adopted"))
                rows.append({
                    "id": ex.get("id"),
                    "p": p_final,
                    "g": g,
                    "steps": int(steps),
                    "ever_checked": bool(ever_checked),
                    "adopted": bool(adopted),
                })
            except Exception:
                continue

        return _summarize_preds(rows)

    def step_batch(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        device = self._device()
        pg_terms: List[torch.Tensor] = []
        ent_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        all_As_for_norm: List[float] = []
        any_var = False

        for ex in batch:
            g = float(ex.get("soft_label", ex.get("label", 0.0)))
            g = max(0.0, min(1.0, g))

            grp: List[Dict[str, Any]] = []
            for _ in range(int(self.cfg.K)):
                p_final, steps, traj = self._rollout_train(ex)
                if not traj:
                    continue

                # ✅ 整条轨迹信用分配：累积 logp/H/KL + logp_behav
                logp_t_sum = None
                H_sum = None
                KL_sum = None
                logp_b_sum = 0.0

                ok = True
                for st in traj:
                    try:
                        lt = st["logp_cur_t"]
                        ht = st["H_t"]
                        kt = st["KL_t"]
                        lb = float(st["logp_behav"])
                        if (not torch.isfinite(lt)) or (not torch.isfinite(ht)) or (not torch.isfinite(kt)):
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break

                    logp_t_sum = lt if logp_t_sum is None else (logp_t_sum + lt)
                    H_sum = ht if H_sum is None else (H_sum + ht)
                    KL_sum = kt if KL_sum is None else (KL_sum + kt)
                    logp_b_sum += lb

                if (not ok) or (logp_t_sum is None) or (H_sum is None) or (KL_sum is None):
                    continue

                R = self._episode_reward(float(p_final), g, traj, int(steps))
                grp.append({
                    "logp_t": logp_t_sum,        # tensor
                    "logp_b": float(logp_b_sum), # float
                    "H_t": H_sum,                # tensor
                    "KL_t": KL_sum,              # tensor
                    "R": float(R),
                })

            if not grp:
                continue

            # 组内 baseline（self-critical）
            R_bar = float(sum(t["R"] for t in grp) / max(1, len(grp)))
            for t in grp:
                t["A"] = float(t["R"] - R_bar)

            As = [t["A"] for t in grp]
            varA = float(np.var(As))

            # fallback：运行 baseline（EMA）
            if varA < 1e-12:
                alpha = 0.05
                R_mean = R_bar
                if self._running_baseline is None:
                    self._running_baseline = R_mean
                else:
                    self._running_baseline = (1 - alpha) * self._running_baseline + alpha * R_mean
                for t in grp:
                    t["A"] = float(t["R"] - float(self._running_baseline))
                As = [t["A"] for t in grp]
                varA = float(np.var(As))
                if varA < 1e-12:
                    continue

            any_var = True
            all_As_for_norm.extend(As)

            # CISPO：重要性权重 + 截断（✅ 防 exp 溢出）
            for t in grp:
                logp_t_det = float(t["logp_t"].detach().item())
                log_w = logp_t_det - float(t["logp_b"])
                clipv = float(self.cfg.is_log_clip)
                if clipv > 0:
                    log_w = float(np.clip(log_w, -clipv, clipv))
                w = float(np.exp(log_w))
                w_hat = float(min(w, float(self.cfg.is_clip)))

                # maximize => minimize negative
                pg_terms.append(-(w_hat * float(t["A"])) * t["logp_t"])
                ent_terms.append(t["H_t"])
                kl_terms.append(t["KL_t"])

        if (not any_var) or (not pg_terms):
            return None

        # ✅ 优势归一化 + 放大（更稳定、更强学习信号）
        A_std = float(np.std(all_As_for_norm))
        A_std = max(A_std, 1e-6)
        scale = float(self.cfg.adv_scale)
        pg_terms = [(t / A_std) * scale for t in pg_terms]

        loss_pg = torch.stack(pg_terms).mean().to(device)
        H_mean = torch.stack(ent_terms).mean().to(device) if ent_terms else torch.tensor(0.0, device=device)
        KL_mean = torch.stack(kl_terms).mean().to(device) if kl_terms else torch.tensor(0.0, device=device)

        loss = loss_pg - float(self.cfg.ent_coef) * H_mean + float(self.cfg.kl_coef) * KL_mean

        if not torch.isfinite(loss):
            print("[rl][warn] loss 非有限（NaN/Inf），跳过本步。")
            self.opt.zero_grad(set_to_none=True)
            return None

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip and float(self.cfg.grad_clip) > 0:
            try:
                # ✅ clip LoRA + action_head gradients
                params = [p for p in self.policy.model.parameters() if p.requires_grad]
                if hasattr(self.policy, "action_head"):
                    params += list(self.policy.action_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.cfg.grad_clip))
            except Exception as e:
                print(f"[rl][warn] grad clip 失败：{e}")

        self.opt.step()

        self._step_i += 1
        if (self._step_i % 50 == 0) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return {
            "loss": float(loss.item()),
            "loss_pg": float(loss_pg.item()),
            "H": float(H_mean.item()),
            "KL": float(KL_mean.item()),
        }

    def _metric_value(self, eval_stats: Dict[str, Any]) -> Optional[float]:
        if not eval_stats or not eval_stats.get("n"):
            return None
        if self.cfg.best_metric.lower() == "ece":
            return float(eval_stats.get("ECE(10bins)", 1e9))
        # default brier
        return float(eval_stats.get("Brier(mean)", 1e9))

    def train(self, dataset: List[Dict[str, Any]]):
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **kwargs):  # type: ignore
                return x

        use_tqdm = bool(self.cfg.use_tqdm)
        bs = max(1, int(self.cfg.batch_size))
        refresh_every = max(0, int(self.cfg.refresh_every))

        # 补 id
        for i, ex in enumerate(dataset):
            ex.setdefault("id", f"ex-{i:06d}")

        # warmup：尽早发现依赖/图构建问题
        if self.cfg.warmup:
            try:
                _ = self.policy.sample_action_eval(prompt="warmup", image=None)
                print("[rl] warmup forward done")
            except Exception as e:
                print(f"[rl] warmup skipped: {e}")

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        best_val = float("inf")
        bad = 0
        last_eval = None

        iterator = range(1, int(self.cfg.steps) + 1)
        pbar = tqdm(iterator, desc="RL steps", dynamic_ncols=True, disable=not use_tqdm)

        for step in pbar:
            if not dataset:
                print("[rl][error] dataset 为空")
                break

            batch = random.sample(dataset, k=min(bs, len(dataset)))
            stats = self.step_batch(batch)

            if stats is not None and use_tqdm:
                prec = max(0, int(self.cfg.log_precision))
                postfix = {
                    "loss": f"{stats['loss']:.3f}",
                    "KL": f"{stats['KL']:.{prec}f}",
                    "H": f"{stats['H']:.3f}",
                }
                if torch.cuda.is_available():
                    try:
                        mem = torch.cuda.memory_allocated() / (1024 ** 3)
                        postfix["mem"] = f"{mem:.1f}G"
                    except Exception:
                        pass
                if last_eval is not None:
                    postfix["val"] = f"{last_eval:.4f}"
                pbar.set_postfix(postfix)

            # ✅ 行为策略同步（LoRA-only + action_head）
            if refresh_every and (step % refresh_every == 0):
                try:
                    self.policy.refresh_behavior(show_progress=use_tqdm)
                except Exception as e:
                    print(f"[rl][warn] refresh_behavior failed: {e}")

            # 周期保存（按 step）
            if self.cfg.ckpt_every and (step % int(self.cfg.ckpt_every) == 0):
                try:
                    sd = os.path.join(self.cfg.ckpt_dir, f"step_{step}")
                    self.policy.model.save_pretrained(sd)
                    self._save_action_rows(sd)
                except Exception as e:
                    print(f"[rl][ckpt] save step_{step} failed: {e}")

            # ✅ 上限关键：验证集评测驱动 best / early stop
            if self._val_data and self.cfg.eval_every and (step % int(self.cfg.eval_every) == 0):
                try:
                    eval_stats = self.eval_on_val(n=int(self.cfg.eval_n), seed=self.cfg.seed + step)
                    mv = self._metric_value(eval_stats)
                    if mv is None:
                        continue

                    last_eval = mv
                    # 记录 eval
                    with open(os.path.join(self.cfg.ckpt_dir, "val_log.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, **eval_stats}, ensure_ascii=False) + "\n")

                    improved = (mv < best_val - float(self.cfg.es_delta))
                    if improved:
                        best_val = mv
                        bad = 0
                        sd = os.path.join(self.cfg.ckpt_dir, "best")
                        self.policy.model.save_pretrained(sd)
                        self._save_action_rows(sd)
                        with open(os.path.join(sd, "best_val_metrics.json"), "w", encoding="utf-8") as f:
                            f.write(json.dumps(eval_stats, ensure_ascii=False, indent=2))
                        print(f"[rl][best] step={step} best_{self.cfg.best_metric}={best_val:.6f}")
                    else:
                        bad += 1
                        if bad >= int(self.cfg.es_patience):
                            print(f"[rl] early stop at step {step} (best_{self.cfg.best_metric}={best_val:.6f})")
                            break
                except Exception as e:
                    print(f"[rl][warn] eval_on_val failed: {e}")
                    continue
