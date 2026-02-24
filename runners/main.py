# runners/main.py
# -*- coding: utf-8 -*-
"""
统一 CLI 入口

用法示例：
  # 评测（等价于 runners/run_seal.py 的流程，写 release 目录和 eval_log.jsonl）
  python -m runners.main eval --data /path/to/vinDr_or_eval.jsonl

  # 训练 head（BiomedCLIP 头，冻结图像编码器）
  python -m runners.main train_head --vindr_root /path/to/vinDr --out artifacts/vhead/biomedclip

  # 应用 SEAL edits（示例）
  python -m runners.main seal --edits configs/edits_example.json

  # 强化学习外环（RL）
  python -m runners.main rl \
    --model /home/neutron/sdc/MODEL/Qwen2.5-VL-3B-Instruct \
    --data artifacts/rl/rl_vindr.jsonl \
    --steps 200 --K 1 --max_steps_per_traj 2 \
    --lr 1e-4 --kl 0.02 --ent 0.005 \
    --use_qwen_prior
"""
from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

# ========= 工程根目录 & sys.path 保护 =========
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ===========================================

# 只导入评测函数（其余 training.* 在各自分支内延迟导入）
from eval.evaluate_dual_env import evaluate, evaluate_policy_init, _maybe_load_dataset


# ---------- 通用工具 ----------
def _backup_configs_to_release(rel_dir: pathlib.Path) -> None:
    """备份关键配置到 release 目录（便于复现）"""
    rel_dir.mkdir(parents=True, exist_ok=True)
    # 必备：pipeline.yaml（若存在）
    p = pathlib.Path("configs/pipeline.yaml")
    if p.exists():
        shutil.copy(str(p), rel_dir / "pipeline.yaml")
    # 可选：校准与运行时配置
    for f in ("configs/calib_kbcs.json", "configs/tool_runtime.json"):
        p = pathlib.Path(f)
        if p.exists():
            shutil.copy(str(p), rel_dir / p.name)
    # 记录 git 版本
    try:
        gitv = subprocess.getoutput("git rev-parse --short HEAD").strip()
    except Exception:
        gitv = "nogit"
    (rel_dir / "git.txt").write_text(gitv + "\n", encoding="utf-8")


def _seal_eval_once(data_root_or_json: str) -> dict:
    """
    等价于 runners/run_seal.py 的主流程：
    - 读取 configs/pipeline.yaml 的 RELEASE_DIR（若不存在则用 artifacts/release）
    - 新建时间戳目录
    - evaluate(..., log_path=...)，写 release.json 并回显
    """
    cfg = {}
    p_cfg = pathlib.Path("configs/pipeline.yaml")
    if p_cfg.exists():
        cfg = yaml.safe_load(open(p_cfg, "r"))
        rel_base = pathlib.Path(cfg.get("PIPELINE", {}).get("RELEASE_DIR", "artifacts/release"))
    else:
        rel_base = pathlib.Path("artifacts/release")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = rel_base / stamp
    rel.mkdir(parents=True, exist_ok=True)

    _backup_configs_to_release(rel)

    # 评测（把 log_path 传进 evaluator，便于后续温标/门控拟合脚本使用）
    log_path = rel / "eval_log.jsonl"
    metrics = evaluate(dataset_or_path=data_root_or_json, log_path=str(log_path))

    (rel / "release.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    print(f"[OK] wrote {rel / 'release.json'}")
    return metrics


def _eval_triplet_once(data_root_or_json: str, model_name: str, lora_path: str | None,
                       max_steps: int, n_eval: int | None = None) -> dict:
    """
    生成一个新的 release 目录，写入：
      - eval_log.jsonl（ZS/ZS+Tool 的流水）
      - eval_policy_init_log.jsonl（Policy(init) 的流水）
      - release.json（合并三基线指标）
    """
    cfg = {}
    p_cfg = pathlib.Path("configs/pipeline.yaml")
    if p_cfg.exists():
        cfg = yaml.safe_load(open(p_cfg, "r"))
        rel_base = pathlib.Path(cfg.get("PIPELINE", {}).get("RELEASE_DIR", "artifacts/release"))
    else:
        rel_base = pathlib.Path("artifacts/release")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = rel_base / stamp
    rel.mkdir(parents=True, exist_ok=True)
    _backup_configs_to_release(rel)

    # 1) ZS（no-tool）+ ZS+Tool
    log_no_tool_and_tool = rel / "eval_log.jsonl"
    m_zs = evaluate(dataset_or_path=data_root_or_json, log_path=str(log_no_tool_and_tool))

    # 2) Policy(init)-no LoRA
    log_policy_n1 = rel / "eval_policy_init_no_lora_log.jsonl"
    m_pi_n1 = evaluate_policy_init(
        dataset_or_path=data_root_or_json,
        model_name=model_name,
        lora_path=None,
        max_steps=max_steps,
        log_path=str(log_policy_n1),
        n_limit=n_eval,
    )
    m_pi_lora = None
    if lora_path:
        log_policy_1 = rel / "eval_policy_lora_log.jsonl"
        m_pi_lora = evaluate_policy_init(
        dataset_or_path=data_root_or_json,
        model_name=model_name,
        lora_path=lora_path,
        max_steps=max_steps,
        log_path=str(log_policy_1),
        n_limit=n_eval,
    )
    # 3) 合并并写 release.json
    merged = dict(m_zs)
    merged["policy_init"] = m_pi_n1
    if lora_path:
        merged["policy_lora"] = m_pi_lora
    (rel / "release.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False))
    print(f"[OK] wrote {rel / 'release.json'}")
    return merged


def _read_jsonl(p: str) -> list[dict]:
    txt = pathlib.Path(p).read_text(encoding="utf-8")
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def _load_examples(src: str | list[dict]) -> list[dict]:
    """
    统一数据加载器：
      - 目录：按评测构建（_maybe_load_dataset）
      - .json/.jsonl：读取为 list[dict]
      - list[dict]：直接返回
    若存在 training.data.load_examples，则优先用它。
    """
    # 优先用项目自带的加载器（若有）
    try:
        from training.data import load_examples as _loader
        return _loader(src)
    except Exception:
        pass

    # 兜底：本地实现
    if isinstance(src, str):
        p = pathlib.Path(src)
        if p.is_dir():
            return _maybe_load_dataset(src)
        if p.suffix.lower() in {".json", ".jsonl"}:
            txt = p.read_text(encoding="utf-8")
            return json.loads(txt) if txt.strip().startswith("[") else [
                json.loads(l) for l in txt.splitlines() if l.strip()
            ]
    if isinstance(src, list):
        return src
    raise ValueError(f"Unsupported dataset source: {src}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(prog="runners.main", description="Unified CLI for eval / head / seal / rl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval：整合 run_seal 主流程
    p_eval = sub.add_parser("eval", help="Evaluate & write a release directory (integrated run_seal).")
    p_eval.add_argument("--data", required=True, help="VinDr-CXR root or a dataset json/jsonl path")

    # 训练 head（BiomedCLIP 头）
    p_head = sub.add_parser("train_head", help="Train a lightweight vision head (BiomedCLIP via OpenCLIP, image encoder frozen).")
    p_head.add_argument("--out", default=str(ROOT / "artifacts/vhead/biomedclip"))
    p_head.add_argument("--vindr_root", required=True, help="VinDr-CXR root (has Annotations/, train/, test/)")
    p_head.add_argument("--split", default="train", choices=["train", "test"])
    p_head.add_argument("--batch_size", type=int, default=32)
    p_head.add_argument("--epochs", type=int, default=3)
    p_head.add_argument("--lr", type=float, default=1e-3)
    p_head.add_argument("--limit_per_concept", type=int, default=200)
    p_head.add_argument("--use_linear_adapter", action="store_true")
    p_head.add_argument("--biomedclip_dir", default="/home/neutron/sdc/MODEL/BiomedCLIP-PubMedBERT",
                        help="Local folder containing open_clip_config.json & open_clip_pytorch_model.bin")

    # seal：占位示例
    p_seal = sub.add_parser("seal", help="Apply SEAL edits then eval/accept (example placeholder).")
    p_seal.add_argument("--edits", required=True, help="JSON with edits to apply (e.g., kbcs_thr.json contents)")

    # rl：强化学习外环（把参数透传给 RLTrainer/RLCfg）
    p_rl = sub.add_parser("rl", help="Train RL policy (outer loop).")
    p_rl.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen2.5-VL-3B-Instruct")
    p_rl.add_argument("--data", required=True, help="VinDr root OR JSON/JSONL of RL examples")
    p_rl.add_argument("--init_lora", default=None, help="Path to init LoRA adapter (optional)")
    # —— 推荐默认更利于在 24G 上迭代；都可用命令行覆盖
    p_rl.add_argument("--steps", type=int, default=200)
    p_rl.add_argument("--K", type=int, default=1)
    p_rl.add_argument("--lr", type=float, default=1e-4)
    p_rl.add_argument("--kl", type=float, default=0.02)
    p_rl.add_argument("--ent", type=float, default=0.005)
    p_rl.add_argument("--max_steps_per_traj", type=int, default=2)
    p_rl.add_argument("--use_qwen_prior", action="store_true")
    p_rl.add_argument("--batch_size", type=int, default=1)
    p_rl.add_argument("--refresh_every", type=int, default=0)
    p_rl.add_argument("--es_patience", type=int, default=100)
    p_rl.add_argument("--es_delta", type=float, default=1e-4)
    p_rl.add_argument("--behav_on_cpu", action="store_true", help="Put behavior policy on CPU (default: GPU 4bit).")

    # eval3：一次跑 ZS（no-tool）/ ZS+Tool / Policy(init)
    p_eval3 = sub.add_parser("eval3", help="Run ZS (no-tool), ZS+Tool, and Policy(init) baselines into one release dir.")
    p_eval3.add_argument("--data", required=True, help="VinDr-CXR root or a dataset json/jsonl path")
    p_eval3.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF model name for Policy(init)")
    p_eval3.add_argument("--lora", default=None, help="Optional LoRA path for Policy(init)")
    p_eval3.add_argument("--max_steps", type=int, default=3, help="Max steps per trajectory for Policy(init)")
    p_eval3.add_argument("--n_eval", type=int, default=None, help="Optional sample cap for quick smoke test")

    args = ap.parse_args()

    if args.cmd == "eval":
        _seal_eval_once(args.data)

    elif args.cmd == "train_head":
        try:
            # 延迟导入（避免没用到也触发导入错误）
            from training.head_train import train_head, TrainConfig
        except ModuleNotFoundError:
            print("[ERR] cannot import training.head_train — 请确认文件存在 newProject/training/head_train.py")
            raise

        cfg = TrainConfig(
            out_dir=args.out,
            vindr_root=args.vindr_root,
            split=args.split,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            limit_per_concept=args.limit_per_concept,
            use_linear_adapter=args.use_linear_adapter,
            biomedclip_dir=args.biomedclip_dir,
        )
        train_head(cfg)

    elif args.cmd == "seal":
        from training.seal_inner_loop import InnerLoop, Gate
        edits = json.loads(pathlib.Path(args.edits).read_text(encoding="utf-8"))
        loop = InnerLoop(gate=Gate())
        # TODO: 把 dataset 与基线指标接上你真实的验证集
        before = {"brier_tool": 0.2, "evidence_lift": 0.0}
        after = loop.apply_and_eval(dataset=[], edits=edits)
        ok = loop.accept_or_revert(before, after, edited_files=list(edits.keys()))
        print(json.dumps({"accepted": ok, "metrics": after}, ensure_ascii=False))

    elif args.cmd == "eval3":
        _eval_triplet_once(
            data_root_or_json=args.data,
            model_name=args.model,
            lora_path=args.lora,
            max_steps=args.max_steps,
            n_eval=args.n_eval,
        )

    elif args.cmd == "rl":
        ds = _load_examples(args.data)
        print(f"[rl] loaded {len(ds)} examples from {args.data}")
        if len(ds) == 0:
            print("[ERR] dataset is empty — 请检查 --data 路径或 JSONL 内容")
            sys.exit(2)
        # ---- 统一数据加载器：支持 VinDr 目录 / .jsonl / .json / 直接 list[dict]
        ds = _load_examples(args.data)

        # 构造 RL 配置（兼容老/新 RLCfg）
        try:
            from training.rl_trainer import RLTrainer, RLCfg
        except ModuleNotFoundError:
            print("[ERR] cannot import training.rl_trainer — 请确认文件存在 newProject/training/rl_trainer.py")
            raise

        try:
            cfg = RLCfg(
                model_name=args.model,
                init_lora=args.init_lora,
                steps=args.steps,
                K=args.K,
                lr=args.lr,
                kl_coef=args.kl,
                ent_coef=args.ent,
                max_steps_per_traj=args.max_steps_per_traj,
                batch_size=args.batch_size,
                refresh_every=args.refresh_every,
                es_patience=args.es_patience,
                es_delta=args.es_delta,
            )
        except TypeError:
            # 兼容老版 RLCfg：只传最小集合，其余若有则 setattr
            cfg = RLCfg(
                model_name=args.model,
                init_lora=args.init_lora,
                steps=args.steps,
                K=args.K,
                lr=args.lr,
                kl_coef=args.kl,
                ent_coef=args.ent,
                max_steps_per_traj=args.max_steps_per_traj,
            )
            for name in ("batch_size", "refresh_every", "es_patience", "es_delta"):
                if hasattr(cfg, name):
                    setattr(cfg, name, getattr(args, name))

        # 行为策略默认不上 CPU（= 上 GPU 4bit），除非你显式 --behav_on_cpu
        try:
            trainer = RLTrainer(cfg, use_qwen_prior=args.use_qwen_prior, behav_on_cpu=bool(args.behav_on_cpu))
        except TypeError:
            trainer = RLTrainer(cfg, use_qwen_prior=args.use_qwen_prior)

        trainer.train(ds)

        # 保存 LoRA（若挂了 PEFT）
        out = pathlib.Path("artifacts/lora/rl-cispo-v1")
        out.mkdir(parents=True, exist_ok=True)
        try:
            trainer.policy.model.save_pretrained(str(out))
            print(f"[OK] saved RL LoRA to {out}")
        except Exception as e:
            print(f"[WARN] saving LoRA failed: {e}")

    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
