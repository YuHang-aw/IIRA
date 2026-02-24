# runners/train_yesno.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, random
from typing import Any, Dict, List, Optional

from training.sft_yesno import SFTCfg, SFTTrainer


def _read_jsonl(p: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)

    # ✅ 兼容：--data / --train 都可
    ap.add_argument("--data", default=None, help="train jsonl")
    ap.add_argument("--train", default=None, help="train jsonl (alias of --data)")
    ap.add_argument("--val", default=None, help="val jsonl (optional)")

    ap.add_argument("--init_lora", default=None)
    ap.add_argument("--save_lora_dir", default="artifacts/lora/sft_yesno")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--short_edge", type=int, default=384)

    # ✅ 不要把 cfg 默认值踩掉：用 BooleanOptionalAction（py3.10有）
    ap.add_argument("--load_4bit", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=None)

    args = ap.parse_args()

    train_path = args.data or args.train
    if not train_path:
        raise SystemExit("ERROR: please provide --data (or --train)")

    train = _read_jsonl(train_path, limit=args.limit)
    val = _read_jsonl(args.val) if args.val else None

    random.seed(args.seed)
    random.shuffle(train)

    # ✅ 让 cfg 的默认值生效；只有当用户显式传参时才覆盖
    cfg_kwargs = dict(
        model_name=args.model,
        init_lora=args.init_lora,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        short_edge=args.short_edge,
        save_dir=args.save_lora_dir,
    )
    if args.load_4bit is not None:
        cfg_kwargs["load_4bit"] = bool(args.load_4bit)
    if args.grad_ckpt is not None:
        cfg_kwargs["grad_ckpt"] = bool(args.grad_ckpt)

    cfg = SFTCfg(**cfg_kwargs)

    trainer = SFTTrainer(cfg)
    out_dir = trainer.train(train, val)
    print(f"[OK] saved to {out_dir}")


if __name__ == "__main__":
    main()
