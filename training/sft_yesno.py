# training/sft_yesno.py
# -*- coding: utf-8 -*-
"""Simple yes/no SFT for the *base* VLM (via LoRA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import random
import numpy as np
import torch

from training.policy import Policy


@dataclass
class SFTCfg:
    model_name: str
    init_lora: str | None = None

    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 1
    seed: int = 123

    load_4bit: bool = True
    short_edge: int = 384
    grad_ckpt: bool = True

    log_every: int = 20
    eval_every: int = 200
    save_dir: str = "artifacts/ckpt_sft"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device_of(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SFTTrainer:
    def __init__(self, cfg: SFTCfg, behav_on_cpu: bool = False):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.policy = Policy(
            cfg.model_name,
            lora_path=cfg.init_lora,
            short_edge=cfg.short_edge,
            load_4bit=cfg.load_4bit,
            behav_on_cpu=behav_on_cpu,
            grad_ckpt=cfg.grad_ckpt,
        )

        trainable = [p for p in self.policy.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable params. Did LoRA mount correctly?")
        self.opt = torch.optim.AdamW(trainable, lr=cfg.lr)

        self.step = 0

    def _example_label01(self, ex: Dict[str, Any]) -> int:
        # ✅ 优先 soft_label（更通用），否则退化 label
        if "soft_label" in ex:
            y = float(ex["soft_label"])
        else:
            y = float(ex.get("label", 0.0))
        return 1 if y >= 0.5 else 0

    def train_epoch(self, train_ds: List[Dict[str, Any]], val_ds: Optional[List[Dict[str, Any]]] = None):
        self.policy.model.train()
        device = _device_of(self.policy.model)

        running_loss = 0.0
        running_n = 0

        for ex in train_ds:
            y01 = self._example_label01(ex)
            img = ex.get("image", None)
            concept = ex.get("concept") or ex.get("finding") or ex.get("label_name")
            if not concept:
                continue

            batch = self.policy.make_yesno_sft_batch(img, str(concept), y01)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            out = self.policy.model(**batch, use_cache=False)
            loss = out.loss
            if loss is None:
                raise RuntimeError("Model output has no `.loss` - does this model support labels?")

            loss = loss / max(1, int(self.cfg.grad_accum))
            loss.backward()

            self.step += 1
            running_loss += float(loss.detach().item())
            running_n += 1

            if self.step % self.cfg.grad_accum == 0:
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

            if self.cfg.log_every and (self.step % self.cfg.log_every == 0):
                avg = running_loss / max(1, running_n)
                print(f"[sft] step={self.step} loss={avg:.4f}")

            if val_ds and self.cfg.eval_every and (self.step % self.cfg.eval_every == 0):
                stats = self.evaluate(val_ds, limit=400)  # ✅ 评估别太重
                print(f"[sft] val@{self.step}: {stats}")
                self.save(os.path.join(self.cfg.save_dir, f"step_{self.step}"))

        # flush last accum
        if self.step % self.cfg.grad_accum != 0:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self, ds: List[Dict[str, Any]], limit: int | None = None) -> Dict[str, float]:
        self.policy.model.eval()
        device = _device_of(self.policy.model)

        n = 0
        loss_sum = 0.0
        acc_sum = 0.0

        take = ds[: (limit or len(ds))]
        for ex in take:
            y01 = self._example_label01(ex)
            img = ex.get("image", None)
            concept = ex.get("concept") or ex.get("finding") or ex.get("label_name")
            if not concept:
                continue

            batch = self.policy.make_yesno_sft_batch(img, str(concept), y01)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = self.policy.model(**batch, use_cache=False)
            if out.loss is None:
                continue
            loss_sum += float(out.loss.item())

            # ✅ 用 teacher-forcing logprob 判别 yes/no（更一致）
            q = f"Based on the image, is there {concept}? Answer 'yes' or 'no'."
            d = self.policy.yesno_logprobs(img, q)
            pred = 1 if d["yes"] >= d["no"] else 0
            acc_sum += float(pred == y01)

            n += 1

        if n == 0:
            return {"n": 0.0, "loss": float("nan"), "acc": float("nan")}
        return {"n": float(n), "loss": loss_sum / n, "acc": acc_sum / n}

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        try:
            self.policy.model.save_pretrained(out_dir)
            print(f"[sft] saved LoRA to {out_dir}")
        except Exception as e:
            print(f"[sft][warn] save failed: {e}")

    def train(self, train_ds: List[Dict[str, Any]], val_ds: Optional[List[Dict[str, Any]]] = None):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        for ep in range(1, int(self.cfg.epochs) + 1):
            print(f"[sft] epoch {ep}/{self.cfg.epochs}")
            self.train_epoch(train_ds, val_ds=val_ds)
            self.save(os.path.join(self.cfg.save_dir, f"epoch_{ep}"))
