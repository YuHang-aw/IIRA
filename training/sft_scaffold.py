"""少量 LoRA‑SFT（仅文本侧）
- 只用于“格式/模板/少量演示”的记忆，不教医学事实
- 规模：几十~几百步即可
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

@dataclass
class SFTCfg:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    lr: float = 5e-5
    steps: int = 200

def run_sft(cfg: SFTCfg, jsonl_path: str, out_dir: str):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    proc = AutoProcessor.from_pretrained(cfg.model_name)  # 方便你做带图 fewshot
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    added = tok.add_special_tokens({"additional_special_tokens": ["<CLAIM>","<CHECK>","<ABSTAIN>","<STOP>"]})
    if added > 0:
        model.resize_token_embeddings(len(tok))

    # 只打“文本侧”LoRA：target_modules 只匹配 language_model 里的投影层
    lora = LoraConfig(
        r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
    )
    model = get_peft_model(model, lora)

    model.save_pretrained(out_dir)
