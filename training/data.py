from __future__ import annotations
from typing import List, Dict, Any
import json, pathlib
from eval.evaluate_dual_env import _maybe_load_dataset as _eval_loader  # 直接复用

def load_examples(src: str | list[dict]) -> List[Dict[str, Any]]:
    # 1) 若传的是 VinDr 根目录，复用评测的构建器（会自动补 soft_label / p_baseline / box）
    if isinstance(src, str) and pathlib.Path(src).is_dir():
        return _eval_loader(src)
    # 2) 若是 *.jsonl / *.json
    if isinstance(src, str) and (src.endswith(".jsonl") or src.endswith(".json")):
        txt = pathlib.Path(src).read_text(encoding="utf-8")
        return json.loads(txt) if txt.strip().startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    # 3) 直接 list[dict]
    if isinstance(src, list):
        return src
    raise ValueError(f"Unsupported dataset source: {src}")
