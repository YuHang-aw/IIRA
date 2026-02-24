"""Schema-constrained decoding for actions.
- 约束解码保证生成合法 JSON；失败时回退 STOP。
- 这里提供占位实现：正则+后验校验；可替换为 jsonformer/rail 等更强方案。
"""
from __future__ import annotations
import json, re
from typing import Dict, Any
from .action_schema import Action

# 简单正则提取 JSON 块（占位；生产可用更健壮的解析器）
JSON_RE = re.compile(r"\{[\s\S]*?\}")

def decode_to_action(text: str) -> Action:
    match = JSON_RE.search(text)
    if not match:
        return Action(action="STOP")
    try:
        js: Dict[str, Any] = json.loads(match.group(0))
    except Exception:
        return Action(action="STOP")
    return Action.from_json(js)
