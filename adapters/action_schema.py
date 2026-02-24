"""Action schema utilities: validation + dataclasses.
- 我们只在 RL/打分时关注 "action" 字段（四类），concept/roi 作为参数但不进优化。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from jsonschema import Draft202012Validator
import json
import pathlib

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "action.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)

@dataclass
class Action:
    action: str  # "CLAIM" | "CHECK" | "ABSTAIN" | "STOP"
    concept: Optional[str] = None
    polarity: Optional[str] = None  # present/absent/uncertain
    roi: Optional[List[float]] = None  # [x1,y1,x2,y2] or None

    @staticmethod
    def from_json(js: Dict[str, Any]) -> "Action":
        # 若不合法，回退为 STOP，避免策略跑飞
        errors = sorted(VALIDATOR.iter_errors(js), key=lambda e: e.path)
        if errors:
            return Action(action="STOP")
        return Action(
            action=js.get("action"),
            concept=js.get("concept"),
            polarity=js.get("polarity"),
            roi=js.get("roi"),
        )
