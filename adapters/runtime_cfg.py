# adapters/runtime_cfg.py
# -*- coding: utf-8 -*-
"""
runtime_cfg.py — 运行时配置与常用工具
职责：
- 统一读取 configs/ 下的 calib.json / kbcs_thr.json / lexicon.json / prompt_template.md / tool_runtime.json
- 概率校准：把 margin (log-odds) -> 概率（温度缩放或保序查表）
- 阈值获取：按 concept 取 margin 空间阈值（thr_pos/thr_neg），并处理全局回退
"""

from __future__ import annotations
import json
import math
import pathlib
from typing import Dict, Tuple

# 项目根目录（newProject/）
ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

def _jload(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_all() -> Dict[str, object]:
    """一次性加载所有运行时配置，缺失就用空/默认。"""
    thr = _jload(CFG / "kbcs_thr.json")
    calib = _jload(CFG / "calib.json")
    lex = _jload(CFG / "lexicon.json")
    tmpl = ""
    pt = CFG / "prompt_template.md"
    if pt.exists():
        try:
            tmpl = pt.read_text(encoding="utf-8")
        except Exception:
            tmpl = ""
    tool = _jload(CFG / "tool_runtime.json")
    return {"thr": thr, "calib": calib, "lex": lex, "tmpl": tmpl, "tool": tool}

# ----------------- 校准相关 -----------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _get_T(concept: str, calib: dict) -> float:
    # 支持全局 / 按概念 / 按视位(view) / 按站点(sites) 的回退；若没提供就返回 1.0
    if not isinstance(calib, dict):
        return 1.0
    # 直接按概念
    cT = calib.get("concepts", {}).get(str(concept), {})
    if isinstance(cT, dict) and "T" in cT:
        return float(cT["T"])
    # 全局
    g = calib.get("global", {})
    if isinstance(g, dict) and "T" in g:
        return float(g["T"])
    return 1.0

def _isotonic_lookup(margin: float, table: dict) -> float:
    """保序回归查表（可选）：table={"knots":[...], "probs":[...]}"""
    knots = table.get("knots") or []
    probs = table.get("probs") or []
    if not knots or not probs or len(knots) != len(probs):
        return _sigmoid(margin)
    # 线性插值
    x = float(margin)
    if x <= knots[0]:
        return float(probs[0])
    if x >= knots[-1]:
        return float(probs[-1])
    # 找到区间
    lo = 0
    hi = len(knots) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x >= knots[mid]:
            lo = mid
        else:
            hi = mid
    x0, x1 = float(knots[lo]), float(knots[hi])
    y0, y1 = float(probs[lo]), float(probs[hi])
    t = (x - x0) / max(1e-12, (x1 - x0))
    return float(y0 * (1 - t) + y1 * t)

def calibrate_margin_to_p(margin: float, concept: str, calib: dict) -> float:
    """
    将 margin (log-odds) 转为概率。
    支持两种方法：
      - temperature（默认）：p = sigmoid(margin / T)
      - isotonic：在 isotonic_tables[concept] 查表，否则回退 sigmoid
    """
    method = str(calib.get("method", "temperature")).lower()
    if method == "isotonic":
        tab = (calib.get("isotonic_tables") or {}).get(str(concept))
        if isinstance(tab, dict):
            return _isotonic_lookup(float(margin), tab)
        return _sigmoid(float(margin))
    # temperature
    T = max(1e-6, float(_get_T(str(concept), calib)))
    return _sigmoid(float(margin) / T)

# ----------------- 阈值相关（统一在 margin 空间） -----------------

def get_thr(concept: str, thr_cfg: dict) -> Tuple[float, float]:
    """
    返回 (thr_pos, thr_neg)，单位都是 margin/log-odds。
    回退逻辑：
      - concepts[concept].thr_pos / thr_neg
      - global.default_thr_pos / default_thr_neg
      - 默认 (0.1, -0.4)
    """
    g = (thr_cfg or {}).get("global", {}) or {}
    c = (thr_cfg or {}).get("concepts", {}).get(str(concept), {}) or {}
    thr_pos = float(c.get("thr_pos", g.get("default_thr_pos", 0.1)))
    thr_neg = float(c.get("thr_neg", g.get("default_thr_neg", -0.4)))
    return thr_pos, thr_neg
