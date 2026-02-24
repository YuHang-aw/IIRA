# adapters/kbcs_check.py
# -*- coding: utf-8 -*-
"""
KBCS 检验器（证据服务版）
--------------------------------------------
新增：
- 读取 runtime.tool.device（如 "cpu"/"cuda"）并传给 XRVGradCAMHead（若支持）
- 读取 runtime.tool.image.short_edge（默认 384），推理前按短边等比缩放
- provenance 中记录 device / image_short_edge
- 其它逻辑保持不变；external_calib=true 时不在此处做温度校准
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import json
import pathlib
import hashlib
import math

import numpy as np
from PIL import Image

from .runtime_cfg import load_all, calibrate_margin_to_p, get_thr
from .heatmap_roi import HeatmapHead, ROIProposer,XRVGradCAMHead
import os, json, torch
from transformers import AutoProcessor, AutoModel

import open_clip

_VHEAD_CACHE = {"loaded": False}

def _load_vhead_if_any():
    if _VHEAD_CACHE.get("loaded", False):
        return _VHEAD_CACHE
    cand = ["artifacts/vhead/biomedclip/vhead.pt", "artifacts/head/vhead.pt"]
    path = next((p for p in cand if os.path.exists(p)), None)
    if path is None:
        _VHEAD_CACHE["loaded"] = True; _VHEAD_CACHE["ok"] = False
        return _VHEAD_CACHE

    ckpt = torch.load(path, map_location="cpu")
    concepts = ckpt["concepts"]; dim = int(ckpt["dim"])
    use_adapter = bool(ckpt.get("use_linear_adapter", False))
    model_dir = ckpt.get("model_dir", "/home/neutron/sdc/MODEL/BiomedCLIP-PubMedBERT")

    # 加载本地 BiomedCLIP(OpenCLIP)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        vis, preprocess, _ = open_clip.create_model_from_pretrained(model_dir, device=dev)
    except Exception:
        weight = os.path.join(model_dir, "open_clip_pytorch_model.bin")
        vis, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=weight, device=dev)
    vis.eval()
    for p in vis.parameters(): p.requires_grad_(False)

    # 小头
    import torch.nn as nn
    class ConceptHead(nn.Module):
        def __init__(self, dim, n_concepts, use_linear_adapter=False):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(10.0)); self.bias = nn.Parameter(torch.zeros(n_concepts))
            self.adapter = nn.Linear(dim, dim, bias=False) if use_linear_adapter else nn.Identity()
        def forward(self, img_emb, txt_emb, cid):
            img2 = self.adapter(img_emb); img2 = img2 / (img2.norm(dim=-1, keepdim=True)+1e-8)
            sim = (img2 * txt_emb).sum(-1)
            return self.scale * sim + self.bias[cid]
    head = ConceptHead(dim, len(concepts), use_linear_adapter=use_adapter)
    head.load_state_dict(ckpt["state_dict"])
    head.to(dev).eval()
    txt_emb = ckpt["txt_emb"].to(torch.float32).to(dev)

    _VHEAD_CACHE.update(dict(
        loaded=True, ok=True, path=path, vis=vis, preprocess=preprocess, head=head,
        txt_emb=txt_emb, concepts=concepts
    ))
    print(f"[kbcs] loaded vision head from {path}")
    return _VHEAD_CACHE

def _infer_vhead(image_path: str, concept: str):
    C = _load_vhead_if_any()
    if not C.get("ok", False): return None
    c2i = {c.lower().strip(): i for i,c in enumerate(C["concepts"])}
    key = concept.lower().strip()
    if key not in c2i: return None
    cid = c2i[key]

    img = Image.open(image_path).convert("RGB")
    px = C["preprocess"](img).unsqueeze(0).to(next(C["head"].parameters()).device)  # [1,3,224,224]
    with torch.no_grad():
        feat = C["vis"].encode_image(px)  # [1,dim]
        feat = feat / (feat.norm(dim=-1, keepdim=True)+1e-8)
        t = C["txt_emb"][cid] / (C["txt_emb"][cid].norm()+1e-8)
        logit = C["head"](feat, t.unsqueeze(0), torch.tensor([cid], device=feat.device)).squeeze(0)
        p_raw = torch.sigmoid(logit).item()
    return {"p_raw": float(p_raw), "roi": None, "decision": "support" if p_raw>=0.5 else "refute", "margin": abs(p_raw-0.5)}

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"

# 单例缓存
_CACHED = {
    "head": None,
    "head_args": None,        # (backend, ckpt_path, stride, device)
    "proposer": None,
    "proposer_args": None,    # (stride, topk, nms, anchor, min_box, score_thr)
    "ckpt_hash": None,
}

def _sha1_file(p: pathlib.Path) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None

def _ensure_head_and_proposer(rt: Dict[str, Any]):
    tool = (rt.get("tool") or {})
    backend = str(tool.get("detector_backend", "heatmap_head")).lower()

    # ---- 运行参数 ----
    params = tool.get("params", {}) or {}
    stride = int(params.get("stride", 16))
    nms = float(params.get("nms", 0.3))
    ckpt_path = tool.get("ckpt_path") or ""
    anchor = int(params.get("anchor", 64))
    # 若未提供 min_box_size，则回落到全局 thr 的配置或 16
    min_box = int(params.get("min_box_size", (rt.get("thr", {}).get("global", {}) or {}).get("min_box_size", 16)))
    topk = int(params.get("topk", 5))
    score_thr = float(params.get("score_thr", 0.0))

    # 新增：device（仅对 xrv_gradcam 有意义）
    device = str(tool.get("device", "cpu")).lower()

    # ---- 选择 head：有 ckpt 的 heatmap_head，否则回退 xrv_gradcam ----
    if backend == "heatmap_head" and ckpt_path:
        args = ("heatmap_head", ckpt_path, stride, "n/a")
        if _CACHED["head"] is None or _CACHED["head_args"] != args:
            _CACHED["head"] = HeatmapHead(ckpt_path=ckpt_path, stride=stride)
            _CACHED["head_args"] = args
            _CACHED["ckpt_hash"] = _sha1_file(pathlib.Path(ckpt_path)) if ckpt_path else None
        stride_from_head = stride
    else:
        # 后备：XRV 预训练 + Grad-CAM（零训练）
        args = ("xrv_gradcam", "pretrained", 1, device)
        if _CACHED["head"] is None or _CACHED["head_args"] != args:
            # XRVGradCAMHead 可能不接受 device 形参；做兼容处理
            try:
                _CACHED["head"] = XRVGradCAMHead(device=device)
            except TypeError:
                _CACHED["head"] = XRVGradCAMHead()
            _CACHED["head_args"] = args
            _CACHED["ckpt_hash"] = "xrv-pretrained"
        stride_from_head = 1  # Grad-CAM 与原图同尺度

    # ---- ROIProposer（与 head 的 stride 对齐）----
    pargs = (stride_from_head, topk, nms, anchor, min_box, score_thr)
    if _CACHED["proposer"] is None or _CACHED["proposer_args"] != pargs:
        _CACHED["proposer"] = ROIProposer(
            stride=stride_from_head, topk=topk, nms_iou=nms,
            anchor=anchor, min_box=min_box, score_thr=score_thr
        )
        _CACHED["proposer_args"] = pargs

def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))

def _get_short_edge(rt: Dict[str, Any]) -> int:
    # 从 runtime.tool.image.short_edge 读取，默认 384；<=0 则不缩放
    img_cfg = (rt.get("tool") or {}).get("image", {}) or {}
    try:
        se = int(img_cfg.get("short_edge", 384))
    except Exception:
        se = 384
    return se

def _resize_keep_short_edge(im: Image.Image, short_edge: int) -> tuple[Image.Image, float]:
    """
    等比缩放到指定短边；返回 (新图, 缩放比例 s=新/旧)。
    short_edge<=0 或已等于短边时返回原图和 1.0。
    """
    W, H = im.size
    if short_edge is None or short_edge <= 0:
        return im, 1.0
    cur_short = min(W, H)
    if cur_short == short_edge:
        return im, 1.0
    s = float(short_edge) / float(cur_short)
    newW, newH = int(round(W * s)), int(round(H * s))
    im2 = im.resize((newW, newH), Image.BILINEAR)
    return im2, s

def score(image_path: str, concept: str, runtime: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    统一证据接口：
    - 根据 tool_runtime.json 选择/配置 heatmap 头与 ROI proposer
    - 读取 device 与 short_edge，推理时等比缩放短边（默认 384）
    - 输出 margin（log-odds）、校准概率、ROI 以及判定（在 margin 空间与 kbcs_thr.json 比较）
    """
    out = _infer_vhead(image_path, concept)
    
    if out is not None:
        out["p"] = float(out.get("p_raw", 0.5))
        out.setdefault("thr_pos", 0.0)
        out.setdefault("thr_neg", 0.0)
        out.setdefault("provenance", {"backend": "biomedclip_vhead"})
        return out
    rt = runtime or load_all()
    _ensure_head_and_proposer(rt)

    # 打开原图（RGB），记录原始尺寸
    im = Image.open(image_path).convert("RGB")
    W0, H0 = im.size  # 原始宽高（注意 PIL 为 (W,H)）
    short_edge = _get_short_edge(rt)

    # 等比缩放（短边对齐）；Grad-CAM/Heatmap 都在缩放后图上运行
    im_in, scale = _resize_keep_short_edge(im, short_edge)

    # 1) heatmap
    head = _CACHED["head"]
    heatmap, meta = head.forward(im_in, concept)

    # 2) ROI（用 head 提供的 stride 对齐映射；orig_hw 用原图尺寸，proposer 内部会做尺度还原）
    _cached = _CACHED["proposer"]
    stride_from_head = int(meta.get("stride", _cached.stride if _cached else 16))
    local_proposer = ROIProposer(
        stride=stride_from_head,
        topk=_cached.topk if _cached else 5,
        nms_iou=_cached.nms_iou if _cached else 0.3,
        anchor=_cached.anchor if _cached else 64,
        min_box=_cached.min_box if _cached else 16,
        score_thr=_cached.score_thr if _cached else 0.0,
    )
    # proposer 负责把缩放后 heatmap 的坐标映射回原图尺度（通过 orig_hw ）
    (x1, y1, x2, y2), raw_score = local_proposer.propose_top1(heatmap, orig_hw=(H0, W0))

    # 3) margin（证据强度，log-odds），对 raw_score 做稳健夹紧
    rs = float(raw_score)
    rs = max(1e-4, min(1.0 - 1e-4, rs))   # 避免 logit(0/1) → ±inf
    margin = math.log(rs / (1.0 - rs))

    # 4) 概率校准（外置；margin -> p）
    use_external_calib = bool((rt.get("tool") or {}).get("external_calib", False))
    if use_external_calib:
        p_cal = 1.0 / (1.0 + math.exp(-margin))  # = rs
    else:
        p_cal = calibrate_margin_to_p(margin, str(concept), rt.get("calib") or {})

    # 5) 阈值比较（margin 空间）
    thr_pos, thr_neg = get_thr(str(concept), rt.get("thr") or {})
    if margin >= thr_pos:
        decision = "support"
    elif margin <= thr_neg:
        decision = "refute"
    else:
        decision = "uncertain"

    # 6) 组织返回（xywh）
    roi_xywh = [int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))]

    # 推断 backend 与 device（记录在 provenance）
    tool = (rt.get("tool") or {})
    backend = str(tool.get("detector_backend", "heatmap_head")).lower()
    device = str(tool.get("device", "cpu")).lower()

    prov = {
        "backend": meta.get("backend", backend),
        "device": device,
        "image_short_edge": int(short_edge if short_edge is not None else -1),
        "head_available": bool(meta.get("available", False)),
        "head_version": meta.get("version"),
        "stride": stride_from_head,
        "topk": int(local_proposer.topk),
        "nms_iou": float(local_proposer.nms_iou),
        "anchor": int(local_proposer.anchor),
        "min_box": int(local_proposer.min_box),
        "score_thr": float(local_proposer.score_thr),
        "ckpt_path": tool.get("ckpt_path", ""),
        "ckpt_sha1": _CACHED.get("ckpt_hash"),
        "raw_score": float(raw_score),           # 原始峰值（0~1，未夹紧）
        "heatmap_channel": meta.get("channel", 0),
        "scale_applied": float(scale),           # 输入图像的缩放比例（相对原图）
        "orig_size": [int(W0), int(H0)]
    }
    return {
        "p": float(p_cal),
        "p_raw": float(rs),
        "margin": float(margin),
        "roi": roi_xywh,
        "decision": decision,
        "thr_pos": float(thr_pos),
        "thr_neg": float(thr_neg),
        "provenance": prov
    }
