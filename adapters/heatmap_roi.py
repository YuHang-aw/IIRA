# adapters/heatmap_roi.py
# -*- coding: utf-8 -*-
"""
Heatmap→ROI 组件（推理侧，固定接口）
-----------------------------------
职责：
1) HeatmapHead：轻量判别头推理，输出 per-concept heatmap（Hh×Wh，0~1）
   - 期望与 training/head_train.py 的结构一致（最后一层输出 C 个概念通道）
   - ckpt 约定：.pt 内含 state_dict；同目录 concepts.json 映射 {concept -> channel_id}
   - 若无 ckpt，则降级为“启发式中央亮度”热图（可跑、可评测）

2) ROIProposer：从 heatmap 生成候选 ROI
   - topk 峰值 → 方框锚框 → NMS → 输出 top-1 或列表
   - 不依赖 OpenCV，仅用 numpy

固定接口：
  head = HeatmapHead(ckpt_path, stride=16)
  heatmap, meta = head.forward(pil_image, concept)

  proposer = ROIProposer(stride=16, topk=5, nms_iou=0.3, anchor=64, min_box=16, score_thr=0.0)
  rois = proposer.propose_many(heatmap, orig_hw=(H,W))  # -> [{"bbox_xyxy":[x1,y1,x2,y2],"score":float}, ...]
  roi_top1 = proposer.propose_top1(heatmap, orig_hw)    # -> (bbox_xyxy, score)

本模块不做阈值比较与概率校准；仅负责 heatmap 和 ROI。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import json
import pathlib
import math

import numpy as np
from PIL import Image
import torchxrayvision as xrv
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# -----------------------------
# Utils
# -----------------------------

def _to_numpy_gray(im: Image.Image) -> np.ndarray:
    g = im.convert("L")
    arr = np.asarray(g, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------
# Heatmap Head
# -----------------------------

class _ToyHead(nn.Module):
    """与 training/head_train.py 对齐的一个极简头：Conv→ReLU→Conv(C)。"""
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) → (B,C,H,W) logits
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return h


class HeatmapHead:
    """
    轻量判别头推理器。
    - ckpt_path：期望指向 .pt（state_dict）或目录（包含 model.pt 与 concepts.json）
    - stride：heatmap 到原图的步幅（与训练时一致；用于 ROI 映射）
    """
    VERSION = "2025-10-22"

    def __init__(self, ckpt_path: Optional[str] = None, stride: int = 16):
        self.stride = int(stride)
        self.available = False
        self._concept2ch: Dict[str, int] = {}
        self._model: Optional[_ToyHead] = None
        self._device = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
        self._init_from_ckpt(ckpt_path)

    def _init_from_ckpt(self, ckpt_path: Optional[str]):
        if not (_HAS_TORCH and ckpt_path):
            self.available = False
            return
        p = pathlib.Path(ckpt_path)
        concepts_json = None
        if p.is_dir():
            model_pt = p / "model.pt"
            cjson = p / "concepts.json"
        else:
            model_pt = p
            cjson = p.with_suffix(".concepts.json")
        try:
            # 概念映射
            if cjson.exists():
                obj = json.loads(cjson.read_text(encoding="utf-8"))
                # 统一小写空格化
                self._concept2ch = {str(k).strip().lower(): int(v) for k, v in obj.items()}
                out_ch = max(self._concept2ch.values()) + 1
            else:
                # 若没有概念文件，默认 1 通道
                out_ch = 1
            # 架构
            self._model = _ToyHead(out_channels=out_ch)
            sd = torch.load(str(model_pt), map_location="cpu")
            # 兼容 state_dict 包装
            state = sd.get("state_dict", sd)
            self._model.load_state_dict(state, strict=False)
            self._model.to(self._device).eval()
            self.available = True
        except Exception:
            self.available = False
            self._model = None
            self._concept2ch = {}

    def _prep(self, im: Image.Image) -> torch.Tensor:
        # 灰度→归一化→加维度
        g = _to_numpy_gray(im)
        t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t.to(self._device)

    def forward(self, im: Image.Image, concept: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns
        -------
        heatmap : np.ndarray (Hh, Wh), 值域 0~1
        meta    : dict，包含 {"backend","stride","channel","available"}
        """
        cname = str(concept or "").strip().lower()
        if self.available and self._model is not None:
            with torch.inference_mode():
                x = self._prep(im)
                logits = self._model(x)              # (1,C,H,W)
                if logits.ndim == 4 and logits.shape[2] >= 2 and logits.shape[3] >= 2:
                    if self._concept2ch and cname in self._concept2ch:
                        ch = int(self._concept2ch[cname])
                    else:
                        ch = 0
                    m = logits[0, ch]               # (H,W)
                    hm = torch.sigmoid(m).detach().cpu().numpy().astype(np.float32)
                    # 归一化防守
                    hm_min, hm_max = float(hm.min()), float(hm.max())
                    if hm_max - hm_min > 1e-6:
                        hm = (hm - hm_min) / (hm_max - hm_min)
                    else:
                        hm.fill(1.0 / 255.0)
                    meta = {
                        "backend": "heatmap_head",
                        "available": True,
                        "stride": self.stride,      # 训练时的下采样步幅
                        "channel": ch,
                        "version": self.VERSION
                    }
                    return hm, meta

        # ---- 无 ckpt 的“稳健降级”路径（仅用图像本身，不需要任何参考） ----
        g = im.convert("L")
        arr = np.asarray(g, dtype=np.float32)

        # 梯度能量（边缘/结构）+ 局部对比度 + 中心先验，全部只靠这张图
        gy, gx = np.gradient(arr)
        grad = np.hypot(gx, gy)  # 0..~255
        # 局部对比度（1/8 尺度差分）
        k = max(8, min(arr.shape[0], arr.shape[1]) // 64)
        if k % 2 == 0: k += 1
        from scipy.ndimage import gaussian_filter  # 如无 scipy，可改用简单均值滤波
        blur = gaussian_filter(arr, sigma=max(1.0, k / 3.0))
        contrast = np.abs(arr - blur)

        # 中心先验（避免全零）
        H, W = arr.shape
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        dist2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.12 * (H * H + W * W))
        center = np.exp(-dist2)

        hm = 0.45 * (grad / (grad.max() + 1e-6)) + \
            0.35 * (contrast / (contrast.max() + 1e-6)) + \
            0.20 * (center / (center.max() + 1e-6))

        hm_min, hm_max = float(hm.min()), float(hm.max())
        if hm_max - hm_min > 1e-6:
            hm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm.fill(1.0 / 255.0)

        meta = {
            "backend": "heuristic_edges_contrast_center",
            "available": False,
            "stride": 1,              # 关键：降级热图与原图同分辨率
            "channel": 0,
            "version": self.VERSION
        }
        return hm.astype(np.float32), meta

# -----------------------------
# ROI Proposer
# -----------------------------

@dataclass
class ROI:
    bbox_xyxy: Tuple[int, int, int, int]
    score: float

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (N,4), b:(M,4)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    inter_x1 = np.maximum(ax1[:, None], bx1[None])
    inter_y1 = np.maximum(ay1[:, None], by1[None])
    inter_x2 = np.minimum(ax2[:, None], bx2[None])
    inter_y2 = np.minimum(ay2[:, None], by2[None])
    iw = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    ih = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a[:, None] + area_b[None] - inter + 1e-6
    return inter / union

class ROIProposer:
    """
    从 heatmap 生成 ROI 的纯 numpy 实现：
      - 取 topk 峰值（阈值过滤）
      - 用固定 anchor（像素）生成方框（可调 anchor/min_box）
      - NMS 合并，输出从大到小的候选
    """
    def __init__(self, stride: int = 16, topk: int = 5, nms_iou: float = 0.3,
                 anchor: int = 64, min_box: int = 16, score_thr: float = 0.0):
        self.stride = int(stride)
        self.topk = int(topk)
        self.nms_iou = float(nms_iou)
        self.anchor = int(anchor)
        self.min_box = int(min_box)
        self.score_thr = float(score_thr)

    def _boxes_from_peaks(self, peaks_xy: np.ndarray, scores: np.ndarray, orig_hw: Tuple[int,int]) -> np.ndarray:
        """把 heatmap 坐标的峰映射到原图 xyxy 框。"""
        H, W = orig_hw
        stride = self.stride
        a = max(self.min_box, self.anchor)
        half = a / 2.0
        boxes = []
        for (y, x), s in zip(peaks_xy, scores):
            cx = (float(x) + 0.5) * stride
            cy = (float(y) + 0.5) * stride
            x1 = max(0, int(round(cx - half))); y1 = max(0, int(round(cy - half)))
            x2 = min(W - 1, int(round(cx + half))); y2 = min(H - 1, int(round(cy + half)))
            if (x2 - x1 + 1) >= self.min_box and (y2 - y1 + 1) >= self.min_box:
                boxes.append([x1, y1, x2, y2])
        return np.asarray(boxes, dtype=np.int32) if boxes else np.zeros((0,4), dtype=np.int32)

    def propose_many(self, heatmap: np.ndarray, orig_hw: Tuple[int,int]) -> List[Dict[str, Any]]:
        hm = np.asarray(heatmap, dtype=np.float32)
        Hh, Wh = hm.shape
        flat = hm.reshape(-1)
        # 过滤低分
        if self.score_thr > 0:
            cand_idx = np.where(flat >= self.score_thr)[0]
            if cand_idx.size == 0:
                return []
        # topk
        k = min(self.topk, flat.size)
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = idxs[np.argsort(flat[idxs])[::-1]]
        ys, xs = np.divmod(idxs, Wh)
        scores = flat[idxs]
        boxes = self._boxes_from_peaks(np.stack([ys, xs], axis=1), scores, orig_hw)
        if boxes.shape[0] == 0:
            return []
        # NMS
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        keep = []
        while boxes.shape[0] > 0:
            keep.append(0)
            if boxes.shape[0] == 1:
                break
            ious = _iou_xyxy(boxes[:1].astype(np.float32), boxes[1:].astype(np.float32)).reshape(-1)
            mask = ious <= self.nms_iou
            boxes = boxes[1:][mask]
            scores = scores[1:][mask]
        # 还原 keep 索引的具体候选
        result = []
        # 注意：keep 是相对 NMS 过程的“第一个”，但我们已经每轮把第一个加入
        # 因此此时 boxes/scores 已是 NMS 后按分数降序的集合
        # 直接组装
        for i in range(len(keep)):
            # NMS 后 boxes/scores 在循环内被裁剪，这里重新计算一次简化逻辑
            pass
        # 简化：我们上面没有保留所有候选，只留下了最后一轮 boxes/scores
        # 更稳妥：重跑一次 NMS，记录所有保留框
        boxes_full = self._boxes_from_peaks(np.stack([ys, xs], axis=1), scores, orig_hw)
        if boxes_full.shape[0] == 0:
            return []
        order = np.argsort(scores)[::-1]
        boxes_full = boxes_full[order]
        scores_full = scores[order]
        keep_idx = []
        b = boxes_full.copy()
        s = scores_full.copy()
        while b.shape[0] > 0:
            keep_idx.append(len(keep_idx))
            if b.shape[0] == 1:
                break
            ious = _iou_xyxy(b[:1].astype(np.float32), b[1:].astype(np.float32)).reshape(-1)
            mask = ious <= self.nms_iou
            b = b[1:][mask]
            s = s[1:][mask]
        # 收集
        picked = []
        used = np.zeros(len(order), dtype=bool)
        cur_boxes = boxes_full.copy(); cur_scores = scores_full.copy()
        while cur_boxes.shape[0] > 0:
            picked.append((cur_boxes[0], float(cur_scores[0])))
            if cur_boxes.shape[0] == 1:
                break
            ious = _iou_xyxy(cur_boxes[:1].astype(np.float32), cur_boxes[1:].astype(np.float32)).reshape(-1)
            mask = ious <= self.nms_iou
            cur_boxes = cur_boxes[1:][mask]
            cur_scores = cur_scores[1:][mask]
        return [{"bbox_xyxy": tuple(map(int, b)), "score": float(sc)} for b, sc in picked]

    def propose_top1(self, heatmap: np.ndarray, orig_hw: Tuple[int,int]) -> Tuple[Tuple[int,int,int,int], float]:
        many = self.propose_many(heatmap, orig_hw)
        if not many:
            H, W = orig_hw
            return (0, 0, W-1, H-1), 0.0
        b = many[0]["bbox_xyxy"]; s = many[0]["score"]
        return tuple(map(int, b)), float(s)

import inspect
import numpy as np
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def _last_conv_layer(module: nn.Module) -> nn.Module | None:
    last = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def _make_cam(model: nn.Module, target_layers, device: str):
    # 兼容不同版本的 pytorch-grad-cam：有的用 use_cuda，有的用 device，有的都没有
    sig = inspect.signature(GradCAM.__init__)
    params = sig.parameters
    if "device" in params:
        return GradCAM(model=model, target_layers=target_layers, device=device)
    elif "use_cuda" in params:
        return GradCAM(model=model, target_layers=target_layers,
                       use_cuda=device.startswith("cuda"))
    else:
        return GradCAM(model=model, target_layers=target_layers)

class XRVGradCAMHead:
    """
    预训练胸片 DenseNet + Grad-CAM 的“零训练后备头”
    forward(PIL.Image, concept) -> (heatmap[np.float32 HxW], meta)
    """
    def __init__(self, arch: str = "densenet121-res224-all", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = xrv.models.DenseNet(weights=arch).to(self.device).eval()

        # 自动抓“最后一个卷积层”作为 CAM 目标，更健壮
        last_conv = _last_conv_layer(self.model)
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found in model for Grad-CAM.")
        self.target_layers = [last_conv]
        self.cam = _make_cam(self.model, self.target_layers, self.device)

        # 概念映射（可按需扩展）
        self.labels = [s.lower().replace("_", " ") for s in self.model.pathologies]
        self.alias = {
            "pleural effusion": "effusion",
            "pneumothorax": "pneumothorax",
            "cardiomegaly": "cardiomegaly",
            "consolidation": "consolidation",
            "atelectasis": "atelectasis",
            "pneumonia": "pneumonia",
        }

        # XRV 推荐的几何预处理（对 tensor: [1,H,W], 值域[-1024,1024]）
        self.geo = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

    def _concept_to_idx(self, concept: str) -> int | None:
        q = self.alias.get(concept.lower().strip(), concept.lower().strip())
        try:
            return self.labels.index(q)
        except ValueError:
            return None

    def _to_xrv_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """
        PIL -> XRV 期望的 numpy 流程 -> torch.Tensor
        输出: torch.float32 [1,1,224,224]，值域≈[-1024,1024]
        """
        import numpy as np
        import torch
        import torchxrayvision as xrv

        # 1) PIL -> numpy 灰度
        arr = np.array(pil_image)
        if arr.ndim == 3:                      # RGB -> 灰度
            arr = arr.mean(axis=2)

        # 2) 强度自适配到 XRV 的 normalize 输入
        #    - 若是 float 且在 [-1,1]，先还原到 [0,255]
        #    - uint16 用 65535，其他按 255 处理
        if np.issubdtype(arr.dtype, np.floating):
            a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
            if a_min >= -1.2 and a_max <= 1.2:     # 常见 [-1,1]
                arr = ((arr + 1.0) * 0.5) * 255.0
                maxv = 255.0
            else:
                if a_max <= 1.5:                   # [0,1] 的情况
                    arr = arr * 255.0
                maxv = 255.0
        elif arr.dtype == np.uint16:
            maxv = 65535.0
        else:
            maxv = 255.0

        # 3) XRV 规范化到 [-1024,1024]，并加 channel 轴 -> [1,H,W] (numpy)
        arr = xrv.datasets.normalize(arr, maxv).astype("float32")
        arr = arr[None, ...]  # [1,H,W]

        # 4) XRV 的几何预处理（**这里必须是 numpy**）
        #    CenterCrop + 等比缩放到 224 -> 仍是 numpy [1,224,224]
        arr = self.geo(arr)

        # 5) numpy -> torch，并加 batch 维 -> [1,1,224,224]
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return x


    def forward(self, pil_image: Image.Image, concept: str):
        idx = self._concept_to_idx(concept)
        W, H = pil_image.size

        if idx is None:
            # 不在词表：返回近乎零的热图
            return np.ones((H, W), np.float32) * 1e-6, {
                "backend": "xrv_gradcam", "available": False,
                "version": "xrv-densenet121-res224-all", "stride": 1, "channel": -1
            }

        x = self._to_xrv_tensor(pil_image)
        targets = [ClassifierOutputTarget(idx)]

        # 一些版本返回 torch.Tensor，一些返回 numpy；统一到 numpy[224,224], 0..1
        cam_224 = self.cam(input_tensor=x, targets=targets)[0]
        if isinstance(cam_224, torch.Tensor):
            cam_224 = cam_224.detach().cpu().numpy()
        cam_224 = cam_224.astype(np.float32)
        cam_224 = (cam_224 - cam_224.min()) / max(1e-6, (cam_224.max() - cam_224.min()))

        # 上采样回原图大小（与 ROI 对齐），因此 stride=1
        cam_img = Image.fromarray((cam_224 * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
        cam = np.array(cam_img, dtype=np.float32) / 255.0

        meta = {
            "backend": "xrv_gradcam",
            "available": True,
            "version": "xrv-densenet121-res224-all",
            "stride": 1,
            "channel": idx,
        }
        return cam, meta
