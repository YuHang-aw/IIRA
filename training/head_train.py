# training/head_train.py  —— BiomedCLIP(OpenCLIP) 冻结 + 轻量视觉头
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import os, json, math, random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import open_clip  # ← 新：用 OpenCLIP 加载本地 BiomedCLIP
from pathlib import Path

# ====== 数据工具（与你 vinDr 一致） ======
def _load_concepts_from_csv(csv_path: str) -> List[str]:
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f); header = next(reader)
    return [h for h in header[1:] if h.lower() != "no finding"]

def _build_examples(vindr_root: str, split: str, concepts: List[str], limit_per_concept: int = 200) -> List[Dict[str, Any]]:
    import csv
    root = Path(vindr_root)
    lab_csv = root / "Annotations" / f"image_labels_{split}.csv"
    cache = root / "CachePNG"
    rows = list(csv.DictReader(open(lab_csv, "r", encoding="utf-8")))
    data = {}
    for r in rows:
        iid = r["image_id"]; d = data.setdefault(iid, {})
        for c in concepts:
            try: d[c] = int(float(r[c]))
            except: d[c] = 0
    ex = []
    for c in concepts:
        pos = [iid for iid, dd in data.items() if dd.get(c,0)==1]
        neg = [iid for iid, dd in data.items() if dd.get(c,0)==0]
        random.shuffle(pos); random.shuffle(neg)
        pos = pos[:limit_per_concept]; neg = neg[:limit_per_concept]
        for iid, y in [(i,1) for i in pos] + [(i,0) for i in neg]:
            png = cache / f"{iid}.png"
            if png.exists():
                ex.append({"image": str(png), "concept": c, "label": y})
    random.shuffle(ex)
    return ex

class VinDRPairDataset(Dataset):
    def __init__(self, examples: List[Dict[str,Any]], concept2id: Dict[str,int], preprocess):
        self.items = examples
        self.c2i = concept2id
        self.preprocess = preprocess  # open_clip 自带 224 预处理
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["image"]).convert("RGB")
        px = self.preprocess(img)  # [3,224,224]
        return {
            "pixel_values": px,
            "concept_id": self.c2i[rec["concept"]],
            "label": torch.tensor(float(rec["label"]), dtype=torch.float32),
        }

# ===== 轻量视觉头 =====
class ConceptHead(nn.Module):
    def __init__(self, dim: int, n_concepts: int, use_linear_adapter: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias  = nn.Parameter(torch.zeros(n_concepts))
        self.adapter = nn.Linear(dim, dim, bias=False) if use_linear_adapter else nn.Identity()
        if use_linear_adapter:
            nn.init.eye_(self.adapter.weight)
    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, concept_ids: torch.Tensor):
        img2 = self.adapter(img_emb)
        img2 = img2 / (img2.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (img2 * txt_emb).sum(-1)
        logits = self.scale * sim + self.bias[concept_ids]
        return logits

@dataclass
class TrainConfig:
    out_dir: str
    vindr_root: str
    split: str = "train"          # train / test
    batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-3
    limit_per_concept: int = 200
    use_linear_adapter: bool = False
    seed: int = 123
    biomedclip_dir: str = "/home/neutron/sdc/MODEL/BiomedCLIP-PubMedBERT"  # ← 本地权重目录

def _load_biomedclip_image_encoder(local_dir: str, device: torch.device):
    """
    从本地目录加载 BiomedCLIP(OpenCLIP)。目录需包含：
      - open_clip_config.json
      - open_clip_pytorch_model.bin
    返回：model(含 .encode_image)、preprocess(224 预处理)、embed_dim
    """
    local_dir = str(local_dir)
    # 优先：一行式从目录加载
    try:
        model, preprocess, _ = open_clip.create_model_from_pretrained(local_dir, device=device)
    except Exception:
        # 兜底：用 config 推断，再手动指定权重文件
        import json as _json, os as _os
        cfg = _json.load(open(_os.path.join(local_dir, "open_clip_config.json"), "r"))
        model_name = cfg.get("model_name", "ViT-B-16")
        weight = _os.path.join(local_dir, "open_clip_pytorch_model.bin")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=weight, device=device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    embed_dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else model.text_projection.shape[-1]
    return model, preprocess, int(embed_dim)

def train_head(cfg: TrainConfig, train_index=None, dev_index=None):
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # 1) 概念列表
    lab_csv = Path(cfg.vindr_root) / "Annotations" / f"image_labels_{cfg.split}.csv"
    concepts = _load_concepts_from_csv(str(lab_csv))
    concepts = [c for c in concepts if c.lower()!="no finding"]
    concept2id = {c:i for i,c in enumerate(concepts)}

    # 2) 数据
    ex = _build_examples(cfg.vindr_root, cfg.split, concepts, cfg.limit_per_concept)
    if len(ex) == 0:
        raise RuntimeError("Dataset is empty. 确认 VinDr 的 CachePNG/ 已生成（先跑一次 eval），或路径是否正确。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model, preprocess, dim = _load_biomedclip_image_encoder(cfg.biomedclip_dir, device)
    ds = VinDRPairDataset(ex, concept2id, preprocess)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3) 轻量头 + 概念向量（可学习）
    txt_emb = nn.Parameter(torch.randn(len(concepts), dim))
    nn.init.normal_(txt_emb, std=0.02)
    head = ConceptHead(dim=dim, n_concepts=len(concepts), use_linear_adapter=cfg.use_linear_adapter)
    parms = list(head.parameters()) + [txt_emb]
    opt = torch.optim.AdamW(parms, lr=cfg.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    head.to(device); txt_emb.data = txt_emb.data.to(device)

    def encode_image(px: torch.Tensor):
        # px: [B,3,224,224]（preprocess 产物）
        with torch.no_grad():
            feat = img_model.encode_image(px)  # [B, dim]
            feat = feat / (feat.norm(dim=-1, keepdim=True)+1e-8)
            return feat

    bce = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(cfg.epochs):
        for batch in dl:
            step += 1
            px = batch["pixel_values"].to(device, non_blocking=True)   # [B,3,224,224]
            cid = batch["concept_id"].to(device, non_blocking=True).long()
            y = batch["label"].to(device, non_blocking=True)

            img_emb = encode_image(px)
            t = txt_emb[cid] / (txt_emb[cid].norm(dim=-1, keepdim=True)+1e-8)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = head(img_emb, t, cid)
                loss = bce(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            if step % 50 == 0:
                print(f"[head] ep{ep} step{step} loss={loss.item():.4f}")

    torch.save({
        "concepts": concepts,
        "dim": int(dim),
        "state_dict": head.state_dict(),
        "txt_emb": txt_emb.data.float().cpu(),
        "model_dir": cfg.biomedclip_dir,   # ← 直接记住本地目录
        "image_size": 224,
        "use_linear_adapter": cfg.use_linear_adapter,
    }, os.path.join(cfg.out_dir, "vhead.pt"))
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"concepts": concepts, "model_dir": cfg.biomedclip_dir, "image_size": 224}, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved vision head to {cfg.out_dir}")
