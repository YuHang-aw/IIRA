# runners/build_vindr_jsonl.py
# -*- coding: utf-8 -*-
"""Build VinDr-CXR jsonl for IIRA.

Why this rewrite?
- The original "full" builder expands to (num_images * num_concepts) which becomes huge (45k * 28 = 1.26M).
- For SFT/RL on a single GPU, a *balanced* dataset is usually better: keep all positives, and sample negatives per concept.

Outputs (default):
- artifacts/rl/vindr_train.jsonl
- artifacts/rl/vindr_val.jsonl
- artifacts/rl/vindr_test.jsonl
- artifacts/rl/vindr_prevalence.json

Each row:
{image, concept, label, soft_label, p_baseline, id}

Notes
- Uses `image_labels_{split}.csv` for per-image multi-labels.
- Uses CachePNG/derived as image roots (auto-detected).
"""

from __future__ import annotations

import argparse, csv, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default label columns for VinDr
DEFAULT_CONCEPTS = [
    "Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Clavicle fracture",
    "Consolidation","Edema","Emphysema","Enlarged PA","ILD","Infiltration","Lung Opacity",
    "Lung cavity","Lung cyst","Mediastinal shift","Nodule/Mass","Pleural effusion",
    "Pleural thickening","Pneumothorax","Pulmonary fibrosis","Rib fracture","Other lesion",
    "COPD","Lung tumor","Pneumonia","Tuberculosis","Other disease","No finding",
]


def _seed(seed: int):
    random.seed(seed)


def _read_labels_csv(p: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = [r for r in reader]
    return cols, rows


def _find_image(image_id: str, roots: List[Path]) -> Optional[str]:
    # The cached PNG name is typically {image_id}.png
    cands = [
        f"{image_id}.png",
        f"{image_id}.jpg",
        f"{image_id}.jpeg",
        f"{image_id}.PNG",
        f"{image_id}.JPG",
    ]
    for root in roots:
        for name in cands:
            p = root / name
            if p.exists():
                return str(p)
    return None


def _parse01(v: str) -> int:
    # VinDr labels are usually 0/1; be defensive
    try:
        x = float(v)
        return int(x >= 0.5)
    except Exception:
        return 0


def _concept_list_from_args(args, header_cols: List[str]) -> List[str]:
    if args.concepts:
        # comma separated
        xs: List[str] = []
        for part in args.concepts.split(","):
            part = part.strip()
            if part:
                xs.append(part)
        return xs
    if args.pair:
        a, b = [x.strip() for x in args.pair.split(",")]
        return [a, b]
    # Use header intersection if possible; else fallback default
    avail = [c for c in header_cols if c != "image_id"]
    if avail:
        return avail
    return list(DEFAULT_CONCEPTS)


def _compute_prevalence(rows: List[Dict[str, str]], concepts: List[str]) -> Dict[str, float]:
    prev: Dict[str, float] = {}
    n = max(1, len(rows))
    for c in concepts:
        pos = 0
        for r in rows:
            if c in r and _parse01(r[c]) == 1:
                pos += 1
        prev[c] = pos / n
    return prev


def _build_full(rows: List[Dict[str, str]], roots: List[Path], concepts: List[str], prev: Dict[str, float], split: str):
    items = []
    miss = 0
    for r in rows:
        image_id = r.get("image_id")
        if not image_id:
            continue
        img_path = _find_image(image_id, roots)
        if img_path is None:
            miss += 1
            continue
        for c in concepts:
            if c not in r:
                continue
            y = _parse01(r[c])
            items.append({
                "id": f"{split}:{image_id}:{c}",
                "image_id": image_id,
                "image": img_path,
                "concept": c,
                "label": int(y),
                "soft_label": float(y),
                "p_baseline": float(prev.get(c, 0.5)),
            })
    return items, miss


def _build_balanced(rows: List[Dict[str, str]], roots: List[Path], concepts: List[str], prev: Dict[str, float], split: str,
                    neg_ratio: float, max_per_concept: Optional[int], seed: int):
    rng = random.Random(seed)
    items = []
    miss = 0

    # Pre-index image paths once
    img_cache: Dict[str, Optional[str]] = {}
    for r in rows:
        image_id = r.get("image_id")
        if not image_id:
            continue
        if image_id not in img_cache:
            img_cache[image_id] = _find_image(image_id, roots)

    for c in concepts:
        pos_ids, neg_ids = [], []
        for r in rows:
            image_id = r.get("image_id")
            if not image_id or c not in r:
                continue
            y = _parse01(r[c])
            (pos_ids if y == 1 else neg_ids).append(image_id)

        if len(pos_ids) == 0 and len(neg_ids) == 0:
            continue

        # Keep all positives; sample negatives
        keep_neg = int(round(len(pos_ids) * float(neg_ratio))) if len(pos_ids) > 0 else min(len(neg_ids), int(neg_ratio))
        keep_neg = min(keep_neg, len(neg_ids))
        rng.shuffle(neg_ids)
        neg_keep = neg_ids[:keep_neg]

        chosen = [(i, 1) for i in pos_ids] + [(i, 0) for i in neg_keep]
        if max_per_concept is not None:
            rng.shuffle(chosen)
            chosen = chosen[: max(1, int(max_per_concept))]

        for image_id, y in chosen:
            img_path = img_cache.get(image_id)
            if img_path is None:
                miss += 1
                continue
            items.append({
                "id": f"{split}:{image_id}:{c}",
                "image_id": image_id,
                "image": img_path,
                "concept": c,
                "label": int(y),
                "soft_label": float(y),
                "p_baseline": float(prev.get(c, 0.5)),
            })

    rng.shuffle(items)
    return items, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr_root", default="/home/neutron/sdc/RAG/data/vinDr-CXR")
    ap.add_argument("--out_dir", default="artifacts/rl")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--mode", choices=["balanced", "full"], default="balanced",
                    help="balanced keeps all positives and samples negatives per concept (recommended)")
    ap.add_argument("--concepts", default=None, help="comma-separated concepts to include (default: all columns in csv)")
    ap.add_argument("--pair", default=None, help="alias for two concepts, e.g. 'Atelectasis,Pleural effusion'")

    # balanced-only knobs
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="#negatives per positive (balanced mode)")
    ap.add_argument("--max_per_concept", type=int, default=None,
                    help="cap examples per concept after balancing (optional)")

    args = ap.parse_args()
    _seed(args.seed)

    vindr_root = Path(args.vindr_root)
    ann_dir = vindr_root / "Annotations"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # image roots (cached png + derived)
    image_roots = []
    for cand in [vindr_root / "CachePNG", vindr_root / "derived", vindr_root / "train", vindr_root / "test"]:
        if cand.exists():
            image_roots.append(cand)
    print("[vindr] image_roots:", [str(x) for x in image_roots])

    train_csv = ann_dir / "image_labels_train.csv"
    test_csv = ann_dir / "image_labels_test.csv"

    cols_train, rows_train = _read_labels_csv(train_csv)
    cols_test, rows_test = _read_labels_csv(test_csv)

    concepts = _concept_list_from_args(args, cols_train)
    # Normalize concept names: keep as in csv header if possible
    # (user may pass lowercase, etc.)
    header_concepts = [c for c in cols_train if c != "image_id"]
    norm_map = {c.lower(): c for c in header_concepts}
    concepts = [norm_map.get(c.strip().lower(), c.strip()) for c in concepts]

    prev = _compute_prevalence(rows_train, concepts)
    (out_dir / "vindr_prevalence.json").write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[vindr] wrote prevalence: {out_dir/'vindr_prevalence.json'}")

    # split train into train/val
    random.shuffle(rows_train)
    n_train = len(rows_train)
    n_val = max(1, int(round(0.1 * n_train)))
    rows_val = rows_train[:n_val]
    rows_tr = rows_train[n_val:]

    def build(rows, split):
        if args.mode == "full":
            return _build_full(rows, image_roots, concepts, prev, split)
        return _build_balanced(rows, image_roots, concepts, prev, split, args.neg_ratio, args.max_per_concept, args.seed)

    tr_items, miss_tr = build(rows_tr, "train")
    va_items, miss_va = build(rows_val, "val")
    te_items, miss_te = build(rows_test, "test")

    print(f"[vindr] built {len(tr_items)} items for train, missing_images={miss_tr}/{len(rows_tr)}")
    print(f"[vindr] built {len(va_items)} items for val, missing_images={miss_va}/{len(rows_val)}")
    print(f"[vindr] built {len(te_items)} items for test, missing_images={miss_te}/{len(rows_test)}")

    def write_jsonl(items: List[dict], path: Path):
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    p_tr = out_dir / "vindr_train.jsonl"
    p_va = out_dir / "vindr_val.jsonl"
    p_te = out_dir / "vindr_test.jsonl"
    write_jsonl(tr_items, p_tr)
    write_jsonl(va_items, p_va)
    write_jsonl(te_items, p_te)

    print("[ok] wrote:", p_tr, p_va, p_te)


if __name__ == "__main__":
    main()
