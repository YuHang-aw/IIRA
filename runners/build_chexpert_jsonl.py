# runners/build_chexpert_jsonl.py
# -*- coding: utf-8 -*-
"""Build a CheXpert jsonl (binary concept classification items).

CheXpert official CSV usually has a column named `Path` and label columns like:
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, ...
Labels are typically: 1 (positive), 0 (negative), -1 (uncertain), ...

This script produces the same sample format used by IIRA runners:
{ "id": ..., "image": .../view1_frontal.jpg, "concept": "atelectasis", "label": 1, "soft_label": 1.0, "p_baseline": ... }

Defaults are tuned for a small proof-of-concept (e.g. 600-case subset), but you can point it at full train.csv as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _norm_concept(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CheXpert labels CSV (train.csv / valid.csv / custom subset)")
    ap.add_argument("--root", required=True, help="Root dir that contains the images referenced by `Path`")
    ap.add_argument("--out", default="artifacts/rl/chexpert.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--concepts", default="Atelectasis,Pleural Effusion", help="Comma-separated list")
    ap.add_argument("--neg_ratio", type=float, default=1.0)
    ap.add_argument("--uncertain", choices=["skip", "neg", "pos"], default="skip")
    ap.add_argument("--max_per_concept", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    rows = _read_csv_rows(args.csv)

    concepts = [_norm_concept(c) for c in args.concepts.split(",") if c.strip()]

    # Build lists per concept
    by_c: Dict[str, Tuple[List[dict], List[dict]]] = {c: ([], []) for c in concepts}  # (pos, neg)

    def conv(v: str) -> int | None:
        v = (v or "").strip()
        if v == "":
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if x == 1:
            return 1
        if x == 0:
            return 0
        if x == -1:
            if args.uncertain == "skip":
                return None
            return 0 if args.uncertain == "neg" else 1
        return None

    hit = 0
    for r in rows:
        rel = r.get("Path") or r.get("path") or r.get("image")
        if not rel:
            continue
        img = root / rel
        if not img.exists():
            continue
        ex_id = r.get("Study") or r.get("Patient") or rel
        for c in concepts:
            # csv col names vary; try several
            col_candidates = [
                c.replace("_", " "),
                c.replace("_", " ").title(),
                c.replace("_", " ").upper(),
                c.replace("_", " ").capitalize(),
                c,
            ]
            val = None
            for col in col_candidates:
                if col in r:
                    val = r[col]
                    break
            y = conv(val) if val is not None else None
            if y is None:
                continue
            item = {
                "id": str(ex_id),
                "image": str(img),
                "concept": c,
                "label": int(y),
                "soft_label": float(y),
            }
            if y == 1:
                by_c[c][0].append(item)
            else:
                by_c[c][1].append(item)
            hit += 1

    out_items: List[dict] = []
    prevalence: Dict[str, float] = {}
    for c, (pos, neg) in by_c.items():
        total = len(pos) + len(neg)
        prevalence[c] = float(len(pos) / max(1, total))
        if len(pos) == 0:
            picked = random.sample(neg, k=min(len(neg), max(50, int(args.max_per_concept) or 0) or 50))
        else:
            k_neg = min(len(neg), int(round(len(pos) * args.neg_ratio)))
            picked = pos + (random.sample(neg, k=k_neg) if k_neg > 0 else [])
        if args.max_per_concept and args.max_per_concept > 0:
            picked = random.sample(picked, k=min(len(picked), args.max_per_concept))
        for it in picked:
            it["p_baseline"] = prevalence[c]
            out_items.append(it)

    random.shuffle(out_items)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    prev_path = str(Path(args.out).with_suffix("")) + "_prevalence.json"
    Path(prev_path).write_text(json.dumps(prevalence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[chexpert] concepts={concepts}")
    for c in concepts:
        pos, neg = by_c[c]
        print(f"  - {c}: pos={len(pos)} neg={len(neg)} prev={prevalence[c]:.4f}")
    print(f"[ok] wrote: {args.out} (n={len(out_items)})")
    print(f"[ok] wrote: {prev_path}")


if __name__ == "__main__":
    main()
