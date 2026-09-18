"""Join precomputed evidence JSON without exposing row identifiers in reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def _read_unique(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for row in rows:
        key = str(row["key"])
        if key in result:
            raise ValueError(f"duplicate key in {path.name}")
        result[key] = row
    return result


def load_feature_records(paths: Mapping[str, str | Path]) -> tuple[list[dict], dict[str, int]]:
    required = ("qwen", "kbcs", "roi", "gradcam")
    tables = {name: _read_unique(Path(paths[name])) for name in required}
    keys = set().union(*(table.keys() for table in tables.values()))
    records: list[dict] = []
    missing = 0
    for key in sorted(keys):
        if not all(key in tables[name] for name in required):
            missing += 1
            continue
        row = {"key": key}
        for name in required:
            row.update({k: v for k, v in tables[name][key].items() if k != "key"})
        records.append(row)
    return records, {"candidates": len(keys), "missing": missing, "joined": len(records)}
