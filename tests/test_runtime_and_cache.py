import json

import pytest

from iira2.cache import load_feature_records
from iira2.runtime import resolve_device


def test_cpu_device_is_always_available():
    assert str(resolve_device("cpu")) == "cpu"


def test_cache_join_counts_missing_and_rejects_duplicate(tmp_path):
    files = {}
    for name, rows in {
        "qwen": [{"key": "a", "qp": 0.7}],
        "kbcs": [{"key": "a", "kp": 0.8}],
        "roi": [{"key": "a", "qroi_p": 0.6}],
        "gradcam": [{"key": "a", "loc_score": 0.3}],
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        files[name] = path
    records, stats = load_feature_records(files)
    assert len(records) == 1 and stats["joined"] == 1
    files["qwen"].write_text(json.dumps([{"key": "a"}, {"key": "a"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_feature_records(files)
