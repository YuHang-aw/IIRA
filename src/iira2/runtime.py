"""Small device adapter that keeps CPU tests independent of Ascend packages."""

from __future__ import annotations

import os
import torch


def resolve_device(requested: str | None = None) -> torch.device:
    name = (requested or os.environ.get("IIRA2_DEVICE", "auto")).lower()
    if name in {"cpu", "cuda", "npu"}:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if name == "npu" and not hasattr(torch, "npu"):
            raise RuntimeError("NPU was requested but torch_npu is unavailable")
        return torch.device(name)
    if name != "auto":
        raise ValueError(f"unsupported device: {name}")
    if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
