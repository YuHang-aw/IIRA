def make_check(cfg):
    backend = cfg["PIPELINE"]["CHECK_BACKEND"]
    if backend == "kbcs":
        from adapters.kbcs_check import KBCSCheck
        return KBCSCheck(cfg)
    elif backend == "qwen_vl":
        from adapters.qwen_check import QwenVLCheck
        return QwenVLCheck(cfg)
    else:
        raise ValueError(f"Unknown CHECK_BACKEND: {backend}")
