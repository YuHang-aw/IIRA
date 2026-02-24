# training/seal_inner_loop.py
# -*- coding: utf-8 -*-
"""
SEAL-style 自适应内环（XAI-Med-VLM）— 加速版
--------------------------------------------
特性：
- 快路径（仅 calib/thr 改动 → 离线重标定，不重跑影像前向）
- 渐进采样（10% 子集先评，富裕通过再全量）
- 评测缓存（配置指纹→指标）
- 事务式回滚/影子/发布/版本化 + diff

依赖：
- eval.evaluate_dual_env.evaluate(dataset) -> Dict[str, Any]
  建议在 evaluate 内写一份 eval_log.jsonl（含 margin_raw/label/used_check/steps/evidence_margin）

用法示例：
from training.seal_inner_loop import InnerLoop, Gate, Edit
loop = InnerLoop(Gate(delta_brier=-0.01, delta_ece=-0.01, min_lift=0.0))
edits = [
    Edit.json("kbcs_thr.json", new_thr_dict),
    Edit.json("calib_kbcs.json", new_calib_dict),
    Edit.text("prompt_template.md", new_template_md, kind="template")
]
res = loop.try_edits(dataset=dev_data, edits=edits, shadow_minutes=60)
print(res["decision"], res["detail"])
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import json
import pathlib
import shutil
import tempfile
import difflib
import hashlib
from datetime import datetime
import random
import os

# =========================
#           路径
# =========================
ROOT = pathlib.Path(__file__).resolve().parents[1]   # project/
CFG_DIR = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
BACKUP_DIR = ARTIFACTS / "cfg_backups"
RELEASES_DIR = ARTIFACTS / "releases"
SHADOW_DIR = ARTIFACTS / "shadow"
EVAL_CACHE_DIR = ARTIFACTS / "eval_cache"
for d in [CFG_DIR, BACKUP_DIR, RELEASES_DIR, SHADOW_DIR, EVAL_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS_CACHE = EVAL_CACHE_DIR / "metrics_cache.json"

# =========================
#        外部评测
# =========================
try:
    from eval.evaluate_dual_env import evaluate  # 你的评测实现
except Exception as e:  # 便于占位/单测
    def evaluate(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise RuntimeError("evaluate(dataset) 未正确导入，请检查 eval/evaluate_dual_env.py") from e

# =========================
#        小工具函数
# =========================

def _read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

def _write_json_atomic(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=str(path.parent), prefix=".json_") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        tmp = pathlib.Path(f.name)
    tmp.replace(path)

def _unified_diff(before: str, after: str, fname: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{fname}:before",
        tofile=f"{fname}:after"
    ))

def _stamp() -> str:
    # UTC，便于跨环境比较
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _file_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def _fingerprint(files: List[str]) -> str:
    """对若干 configs 文件内容做指纹，用于命中评测缓存。"""
    h = hashlib.sha1()
    for f in sorted(files):
        p = (CFG_DIR / f)
        h.update(f.encode())
        h.update(_file_text(p).encode("utf-8"))
    return h.hexdigest()

def _load_metrics_cache() -> Dict[str, Any]:
    try:
        return json.loads(METRICS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_metrics_cache(obj: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=str(METRICS_CACHE.parent), prefix=".mc_") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
        tmp = pathlib.Path(f.name)
    tmp.replace(METRICS_CACHE)

def _find_latest_eval_log() -> Optional[pathlib.Path]:
    """
    搜索最近的 eval_log.jsonl（你可按项目实际调整搜索范围）
    优先：artifacts/releases/*/files/eval_log.jsonl，其次 artifacts/**/eval_log.jsonl
    """
    cands: List[pathlib.Path] = []
    cands += sorted((RELEASES_DIR).glob("*/files/eval_log.jsonl"))
    cands += sorted(ARTIFACTS.glob("**/eval_log.jsonl"))
    if cands:
        return cands[-1]
    # 兜底：项目根目录下
    root_log = ROOT / "eval_log.jsonl"
    return root_log if root_log.exists() else None

# =========================
#           门槛
# =========================

@dataclass
class Gate:
    """上线门槛（负数表示至少下降这么多；正数表示至少上升这么多）"""
    delta_brier: float = -0.01
    delta_ece: float = -0.01
    min_lift: float = 0.0
    min_consistency_gain: float = 0.0
    max_avg_steps_increase: float = 0.0
    # 可选定位/忠实性相关门槛（默认不启用）
    min_pointing_gain: float = 0.0
    min_occlusion_drop_gain: float = 0.0

# =========================
#        自编辑对象
# =========================

@dataclass
class Edit:
    """
    文件级“自编辑”（JSON/模板/词表等）
    - target: 相对 configs/ 的路径，如 "kbcs_thr.json"、"prompt_template.md"
    - content: 写入的文本（JSON 会在 apply 前做解析校验）
    """
    kind: str          # "json" | "template" | "lexicon" | "text" | ...
    target: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    _backup_path: Optional[pathlib.Path] = None

    # ---- 便捷构造 ----
    @staticmethod
    def json(target: str, obj: Dict[str, Any], **meta) -> "Edit":
        return Edit(kind="json", target=target,
                    content=json.dumps(obj, ensure_ascii=False, indent=2),
                    meta=meta)

    @staticmethod
    def text(target: str, text: str, **meta) -> "Edit":
        # 模板/词表/普通文本均可用本构造；kind 仅用于审计描述
        k = meta.pop("kind", "text")
        return Edit(kind=k, target=target, content=text, meta=meta)

    # ---- 应用/回滚 ----
    def _abs_target(self) -> pathlib.Path:
        p = (CFG_DIR / self.target).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def apply(self) -> None:
        dst = self._abs_target()
        # 解析校验（JSON）
        if self.kind == "json":
            try:
                json.loads(self.content)
            except Exception as e:
                raise ValueError(f"[Edit] JSON 解析失败：{self.target}，错误：{e}") from e
        # 备份
        if dst.exists():
            bak = BACKUP_DIR / f"{self.target}.bak"
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(dst, bak)
            self._backup_path = bak
        # 原子写
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                         dir=str(dst.parent), prefix=".apply_") as f:
            f.write(self.content if self.content is not None else "")
            tmp = pathlib.Path(f.name)
        tmp.replace(dst)

    def revert(self) -> None:
        dst = self._abs_target()
        if self._backup_path and self._backup_path.exists():
            shutil.copy(self._backup_path, dst)

    def descriptor(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "meta": self.meta}

# =========================
#     可选：LoRA 计划（占位）
# =========================

@dataclass
class LoraPlan:
    name: str
    train_jsonl: str
    output_dir: str
    max_steps: int = 200
    lr: float = 5e-5
    batch_size: int = 4
    instruction: str = "format/act compliance"
    extra_args: Dict[str, Any] = field(default_factory=dict)

class LoraTrainer:
    @staticmethod
    def run(plan: LoraPlan) -> pathlib.Path:
        """
        对接你的 LoRA 训练脚本。这里放占位：创建一个 adapter.safetensors。
        """
        out = pathlib.Path(plan.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adapter = out / "adapter.safetensors"
        adapter.write_text(f"PLACEHOLDER LoRA {plan.name}\n", encoding="utf-8")
        return adapter

# =========================
#   可选：Detector 计划（占位）
# =========================

@dataclass
class DetectorPlan:
    name: str
    backend: str = "heatmap_head"                 # 例："heatmap_head" / "external_dino" ...
    train_cfg: str = "configs/detector_train_v1.json"
    output_dir: str = "artifacts/detectors/heatmap_head_v2"
    max_epochs: int = 3
    extra_args: Dict[str, Any] = field(default_factory=dict)

class DetectorTrainer:
    @staticmethod
    def run(plan: DetectorPlan) -> pathlib.Path:
        """
        对接你的检测/热图头训练脚本。这里放占位：创建一个 ckpt.pt。
        """
        out = pathlib.Path(plan.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / "ckpt.pt"
        ckpt.write_text(f"PLACEHOLDER DETECTOR {plan.name}\n", encoding="utf-8")
        return ckpt

# =========================
#     快路径：离线重标定
# =========================

def _try_fast_recalib(dataset: List[Dict[str, Any]], changed_files: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    仅当编辑只涉及 calib/thr（校准/阈值）时，离线重算指标，避免重跑影像前向。
    返回: (可快算?, 指标dict或空)
    依赖：最近一次完整评测留下的 eval_log.jsonl，内含每样本的 margin/raw_p/label 等。
    需要字段（建议在 evaluate 中写入）：
      - margin_raw: float
      - label: int(0/1)
      - used_check: bool
      - steps: int
      - evidence_margin: float （或其他你定义的证据力度量）
    """
    # 仅 calib/thr 改动才走快路径
    only_calib = all(x in {"calib_kbcs.json", "kbcs_thr.json"} for x in changed_files)
    if not only_calib:
        return False, {}

    log_path = _find_latest_eval_log()
    if log_path is None or not log_path.exists():
        return False, {}

    # 读取新配置（必要时可用 thr 参与一致性/决策重算，这里先不强依赖 thr）
    calib_path = CFG_DIR / "calib_kbcs.json"
    try:
        calib = json.loads(_read_text(calib_path) or "{}")
    except Exception:
        calib = {}

    # 聚合指标
    tot = 0
    brier = 0.0
    import math
    # ECE（10-bin）
    bins = [[] for _ in range(10)]
    consistent = 0
    steps_sum = 0
    lift_vals: List[float] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                jo = json.loads(line)
            except Exception:
                continue

            m = float(jo.get("margin_raw", jo.get("margin", 0.0)))
            y = int(jo.get("label", 0))
            used_check = bool(jo.get("used_check", False))
            steps = int(jo.get("steps", 1))
            ev_m = jo.get("evidence_margin", None)

            # 新 calib：p = sigmoid((m/T) + b)
            T  = float(calib.get("default", {}).get("T", 1.0))
            b0 = float(calib.get("default", {}).get("b", 0.0))
            p  = 1.0 / (1.0 + math.exp(-(m / max(T, 1e-6) + b0)))

            # Brier
            brier += (p - y) ** 2
            tot += 1

            # ECE（10 bin）
            bi = min(9, max(0, int(p * 10)))
            bins[bi].append((p, y))

            # Consistency（近似用 used_check 比率；如需更严格可基于新阈值重判）
            if used_check:
                consistent += 1

            # Avg steps
            steps_sum += steps

            # Evidence-lift（这里简单用 evidence_margin 的均值；可换你定义的度量）
            if ev_m is not None:
                try:
                    lift_vals.append(float(ev_m))
                except Exception:
                    pass

    if tot == 0:
        return True, {"brier_tool": 1.0, "ece_tool": 1.0, "consistency": 0.0, "avg_steps": 0.0, "evidence_lift": 0.0}

    brier /= tot

    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += abs(conf - acc) * (len(b) / tot)

    consistency = consistent / max(1, tot)
    avg_steps = steps_sum / max(1, tot)
    ev_lift = (sum(lift_vals) / len(lift_vals)) if lift_vals else 0.0

    return True, {
        "brier_tool": brier,
        "ece_tool": ece,
        "consistency": consistency,
        "avg_steps": avg_steps,
        "evidence_lift": ev_lift
    }

# =========================
#         内环主体
# =========================

class InnerLoop:
    """
    SEAL-style 自适应内环（事务式）：
      1) 应用候选自编辑（文件/LoRA/检测器版本）
      2) 评测（双环境）
      3) 门槛判定：不通过→回滚；通过→影子或发布
      4) 版本化：保存快照、diff、指标与产物路径
    """

    def __init__(self, gate: Gate):
        self.gate = gate

    # ---- 评测 ----
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return evaluate(dataset)

    @staticmethod
    def _metric(m: Dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(m.get(key, default))
        except Exception:
            return default

    def _passed(self, before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        d_brier = self._metric(after, "brier_tool") - self._metric(before, "brier_tool")
        d_ece   = self._metric(after, "ece_tool")   - self._metric(before, "ece_tool")
        d_steps = self._metric(after, "avg_steps")  - self._metric(before, "avg_steps")
        lift    = self._metric(after, "evidence_lift")
        d_cons  = self._metric(after, "consistency") - self._metric(before, "consistency")

        # 可选指标：定位/忠实性
        d_point = self._metric(after, "pointing", None)
        d_point = (d_point - self._metric(before, "pointing", d_point)) if d_point is not None else None

        d_occ = self._metric(after, "occlusion_drop", None)
        d_occ = (d_occ - self._metric(before, "occlusion_drop", d_occ)) if d_occ is not None else None

        ok = (
            d_brier <= self.gate.delta_brier and
            d_ece   <= self.gate.delta_ece   and
            lift    >= self.gate.min_lift    and
            d_cons  >= self.gate.min_consistency_gain and
            d_steps <= self.gate.max_avg_steps_increase
        )

        if d_point is not None:
            ok = ok and (d_point >= self.gate.min_pointing_gain)
        if d_occ is not None:
            ok = ok and (d_occ >= self.gate.min_occlusion_drop_gain)

        details = {
            "delta_brier": d_brier,
            "delta_ece": d_ece,
            "evidence_lift": lift,
            "delta_consistency": d_cons,
            "delta_avg_steps": d_steps
        }
        if d_point is not None:
            details["delta_pointing"] = d_point
        if d_occ is not None:
            details["delta_occlusion_drop"] = d_occ

        return ok, details

    def _shadow_on(self, files: List[str], minutes: int) -> str:
        tag = f"shadow_{_stamp()}"
        sd = SHADOW_DIR / tag
        sd.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = CFG_DIR / f
            dst = sd / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy(src, dst)
        (sd / "TTL_MINUTES").write_text(str(minutes), encoding="utf-8")
        return tag

    def _commit_release(
        self,
        files: List[str],
        diffs: List[str],
        metrics_before: Dict[str, Any],
        metrics_after: Dict[str, Any],
        detail: Dict[str, float],
        shadow_tag: Optional[str],
        lora_artifact: Optional[pathlib.Path],
        detector_ckpt: Optional[pathlib.Path],
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        rid = f"r_{_stamp()}_{hashlib.sha1(('|'.join(files)).encode()).hexdigest()[:6]}"
        rdir = RELEASES_DIR / rid
        (rdir / "files").mkdir(parents=True, exist_ok=True)

        # 保存修改文件快照
        for f in files:
            src = CFG_DIR / f
            dst = rdir / "files" / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy(src, dst)

        # 保存 diff 与指标
        (rdir / "diff.txt").write_text("\n\n".join(diffs), encoding="utf-8")
        payload = {
            "release_id": rid,
            "time": _stamp(),
            "files": files,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": detail,
            "shadow_tag": shadow_tag,
            "lora_artifact": str(lora_artifact) if lora_artifact else None,
            "detector_ckpt": str(detector_ckpt) if detector_ckpt else None,
            "tags": tags or {}
        }
        (rdir / "release.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # 维护全球索引
        idx = RELEASES_DIR / "releases.jsonl"
        with idx.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    # ---- 主流程：应用编辑 → 评测 → 回滚/影子/发布（带加速）----
    def try_edits(
        self,
        dataset: List[Dict[str, Any]],
        edits: List[Edit],
        lora_plan: Optional[LoraPlan] = None,
        detector_plan: Optional[DetectorPlan] = None,
        shadow_minutes: int = 0,
        tags: Optional[Dict[str, str]] = None,
        quick_ratio: float = 0.1,          # 渐进采样：先用 10%
        safe_margin: float = 0.003,        # 小集通过需留的“安全裕度”
        cache_keys: Optional[List[str]] = None  # 参与指纹的文件列表；None 用默认
    ) -> Dict[str, Any]:
        """
        增强版 try_edits：
          - 先查缓存拿 baseline 指标；
          - 应用编辑；
          - 若仅 calib/thr 改动 → 离线重标定 → 直接判 Gate；
          - 否则先用子集评（满足 Gate 且留出 safe_margin 才全量评）；
          - 命中缓存的配置直接复用指标；
          - 不通过则回滚；通过则发布/影子，并写入 releases 索引。
        """
        # tqdm（若不可用则降级）
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **kwargs): return x

        # ---- 参与指纹的关键文件（可按需增减）----
        key_files = cache_keys or [
            "calib_kbcs.json",
            "kbcs_thr.json",
            "tool_runtime.json",
            "prompt_template.md",
            "lexicon.json"
        ]

        cache = _load_metrics_cache()

        # ===== 1) 基线评测（先查缓存）=====
        fp_before = _fingerprint(key_files)
        if fp_before in cache:
            metrics_before = cache[fp_before]
        else:
            metrics_before = self.evaluate(dataset)
            cache[fp_before] = metrics_before
            _save_metrics_cache(cache)

        diffs: List[str] = []
        applied_files: List[str] = []
        lora_artifact: Optional[pathlib.Path] = None
        detector_ckpt: Optional[pathlib.Path] = None

        # ===== 2) 应用文件类编辑（阈值/温度/词表/模板/…）=====
        for e in edits:
            tgt = (CFG_DIR / e.target)
            before_text = _read_text(tgt)
            e.apply()
            applied_files.append(e.target)
            after_text = _read_text(tgt)
            diffs.append(_unified_diff(before_text, after_text, e.target))

        # ===== 3) 可选：LoRA（文本侧）/ Detector（热图头）=====
        if lora_plan is not None:
            lora_artifact = LoraTrainer.run(lora_plan)
            lora_cfg_path = CFG_DIR / "lora_runtime.json"
            _write_json_atomic(lora_cfg_path, {
                "adapter_path": str(lora_artifact),
                "plan": asdict(lora_plan)
            })
            applied_files.append(str(lora_cfg_path.relative_to(CFG_DIR)))
            diffs.append(_unified_diff("", _read_text(lora_cfg_path), "lora_runtime.json"))

        if detector_plan is not None:
            detector_ckpt = DetectorTrainer.run(detector_plan)
            tr_path = CFG_DIR / "tool_runtime.json"
            try:
                old = json.loads(_read_text(tr_path) or "{}")
            except Exception:
                old = {}
            new_conf = {
                **old,
                "detector_backend": detector_plan.backend,
                "ckpt_path": str(detector_ckpt),
                "version": detector_plan.name
            }
            _write_json_atomic(tr_path, new_conf)
            applied_files.append(str(tr_path.relative_to(CFG_DIR)))
            diffs.append(_unified_diff(
                json.dumps(old, indent=2, ensure_ascii=False),
                json.dumps(new_conf, indent=2, ensure_ascii=False),
                "tool_runtime.json"
            ))

        # ===== 4) 快路径：仅 calib/thr 改动 → 离线重算 =====
        fast_ok, fast_metrics = _try_fast_recalib(dataset, applied_files)
        if fast_ok:
            metrics_after = fast_metrics
            passed, detail = self._passed(metrics_before, metrics_after)
            if not passed:
                # 回滚
                for e in reversed(edits):
                    try:
                        e.revert()
                    except Exception:
                        pass
                return {
                    "decision": "reverted",
                    "before": metrics_before, "after": metrics_after, "detail": detail,
                    "diff": diffs, "applied_files": applied_files,
                    "lora_artifact": str(lora_artifact) if lora_artifact else None,
                    "detector_ckpt": str(detector_ckpt) if detector_ckpt else None
                }

            # 通过 → 影子/发布 & 缓存
            shadow_tag = None
            if shadow_minutes and shadow_minutes > 0:
                shadow_tag = self._shadow_on(applied_files, minutes=shadow_minutes)
            release = self._commit_release(
                files=applied_files, diffs=diffs,
                metrics_before=metrics_before, metrics_after=metrics_after,
                detail=detail, shadow_tag=shadow_tag,
                lora_artifact=lora_artifact, detector_ckpt=detector_ckpt, tags=tags
            )
            fp_after = _fingerprint(key_files)
            cache = _load_metrics_cache()
            cache[fp_after] = metrics_after
            _save_metrics_cache(cache)
            return {
                "decision": "shadow" if shadow_tag else "released",
                "release": release,
                "before": metrics_before, "after": metrics_after, "detail": detail,
                "diff": diffs, "applied_files": applied_files,
                "lora_artifact": str(lora_artifact) if lora_artifact else None,
                "detector_ckpt": str(detector_ckpt) if detector_ckpt else None
            }

        # ===== 5) 需要真评测：先小子集 → 再全量 =====
        n = len(dataset)
        k = max(1, int(n * quick_ratio))
        subset = random.sample(dataset, k=k)

        # 5.1 小子集评测（不做缓存，避免子集随机性污染）
        metrics_quick = self.evaluate(subset)

        # 5.2 小子集 Gate + 安全裕度
        ok_quick, quick_detail = self._passed(metrics_before, metrics_quick)
        rich_pass = (
            quick_detail.get("delta_brier", 0.0) <= (self.gate.delta_brier - safe_margin) and
            quick_detail.get("delta_ece", 0.0)   <= (self.gate.delta_ece   - safe_margin)
        )
        if not ok_quick or not rich_pass:
            # 回滚
            for e in reversed(edits):
                try:
                    e.revert()
                except Exception:
                    pass
            return {
                "decision": "reverted",
                "before": metrics_before, "after": metrics_quick, "detail": quick_detail,
                "diff": diffs, "applied_files": applied_files,
                "lora_artifact": str(lora_artifact) if lora_artifact else None,
                "detector_ckpt": str(detector_ckpt) if detector_ckpt else None
            }

        # 5.3 全量评测（查缓存）
        fp_after = _fingerprint(key_files)
        cache = _load_metrics_cache()
        if fp_after in cache:
            metrics_after = cache[fp_after]
        else:
            metrics_after = self.evaluate(dataset)
            cache[fp_after] = metrics_after
            _save_metrics_cache(cache)

        passed, detail = self._passed(metrics_before, metrics_after)

        if not passed:
            for e in reversed(edits):
                try:
                    e.revert()
                except Exception:
                    pass
            return {
                "decision": "reverted",
                "before": metrics_before, "after": metrics_after, "detail": detail,
                "diff": diffs, "applied_files": applied_files,
                "lora_artifact": str(lora_artifact) if lora_artifact else None,
                "detector_ckpt": str(detector_ckpt) if detector_ckpt else None
            }

        shadow_tag = None
        if shadow_minutes and shadow_minutes > 0:
            shadow_tag = self._shadow_on(applied_files, minutes=shadow_minutes)

        release = self._commit_release(
            files=applied_files,
            diffs=diffs,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            detail=detail,
            shadow_tag=shadow_tag,
            lora_artifact=lora_artifact,
            detector_ckpt=detector_ckpt,
            tags=tags
        )

        return {
            "decision": "shadow" if shadow_tag else "released",
            "release": release,
            "before": metrics_before, "after": metrics_after, "detail": detail,
            "diff": diffs, "applied_files": applied_files,
            "lora_artifact": str(lora_artifact) if lora_artifact else None,
            "detector_ckpt": str(detector_ckpt) if detector_ckpt else None
        }
