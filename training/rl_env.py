# training/rl_env.py
# -*- coding: utf-8 -*-
"""
A lightweight RL environment for IIRA.

Key points (IMPORTANT):
- Action order MUST match training/policy.py's action head order.
  In our patched Policy, the action head order is:
      ["<CHECK>", "<CLAIM>", "<ABSTAIN>", "<STOP>"]

If you change Policy's action order, you must update ACTION_LIST here accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set
import os
import time
import json
import pathlib
import math
import atexit

from PIL import Image

import torch

# ✅ MUST match Policy action head order
ACTION_LIST = ["<CHECK>", "<CLAIM>", "<ABSTAIN>", "<STOP>"]


# =========================
# Env config
# =========================
@dataclass
class EnvCfg:
    # interaction
    max_steps: int = 3
    disable_check: bool = False
    use_baseline_when_no_check: bool = True

    # initial prior source
    use_qwen_prior: bool = False  # True: p0 = policy.score_yesno(image, concept); False: p0 = ex["p_baseline"]

    # CHECK evidence source: "none" | "prior" | "kbcs" | "grid" | "self"
    check_source: str = "none"

    # self/grid probe
    self_mode: str = "grid"           # "grid" | "bbox"(placeholder)
    grid_n: int = 3
    grid_allow_repeat: bool = False

    # fusion: "none" | "mix" | "gate"
    fuse_mode: str = "mix"
    eta: float = 0.5
    tau: float = 0.12
    gamma_gate: float = 0.35
    calibrate_tool: bool = False

    # legacy knobs
    check_alpha: float = 0.9
    claim_gamma: float = 1.5

    # debug
    debug_traj: bool = False  # RL_DEBUG_TRAJ=1 also enables


# =========================
# Image helpers
# =========================
def _to_pil(img: Any | None) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, (str, pathlib.Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return img


def _grid_boxes(w: int, h: int, n: int):
    """Return n x n grid bboxes (x1,y1,x2,y2,cell_id)."""
    n = max(1, int(n))
    xs = [int(round(i * w / n)) for i in range(n + 1)]
    ys = [int(round(i * h / n)) for i in range(n + 1)]
    out = []
    cid = 0
    for r in range(n):
        for c in range(n):
            x1, x2 = xs[c], xs[c + 1]
            y1, y2 = ys[r], ys[r + 1]
            out.append((x1, y1, x2, y2, cid))
            cid += 1
    return out


# =========================
# Env
# =========================
class SimpleEnv:
    """
    Unified interaction environment.

    - p0: use_qwen_prior ? policy.score_yesno : ex["p_baseline"]
    - CHECK: evidence from check_source, fused by fuse_mode
    - CLAIM: sharpen probability (logit scaling) by claim_gamma
    - ABSTAIN: output 0.5
    - STOP: output current p
    - final: if never CHECK and use_baseline_when_no_check -> return p0
    """

    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg

        # debug
        self._debug_on = bool(cfg.debug_traj) or (os.getenv("RL_DEBUG_TRAJ", "0") == "1")
        self._seq = 0
        self._traj_log = None
        if self._debug_on:
            os.makedirs("artifacts/debug", exist_ok=True)
            self._traj_log = open("artifacts/debug/rl_traj.jsonl", "a", encoding="utf-8")
            atexit.register(self._close_debug)

        # runtime resources for kbcs
        self._rt = None
        if self.cfg.check_source == "kbcs":
            try:
                from adapters.runtime_cfg import load_all as _load_rt
                self._rt = _load_rt()
            except Exception as e:
                print(f"[warn] load runtime cfg failed: {e}")
                self._rt = None

        self._calib_cfg = self._load_calib_json()

    def _close_debug(self):
        try:
            if self._traj_log is not None:
                self._traj_log.close()
        except Exception:
            pass

    # ---------- numeric utils ----------
    @staticmethod
    def _clip01(x: float, eps: float = 1e-6) -> float:
        return max(eps, min(1.0 - eps, float(x)))

    @staticmethod
    def _logit(x: float) -> float:
        x = SimpleEnv._clip01(x)
        return math.log(x / (1.0 - x))

    @staticmethod
    def _sharpen(p: float, gamma: float) -> float:
        p = SimpleEnv._clip01(p)
        z = math.log(p / (1.0 - p))
        return float(1.0 / (1.0 + math.exp(-gamma * z)))

    @staticmethod
    def _blend(p: float, q: float, alpha: float) -> float:
        return float((1.0 - alpha) * p + alpha * q)

    # ---------- prompt helpers ----------
    @staticmethod
    def _kbcs_hint(ex: Dict[str, Any]) -> str:
        return str(
            ex.get(
                "kbcs_hint",
                "Use radiological criteria and be conservative about false positives.",
            )
        )

    @staticmethod
    def _score_yesno_with_context(policy, image, concept: str, ctx_text: str) -> float:
        concept_with_ctx = f"{concept}. Consider the following context/evidence when answering: {ctx_text}"
        with torch.no_grad():
            return float(policy.score_yesno(image, concept_with_ctx))

    # ---------- debug log ----------
    def _log_traj(self, rec: Dict[str, Any]) -> None:
        if not self._debug_on or self._traj_log is None:
            return
        try:
            self._traj_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._traj_log.flush()
        except Exception:
            pass

    # ---------- sample id ----------
    def _ex_id(self, ex: Dict[str, Any]) -> str:
        for k in ("id", "image_id", "study_id", "uid", "path"):
            v = ex.get(k)
            if v:
                return str(v)
        self._seq += 1
        return f"ex{self._seq}"

    # ---------- action mask ----------
    def _valid_mask(self) -> List[bool]:
        # order matches ACTION_LIST
        mask = [True] * len(ACTION_LIST)
        if getattr(self.cfg, "disable_check", False):
            try:
                mask[ACTION_LIST.index("<CHECK>")] = False
            except ValueError:
                pass
        return mask

    # ---------- calib / gate ----------
    def _load_calib_json(self) -> Optional[dict]:
        p = pathlib.Path("configs/calib_kbcs.json")
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except Exception:
                return None
        return None

    def _calibrate_tool_p(self, concept: str, p_raw: float) -> float:
        if not self.cfg.calibrate_tool:
            return float(self._clip01(p_raw))
        T, b = 1.0, 0.0
        if self._calib_cfg:
            c = self._calib_cfg.get(concept, self._calib_cfg.get("default", {}))
            T = float(c.get("T", 1.0))
            b = float(c.get("b", 0.0))
        z = (self._logit(p_raw) + b) / max(1e-6, T)
        return float(1.0 / (1.0 + math.exp(-z)))

    def _gate_fuse(self, concept: str, p0: float, p1: float, margin: float | None) -> float:
        tau = float(self.cfg.tau)
        gamma = float(self.cfg.gamma_gate)
        eta = float(self.cfg.eta)

        if self._calib_cfg:
            c = self._calib_cfg.get(concept, self._calib_cfg.get("default", {}))
            tau = float(c.get("tau", tau))
            gamma = float(c.get("gamma", gamma))
            eta = float(c.get("eta", eta))

        p0 = float(self._clip01(p0))
        p1 = float(self._clip01(p1))
        p_mix = p0 + eta * (p1 - p0)

        gain_tool = abs(p_mix - 0.5) + gamma * abs(p_mix - p0)
        gain_base = abs(p0 - 0.5)
        if margin is not None:
            try:
                gain_tool += 0.5 * max(0.0, float(margin))
            except Exception:
                pass

        return float(p_mix if (gain_tool >= gain_base + tau) else p0)

    # ---------- check implementations ----------
    def _check_prior(self, policy, image, concept: str, ctx_text: str) -> Tuple[float, dict]:
        t0 = time.perf_counter()
        q = float(self._score_yesno_with_context(policy, image, concept, ctx_text))
        ms = (time.perf_counter() - t0) * 1000.0
        return q, {"tool": "prior", "margin": 0.0, "roi": None, "tool_time_ms": ms, "decision": "n/a"}

    def _check_grid(
        self,
        policy,
        image,
        concept: str,
        ctx_text: str,
        forbidden: Optional[Set[int]] = None,
    ) -> Tuple[float, dict]:
        t0 = time.perf_counter()
        pil = _to_pil(image)
        if pil is None:
            return 0.5, {
                "tool": "grid",
                "margin": 0.0,
                "roi": None,
                "cell_id": None,
                "tool_time_ms": 0.0,
                "decision": "no_image",
            }

        w, h = pil.size
        boxes = _grid_boxes(w, h, getattr(self.cfg, "grid_n", 3))
        forbid = forbidden or set()

        scores: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        for (x1, y1, x2, y2, cid) in boxes:
            if (cid in forbid) and (not getattr(self.cfg, "grid_allow_repeat", False)):
                continue
            crop = pil.crop((x1, y1, x2, y2))
            q = float(self._score_yesno_with_context(policy, crop, concept, ctx_text))
            scores.append((q, cid, (x1, y1, x2, y2)))

        if not scores:
            return 0.5, {
                "tool": "grid",
                "margin": 0.0,
                "roi": None,
                "cell_id": None,
                "tool_time_ms": (time.perf_counter() - t0) * 1000.0,
                "decision": "no_candidate",
            }

        scores.sort(key=lambda x: x[0], reverse=True)
        q1, cid1, box1 = scores[0]
        q2 = scores[1][0] if len(scores) > 1 else 0.5
        margin = float(q1 - q2)

        ms = (time.perf_counter() - t0) * 1000.0
        extra = {
            "tool": "grid",
            "margin": margin,
            "roi": [int(v) for v in box1],
            "cell_id": int(cid1),
            "tool_time_ms": ms,
            "decision": "ok",
        }
        return float(q1), extra

    def _check_self(
        self,
        policy,
        image,
        concept: str,
        ctx_text: str,
        forbidden: Optional[Set[int]] = None,
    ) -> Tuple[float, dict]:
        mode = str(getattr(self.cfg, "self_mode", "grid")).lower()
        if mode in ("grid", "self", ""):
            return self._check_grid(policy, image, concept, ctx_text, forbidden=forbidden)

        # placeholder bbox mode -> fallback to grid
        q, extra = self._check_grid(policy, image, concept, ctx_text, forbidden=forbidden)
        extra["decision"] = f"bbox_placeholder->grid({extra.get('decision','')})"
        extra["tool"] = "self_bbox_placeholder"
        return q, extra

    def _check_kbcs(self, image: str, concept: str) -> Tuple[float, dict]:
        try:
            from adapters.kbcs_check import score as kbcs_score
        except Exception as e:
            return 0.5, {
                "tool": "kbcs",
                "error": f"import_failed:{e}",
                "margin": 0.0,
                "roi": None,
                "tool_time_ms": 0.0,
                "decision": "uncertain",
            }

        t0 = time.perf_counter()
        try:
            out = kbcs_score(image, concept, self._rt)
        except Exception as e:
            out = {"p": 0.5, "p_raw": 0.5, "margin": 0.0, "roi": None, "decision": f"error:{e}"}
        ms = (time.perf_counter() - t0) * 1000.0

        p_raw = float(out.get("p_raw", out.get("p", 0.5)))
        q = self._calibrate_tool_p(concept, p_raw)
        extra = {
            "tool": "kbcs",
            "p_raw": p_raw,
            "margin": float(out.get("margin", 0.0)),
            "roi": out.get("roi", None),
            "decision": str(out.get("decision", "uncertain")),
            "tool_time_ms": float(ms),
        }
        return q, extra

    # ---------- main rollout ----------
    def rollout_once(self, policy, ex: Dict[str, Any]) -> Tuple[float, int, List[Dict[str, Any]]]:
        image = ex.get("image", None)
        concept = ex["concept"]

        # initial prior p0
        if self.cfg.use_qwen_prior and hasattr(policy, "score_yesno"):
            p = float(policy.score_yesno(image, concept))
        else:
            p = float(ex.get("p_baseline", 0.5))
        p0 = float(self._clip01(p))

        traj: List[Dict[str, Any]] = []
        checked_cells: Set[int] = set()
        steps = 0
        ever_checked = False

        ctx_text = f"Concept: {concept}"
        valid_mask_list = self._valid_mask()

        while steps < int(self.cfg.max_steps):
            decide_prompt = (
                f"{ctx_text}\n"
                "You can act next: <CHECK>/<CLAIM>/<ABSTAIN>/<STOP>. "
                "Choose exactly ONE token."
            )

            action, logp_cur_t, logp_behav, H_t, KL_t = policy.sample_action_train(
                prompt=decide_prompt,
                image=image,
                valid_mask=valid_mask_list,
            )
            steps += 1

            rec: Dict[str, Any] = {
                "action": action,
                "logp_cur": float(logp_cur_t.item()),
                "logp_cur_t": logp_cur_t,
                "logp_behav": float(logp_behav),
                "H": float(H_t.item()),
                "H_t": H_t,
                "KL": float(KL_t.item()),
                "KL_t": KL_t,
            }
            traj.append(rec)

            # ---- execute action ----
            if action == "<CHECK>":
                ever_checked = True
                q: Optional[float] = None
                extra: Dict[str, Any] = {"tool_time_ms": 0.0, "margin": 0.0, "roi": None, "decision": "n/a"}

                # 1) evidence source
                if self.cfg.check_source == "prior":
                    q, extra = self._check_prior(policy, image, concept, ctx_text)
                    hint = self._kbcs_hint(ex)
                    ctx_text += f"\n[CHECK/prior] {hint}"
                elif self.cfg.check_source == "kbcs":
                    q, extra = self._check_kbcs(image, concept)
                elif self.cfg.check_source == "grid":
                    q, extra = self._check_grid(policy, image, concept, ctx_text, forbidden=checked_cells)
                    if extra.get("cell_id") is not None:
                        checked_cells.add(int(extra["cell_id"]))
                elif self.cfg.check_source == "self":
                    q, extra = self._check_self(policy, image, concept, ctx_text, forbidden=checked_cells)
                    if extra.get("cell_id") is not None:
                        checked_cells.add(int(extra["cell_id"]))
                else:
                    q = None

                # 2) fuse
                p_before = float(p)
                if q is None:
                    p = p_before
                elif self.cfg.fuse_mode == "none":
                    p = float(q)
                elif self.cfg.fuse_mode == "mix":
                    p = self._blend(p_before, float(q), float(self.cfg.eta))
                elif self.cfg.fuse_mode == "gate":
                    p = self._gate_fuse(concept, p_before, float(q), extra.get("margin", None))
                else:
                    # legacy
                    p = self._blend(p_before, float(q), float(self.cfg.check_alpha))

                adopted = (abs(p - p_before) > 1e-8)

                # 3) write back to step
                traj[-1].update(
                    {
                        "tool": str(self.cfg.check_source),
                        "fuse_mode": str(self.cfg.fuse_mode),
                        "p_before": float(p_before),
                        "q": (float(q) if q is not None else None),
                        "p_after": float(p),
                        "tool_time_ms": float(extra.get("tool_time_ms", 0.0)),
                        "margin": float(extra.get("margin", 0.0)),
                        "roi": extra.get("roi"),
                        "decision": str(extra.get("decision", "n/a")),
                        "adopted": int(adopted),
                    }
                )

                # 4) add to context for later steps (optional)
                try:
                    q_txt = f"{float(q):.3f}" if q is not None else "n/a"
                    m_txt = f"{float(extra.get('margin', 0.0)):.3f}"
                    roi = extra.get("roi", None)
                    ctx_text += f"\n[CHECK/{self.cfg.check_source}] q={q_txt}, margin={m_txt}, roi={roi}, adopted={int(adopted)}"
                except Exception:
                    pass

                # 5) debug row
                if self._debug_on:
                    self._log_traj(
                        {
                            "id": self._ex_id(ex),
                            "image": image,
                            "concept": concept,
                            "t": steps,
                            "action": action,
                            "p_before": float(p_before),
                            "q": (float(q) if q is not None else None),
                            "p_after": float(p),
                            "tool": self.cfg.check_source,
                            "fuse_mode": self.cfg.fuse_mode,
                            "tool_time_ms": float(extra.get("tool_time_ms", 0.0)),
                            "margin": float(extra.get("margin", 0.0)),
                            "roi": extra.get("roi"),
                            "decision": extra.get("decision", "n/a"),
                            "adopted": int(adopted),
                        }
                    )
                # continue loop unless max_steps reached
                continue

            if action == "<ABSTAIN>":
                if self._debug_on:
                    self._log_traj({"id": self._ex_id(ex), "t": steps, "action": action, "p_before": float(p), "p_after": 0.5})
                p = 0.5
                break

            if action == "<STOP>":
                if self._debug_on:
                    self._log_traj({"id": self._ex_id(ex), "t": steps, "action": action, "p_before": float(p), "p_after": float(p)})
                break

            if action == "<CLAIM>":
                p_before = float(p)
                p = self._sharpen(p_before, float(self.cfg.claim_gamma))
                if self._debug_on:
                    self._log_traj({"id": self._ex_id(ex), "t": steps, "action": action, "p_before": float(p_before), "p_after": float(p)})
                break

            # unknown action -> stop
            break

        # final: optionally revert to p0 if never checked
        if (not ever_checked) and bool(self.cfg.use_baseline_when_no_check):
            p_final = p0
        else:
            p_final = float(p)

        p_final = float(self._clip01(p_final))

        if self._debug_on:
            def _exportable_step(st: dict) -> dict:
                keep = [
                    "action",
                    "logp_cur",
                    "logp_behav",
                    "H",
                    "KL",
                    "tool",
                    "fuse_mode",
                    "p_before",
                    "q",
                    "p_after",
                    "tool_time_ms",
                    "margin",
                    "roi",
                    "decision",
                    "adopted",
                ]
                out = {}
                for k in keep:
                    if k in st:
                        out[k] = st[k]
                return out

            case = {
                "id": self._ex_id(ex),
                "image": (pathlib.Path(image).name if image else ""),
                "image_path": image,
                "concept": concept,
                "traj": [_exportable_step(s) for s in traj],
            }
            os.makedirs("artifacts/debug", exist_ok=True)
            with open("artifacts/debug/rl_traj_cases.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        return float(p_final), int(steps), traj
