# -*- coding: utf-8 -*-
"""
training/policy.py

Qwen3-VL friendly policy wrapper for IIRA (single-GPU 24G friendly).

Key change (to fix "actions stuck uniform"):
- ✅ Replaces "action logits from 4 special-token LM head rows" with an explicit, trainable 4-way action_head (Linear).
  New special tokens often share near-identical LM-head rows, causing uniform actions and no learning signal.
- Keeps behavior snapshot on CPU for IS/KL-style RL, and now also keeps a behav_action_head synced from action_head.
- Saves/loads action_head weights separately as action_rows.safetensors (because save_pretrained() won't include it).

Other features preserved:
- Qwen3-VL official processor.apply_chat_template(...) path
- Optional 4-bit + LoRA (PEFT)
- make_yesno_sft_batch() for train_yesno
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,  # fallback
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save
except Exception:
    _st_load = None
    _st_save = None


DEBUG_DIR = pathlib.Path("artifacts/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = ["<CLAIM>", "<CHECK>", "<ABSTAIN>", "<STOP>"]
ACTION2IDX = {a:i for i,a in enumerate(ACTIONS)}
IDX2ACTION = {i:a for a,i in ACTION2IDX.items()}

# -------------------------
# helpers
# -------------------------

def _to_pil(img: Any | None) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, (str, pathlib.Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return img


def _resize_short_edge(img: Image.Image, s: int) -> Image.Image:
    if s is None or s <= 0:
        return img
    w, h = img.size
    if min(w, h) == s:
        return img
    if w <= h:
        new_w = s
        new_h = int(round(h * s / max(1, w)))
    else:
        new_h = s
        new_w = int(round(w * s / max(1, h)))
    return img.resize((new_w, new_h), resample=Image.BILINEAR)


def _first_device(module: torch.nn.Module) -> torch.device:
    for p in module.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)


def _enable_input_require_grads(model: torch.nn.Module):
    """Ensure at least one input tensor to checkpointed blocks requires grad.

    For re-entrant gradient checkpointing, if *all* inputs to a checkpointed function
    have requires_grad=False, PyTorch will run forward under no_grad and parameter
    gradients inside the checkpointed region will be None.

    Transformers provides `enable_input_require_grads()` for this exact purpose.
    """
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            return
        except Exception:
            pass

    # Fallback: register a hook on input embeddings to set outputs requires_grad=True
    try:
        emb = model.get_input_embeddings()
        if emb is None:
            return

        def _hook(module, inputs, output):
            try:
                if hasattr(output, "requires_grad") and (not output.requires_grad):
                    output.requires_grad_(True)
            except Exception:
                pass

        if not hasattr(emb, "_iira_reqgrad_hook"):
            emb._iira_reqgrad_hook = emb.register_forward_hook(_hook)
    except Exception:
        pass


def _ensure_action_ids(tok) -> Dict[str, int]:
    # NOTE: We keep registering these special tokens for backward compatibility
    # (they may be used elsewhere), but we no longer use their LM-head rows as the action policy.
    ids: Dict[str, int] = {}
    for a in ACTIONS:
        tid = tok.convert_tokens_to_ids(a)
        if tid is None or tid == tok.unk_token_id:
            raise ValueError(
                f"Special token {a} not registered correctly. "
                f"Please ensure tokenizer has it in additional_special_tokens."
            )
        ids[a] = tid
    return ids


def _safe_categorical_from_logits(
    logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    fallback_idx: int = 0,
    eps: float = 1e-12,
):
    x = logits.clone()
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, A]

    if valid_mask is not None:
        vm = valid_mask.to(dtype=torch.bool, device=x.device)
        x = torch.where(vm, x, torch.full_like(x, -1e9))

    x = torch.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)
    p = F.softmax(x, dim=-1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = p.sum(dim=-1)
    bad = (row_sum <= eps)
    info = {"fallback": "none", "bad_rows": []}

    if bad.any():
        if valid_mask is not None:
            vm = valid_mask.to(x.device)
            uni_logits = torch.where(vm, torch.zeros_like(x), torch.full_like(x, -1e9))
        else:
            uni_logits = torch.zeros_like(x)

        p_uni = F.softmax(uni_logits, dim=-1)
        still_bad = (p_uni.sum(dim=-1) <= eps)

        if still_bad.any():
            fb = torch.full_like(x, -1e9)
            fb[:, fallback_idx] = 0.0
            x = torch.where(bad.unsqueeze(-1), fb, x)
            info["fallback"] = f"force_single@{fallback_idx}"
        else:
            x = torch.where(bad.unsqueeze(-1), uni_logits, x)
            info["fallback"] = "uniform_valid"

        p = F.softmax(x, dim=-1)
        info["bad_rows"] = torch.nonzero(bad, as_tuple=False).view(-1).tolist()

    dist = Categorical(logits=x)
    return dist, x.squeeze(0), p.squeeze(0), info


def _detect_model_type(model_name: str) -> str:
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return str(getattr(cfg, "model_type", ""))
    except Exception:
        return ""


def _get_model_cls(model_type: str):
    mt = (model_type or "").lower()
    if "qwen3_vl" in mt or "qwen3-vl" in mt:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    return AutoModelForCausalLM


def _from_pretrained_compat(ModelCls, model_name: str, **kwargs):
    if "dtype" in kwargs:
        try:
            return ModelCls.from_pretrained(model_name, **kwargs)
        except TypeError:
            dtype_val = kwargs.pop("dtype")
            kwargs["torch_dtype"] = dtype_val
            return ModelCls.from_pretrained(model_name, **kwargs)
    return ModelCls.from_pretrained(model_name, **kwargs)


def _infer_hidden_size(model: torch.nn.Module) -> int:
    """Infer transformer hidden size robustly across model classes.

    Preference order:
    1) Common config fields (hidden_size, n_embd, d_model, etc.)
    2) Nested text_config (for multi-modal configs)
    3) Input embedding dimension / weight shape
    4) Output embedding (lm_head) weight shape
    """
    cfg = getattr(model, "config", None)

    # (1) common config fields
    for attr in ("hidden_size", "n_embd", "d_model", "dim", "model_dim"):
        if cfg is not None and hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, int) and v > 0:
                return int(v)

    # (2) nested configs often used by multi-modal models
    for sub in ("text_config", "language_config", "llm_config"):
        subcfg = getattr(cfg, sub, None) if cfg is not None else None
        if subcfg is not None:
            for attr in ("hidden_size", "n_embd", "d_model", "dim", "model_dim"):
                if hasattr(subcfg, attr):
                    v = getattr(subcfg, attr)
                    if isinstance(v, int) and v > 0:
                        return int(v)

    # (3) embeddings (most reliable)
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            if hasattr(emb, "embedding_dim") and isinstance(getattr(emb, "embedding_dim"), int):
                return int(getattr(emb, "embedding_dim"))
            w = getattr(emb, "weight", None)
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                return int(w.shape[1])
    except Exception:
        pass

    # (4) output embeddings / lm head
    try:
        out = model.get_output_embeddings()
        if out is not None:
            w = getattr(out, "weight", None)
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                return int(w.shape[1])
    except Exception:
        pass

    raise RuntimeError(
        "Cannot infer hidden size. Please ensure the model exposes input/output embeddings "
        "or add a mapping for this model type in _infer_hidden_size()."
    )



# -------------------------
# Policy
# -------------------------

class Policy:
    """
    Public API kept:
      - sample_action_train(prompt, image, valid_mask) -> (action, logp_cur_t, logp_behav_float, H_t, KL_t)
      - sample_action_eval(...)
      - score_yesno(image, concept)
      - yesno_logprobs(image, prompt)
      - make_yesno_sft_batch(image, concept, y01)

    New:
      - save_action_rows(out_dir)
      - load_action_rows(file_path)
    """

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        behavior_init: Optional[str] = None,  # backward compat, unused
        *,
        short_edge: int = 384,
        load_4bit: bool = True,
        behav_mode: str = "none",            # "none" | "cpu"
        behav_on_cpu: Optional[bool] = None, # legacy flag
        grad_ckpt: bool = True,
        attn_impl: Optional[str] = None,     # e.g. "flash_attention_2"
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        self.model_name = model_name
        self.short_edge = int(short_edge)

        if behav_on_cpu is not None:
            self.behav_mode = "cpu" if bool(behav_on_cpu) else "none"
        else:
            self.behav_mode = str(behav_mode).lower().strip()

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # add action tokens (kept for compatibility, but not used for action policy logits anymore)
        added = self.tok.add_special_tokens({
            "additional_special_tokens": [
                t for t in ACTIONS if self.tok.convert_tokens_to_ids(t) in (None, self.tok.unk_token_id)
            ]
        })

        model_type = _detect_model_type(model_name)
        ModelCls = _get_model_cls(model_type)

        # 4bit quant (optional)
        quant4 = None
        if load_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("load_4bit=True but BitsAndBytesConfig not available (bitsandbytes missing/mismatch).")
            quant4 = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if torch.cuda.is_available() else {"": "cpu"},
            "dtype": "auto",  # Qwen3 official
        }
        if quant4 is not None:
            load_kwargs["quantization_config"] = quant4
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        self.model = _from_pretrained_compat(ModelCls, model_name, **load_kwargs)

        if added > 0:
            self.model.resize_token_embeddings(len(self.tok))
        # LoRA / adapter loading
        if load_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Build a default LoRA config (used when creating a fresh adapter, and as a fallback
        # for behav model structure if checkpoint config is unavailable).
        _default_lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        if lora_path:
            # IMPORTANT: load adapter weights onto the *base* model.
            # Do NOT call get_peft_model() first, otherwise you'll end up with multiple adapters
            # and lots of missing keys warnings.
            from peft import PeftModel
            base = self.model
            self.model = PeftModel.from_pretrained(base, lora_path, is_trainable=True)

            # Prefer the checkpoint's LoRA config for constructing the behavior model's adapter.
            lora_cfg = None
            try:
                pc = getattr(self.model, 'peft_config', None)
                if isinstance(pc, dict) and pc:
                    lora_cfg = list(pc.values())[0]
                else:
                    lora_cfg = pc
            except Exception:
                lora_cfg = None

            self.lora_cfg = lora_cfg if isinstance(lora_cfg, LoraConfig) else _default_lora_cfg
        else:
            self.lora_cfg = _default_lora_cfg
            self.model = get_peft_model(self.model, self.lora_cfg)

        # Gradient checkpointing can silently kill grads under the *re-entrant* variant
        # (PyTorch warns: "None of the inputs have requires_grad=True. Gradients will be None").
        # Using use_reentrant=False avoids that requirement and is the recommended path.
        if grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            _enable_input_require_grads(self.model)
            try:
                self.model.config.use_cache = False
            except Exception:
                pass

            # HF supports passing kwargs; older versions may not.
            enabled = False
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                enabled = True
            except TypeError:
                try:
                    self.model.gradient_checkpointing_enable(use_reentrant=False)
                    enabled = True
                except TypeError:
                    # Fall back to default behavior (may still warn / drop grads).
                    self.model.gradient_checkpointing_enable()
                    enabled = True
            except Exception:
                pass
        elif (not grad_ckpt) and hasattr(self.model, "gradient_checkpointing_disable"):
            try:
                self.model.gradient_checkpointing_disable()
            except Exception:
                pass

        self.model.train()

        # === NEW: explicit trainable action head ===
        self.hidden_size = _infer_hidden_size(self.model)
        self.action_head = nn.Linear(self.hidden_size, 4, bias=True).to(_first_device(self.model))

        # Strongly recommended: bias initialization to avoid a cold-start collapse to <STOP>.
        # Order is [<CLAIM>, <CHECK>, <ABSTAIN>, <STOP>].
        # This makes early training explore CHECK so the agent can actually observe reward differences.
        with torch.no_grad():
            try:
                nn.init.zeros_(self.action_head.weight)
            except Exception:
                pass
            try:
                self.action_head.bias.zero_()
                self.action_head.bias[1] = 1.0   # prefer CHECK
                self.action_head.bias[3] = -0.5  # slightly discourage STOP
            except Exception:
                pass

        # behavior snapshot on CPU (optional)
        self.behav: Optional[torch.nn.Module] = None
        self.behav_action_head: Optional[nn.Linear] = None
        if self.behav_mode == "cpu":
            self.behav = _from_pretrained_compat(
                ModelCls,
                model_name,
                trust_remote_code=True,
                device_map={"": "cpu"},
                torch_dtype=torch.float32,
            )
            if added > 0:
                self.behav.resize_token_embeddings(len(self.tok))
            self.behav = get_peft_model(self.behav, self.lora_cfg)
            _freeze_module(self.behav)
            self.behav.eval()
            try:
                self.behav.config.use_cache = False
            except Exception:
                pass

            self.behav_action_head = nn.Linear(self.hidden_size, 4, bias=True).to(torch.device("cpu"))
            _freeze_module(self.behav_action_head)
            self._sync_behav_action_head()

        self.action_ids = _ensure_action_ids(self.tok)

        # If adapter dir contains action_rows.safetensors, load it (keeps eval/restore consistent)
        self._maybe_load_action_rows(lora_path)
        if self.behav is not None:
            self.refresh_behavior(show_progress=False)
    # -------------------------
    # action_rows save/load
    # -------------------------
    def save_action_rows(self, out_dir: str) -> str:
        """
        Save action_head weights to <out_dir>/action_rows.safetensors
        (separate from PEFT adapter files).
        """
        if _st_save is None:
            raise RuntimeError("safetensors not available; please install safetensors.")
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "action_rows.safetensors"
        sd = {
            "action_head.weight": self.action_head.weight.detach().cpu(),
            "action_head.bias": self.action_head.bias.detach().cpu(),
        }
        _st_save(sd, str(path))
        return str(path)

    def load_action_rows(self, file_path: str):
        if _st_load is None:
            raise RuntimeError("safetensors not available; please install safetensors.")
        sd = _st_load(file_path)
        w = sd.get("action_head.weight") or sd.get("W")
        b = sd.get("action_head.bias") or sd.get("b")
        if w is None or b is None:
            raise RuntimeError(f"Bad action_rows file: {file_path}; keys={list(sd.keys())}")
        dev = _first_device(self.model)
        with torch.no_grad():
            self.action_head.weight.copy_(w.to(dev))
            self.action_head.bias.copy_(b.to(dev))
        self._sync_behav_action_head()

    def _maybe_load_action_rows(self, lora_path: Optional[str]):
        if not lora_path:
            return
        p = pathlib.Path(lora_path)
        if p.is_dir():
            f = p / "action_rows.safetensors"
            if f.exists():
                try:
                    self.load_action_rows(str(f))
                except Exception:
                    # Do not hard-fail adapter loading on missing/old action_rows format
                    pass

    # -------------------------
    # behavior sync
    # -------------------------
    def _sync_behav_action_head(self):
        if self.behav_action_head is None:
            return
        with torch.no_grad():
            self.behav_action_head.load_state_dict({k: v.detach().cpu() for k, v in self.action_head.state_dict().items()})

    def refresh_behavior(self, show_progress: bool = False):
        if self.behav is None:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **kwargs): return x

        with torch.no_grad():
            src = dict(self.model.named_parameters())
            dst = dict(self.behav.named_parameters())

            def is_lora(k: str) -> bool:
                kl = k.lower()
                return "lora_" in kl or "loraup" in kl or "loradown" in kl

            keys = [k for k in dst.keys() if is_lora(k) and (k in src) and (src[k].shape == dst[k].shape)]
            for k in tqdm(keys, desc="[policy] Sync LoRA → behav", dynamic_ncols=True, disable=not show_progress):
                dst[k].data.copy_(src[k].detach().to(dst[k].device, dtype=dst[k].dtype))
        self.behav.eval()
        self._sync_behav_action_head()

    # -------------------------
    # inputs (Qwen3 path)
    # -------------------------
    def _build_inputs_qwen3(self, prompt: str, image: Any | None):
        pil_img = _to_pil(image)
        if pil_img is not None and self.short_edge:
            pil_img = _resize_short_edge(pil_img, self.short_edge)

        messages = [
            {"role": "user", "content": (
                ([{"type": "image", "image": pil_img}] if pil_img is not None else [])
                + [{"type": "text", "text": prompt}]
            )}
        ]

        inputs = self.proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "token_type_ids" in inputs:
            inputs.pop("token_type_ids", None)
        return inputs

    # -------------------------
    # ✅ SFT helper: build teacher-forcing batch for yes/no
    # -------------------------
    def make_yesno_sft_batch(self, image: Any | None, concept: str, y01: int) -> Dict[str, Any]:
        """
        Build a supervised batch where the model must generate " yes" or " no" as the next tokens.
        Returns dict with input_ids/attention_mask/(vision fields...)/labels.
        """
        y01 = int(y01)
        ans_text = " yes" if y01 == 1 else " no"  # keep leading space

        prompt = f"Based on the image, is there {concept}? Answer 'yes' or 'no'."
        base = self._build_inputs_qwen3(prompt, image)

        # tokenize answer without special tokens
        ans_ids = self.tok(ans_text, add_special_tokens=False).input_ids
        ans = torch.tensor([ans_ids], dtype=base["input_ids"].dtype)

        # concat seq fields
        input_ids = torch.cat([base["input_ids"], ans], dim=1)
        attn = torch.cat([base["attention_mask"], torch.ones_like(ans)], dim=1)

        # labels: ignore prompt tokens, supervise answer tokens
        labels = torch.full_like(input_ids, -100)
        labels[:, -ans.size(1):] = ans

        batch: Dict[str, Any] = dict(base)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attn
        batch["labels"] = labels
        return batch

    # -------------------------
    # logits (NEW: action_head over hidden state)
    # -------------------------
    @staticmethod
    def _unwrap_for_core(m: torch.nn.Module):
        # PEFT: PeftModel.get_base_model() 返回底层 ForConditionalGeneration
        if hasattr(m, "get_base_model"):
            try:
                return m.get_base_model()
            except Exception:
                pass
        # 兜底
        return m

    def _action_logits(self, prompt: str, image: Any | None, *, use_behavior: bool) -> torch.Tensor:
        """
        Return 4 logits for ACTIONS in order:
        idx 0 -> <CLAIM>
        idx 1 -> <CHECK>
        idx 2 -> <ABSTAIN>
        idx 3 -> <STOP>
        """
        if use_behavior and self.behav is not None:
            m = self.behav
            head = self.behav_action_head
            assert head is not None
        else:
            m = self.model
            head = self.action_head

        user_suffix = "Respond with EXACTLY ONE token from <CLAIM>/<CHECK>/<ABSTAIN>/<STOP>."
        full_prompt = prompt.strip() + "\n" + user_suffix

        inputs = self._build_inputs_qwen3(full_prompt, image)
        dev = _first_device(m)
        inputs = _move_to_device(inputs, dev)

        # ✅ 不要 output_hidden_states=True（会保存所有层，8B 很容易 OOM）
        core = self._unwrap_for_core(m)

        # 优先走 core.model（BaseModel）拿 last_hidden_state
        if hasattr(core, "model"):
            out = core.model(**inputs, use_cache=False, return_dict=True)
            hs_last = out.last_hidden_state  # [1, T, H]
        else:
            # 兜底：如果没有 .model，只能要 hidden_states，但尽量只取最后一层
            out = core(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
            hs_last = out.hidden_states[-1]  # [1, T, H]

        h_last = hs_last[:, -1, :]  # [1, H]
        h_last = h_last.to(head.weight.device)
        logits = head(h_last).float()[0]  # [4]
        return logits


    # -------------------------
    # public API
    # -------------------------
    def sample_action_train(
        self,
        prompt: str,
        image: Any | None = None,
        valid_mask: Optional[List[bool]] = None,
    ):
        lg_cur = self._action_logits(prompt, image=image, use_behavior=False)

        vm_t = None
        if valid_mask is not None:
            vm_t = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.tensor(valid_mask)
            vm_t = vm_t.to(dtype=torch.bool, device=lg_cur.device)

        fb = ACTIONS.index("<STOP>")
        dist_cur, clean_lg_cur, p_cur, info_cur = _safe_categorical_from_logits(lg_cur, vm_t, fb)

        act_idx_t = dist_cur.sample()
        idx = int(act_idx_t.item())
        action = ACTIONS[idx]

        logp_cur_all = torch.log_softmax(clean_lg_cur, dim=-1)
        logp_cur_t = dist_cur.log_prob(act_idx_t)  # grad
        H_t = -(p_cur * logp_cur_all).sum()

        if self.behav is None:
            logp_behav = float(logp_cur_all[idx].detach().item())
            KL_t = torch.zeros((), device=lg_cur.device)
        else:
            with torch.no_grad():
                lg_beh = self._action_logits(prompt, image=image, use_behavior=True).to(lg_cur.device)
                _, clean_lg_beh, _, info_beh = _safe_categorical_from_logits(lg_beh, vm_t, fb)
                logp_beh_all = torch.log_softmax(clean_lg_beh, dim=-1)
                logp_behav = float(logp_beh_all[idx].item())
            KL_t = (p_cur * (logp_cur_all - logp_beh_all)).sum()
            if info_beh.get("fallback") != "none":
                info_cur = {**info_cur, "fallback_beh": info_beh.get("fallback")}

        if info_cur.get("fallback") != "none":
            rec = {
                "where": "train",
                **info_cur,
                "p_min": float(p_cur.min().item()),
                "p_max": float(p_cur.max().item()),
                "p_sum": float(p_cur.sum().item()),
            }
            with (DEBUG_DIR / "invalid_probs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return action, logp_cur_t, logp_behav, H_t, KL_t

    @torch.no_grad()
    def sample_action_eval(
        self,
        prompt: str,
        image: Any | None = None,
        valid_mask: Optional[List[bool]] = None,
    ):
        lg_cur = self._action_logits(prompt, image=image, use_behavior=False)

        vm_t = None
        if valid_mask is not None:
            vm_t = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.tensor(valid_mask)
            vm_t = vm_t.to(dtype=torch.bool, device=lg_cur.device)

        fb = ACTIONS.index("<STOP>")
        dist, clean_lg, p, info = _safe_categorical_from_logits(lg_cur, vm_t, fb)

        act_idx = dist.sample()
        idx = int(act_idx.item())
        action = ACTIONS[idx]

        logp_all = torch.log_softmax(clean_lg, dim=-1)
        logp = float(logp_all[idx].item())
        H = float(-(p * logp_all).sum().item())
        KL = 0.0

        if info.get("fallback") != "none":
            rec = {
                "where": "eval",
                **info,
                "p_min": float(p.min().item()),
                "p_max": float(p.max().item()),
                "p_sum": float(p.sum().item()),
            }
            with (DEBUG_DIR / "invalid_probs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return action, logp, logp, H, KL

    @torch.no_grad()
    def score_yesno(self, image: Any, concept: str) -> float:
        was_train = bool(getattr(self.model, "training", False))
        try:
            # 评估用 eval，避免 dropout 等随机性
            try:
                self.model.eval()
            except Exception:
                pass

            q = f"Based on the image, is there {concept}? Answer 'yes' or 'no'."
            inputs = self._build_inputs_qwen3(q, image)

            dev = _first_device(self.model)
            inputs = _move_to_device(inputs, dev)

            logits = self.model(**inputs, use_cache=False).logits[:, -1, :][0]
            yes_id = self.tok(" yes", add_special_tokens=False).input_ids[0]
            no_id = self.tok(" no", add_special_tokens=False).input_ids[0]
            p = torch.softmax(logits[[yes_id, no_id]], dim=-1)[0]
            return float(p)

        finally:
            if was_train:
                try:
                    self.model.train()
                except Exception:
                    pass


    @torch.no_grad()
    def yesno_logprobs(self, image: Any, prompt: str) -> Dict[str, float]:
        """
        Return log-prob of generating answer " yes" vs " no" (can be multi-token)
        given the prompt (image optional). Uses teacher-forcing in ONE forward.
        """
        self.model.eval()

        base = self._build_inputs_qwen3(prompt, image)
        dev = _first_device(self.model)
        base = _move_to_device(base, dev)

        base_ids = base["input_ids"]          # [1, L]
        base_attn = base["attention_mask"]    # [1, L]
        L = int(base_ids.size(1))

        def _seq_logprob(ans_text: str) -> float:
            ans_ids = self.tok(ans_text, add_special_tokens=False).input_ids
            if len(ans_ids) == 0:
                return float("-inf")
            ans = torch.tensor([ans_ids], device=base_ids.device, dtype=base_ids.dtype)  # [1, T]
            attn_ans = torch.ones_like(ans, device=base_attn.device)

            input_ids = torch.cat([base_ids, ans], dim=1)            # [1, L+T]
            attention_mask = torch.cat([base_attn, attn_ans], dim=1)  # [1, L+T]

            batch = dict(base)
            batch["input_ids"] = input_ids
            batch["attention_mask"] = attention_mask

            out = self.model(**batch, use_cache=False)
            logits = out.logits  # [1, L+T, V]

            logp = 0.0
            log_probs = torch.log_softmax(logits, dim=-1)

            for i, tid in enumerate(ans_ids):
                pos = (L + i - 1)
                lp = float(log_probs[0, pos, tid].item())
                logp += lp
            return float(logp)

        lp_yes = _seq_logprob(" yes")
        lp_no = _seq_logprob(" no")
        return {"yes": lp_yes, "no": lp_no}
