# training/rl_training.py
# -*- coding: utf-8 -*-
"""Backward-compatible shim.

Older experiments imported `training.rl_training`.
The current code uses `training.rl_trainer`.
"""

from .rl_trainer import RLCfg, RLTrainer

__all__ = ["RLCfg", "RLTrainer"]
