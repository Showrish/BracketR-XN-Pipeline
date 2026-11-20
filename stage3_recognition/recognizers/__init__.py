"""Recognizers for stage 3a. Each model is imported only when it is used, so
the package works in an environment that has just one of them installed."""
from __future__ import annotations

from ..config import Stage3Config
from .base import Prediction, Recognizer

AVAILABLE = ("molscribe", "decimer")


def build_recognizer(tool: str, cfg: Stage3Config) -> Recognizer:
    """Create the recognizer named in the config or on the command line."""
    if tool == "molscribe":
        from .molscribe_recognizer import MolScribeRecognizer
        return MolScribeRecognizer(cfg.molscribe)
    if tool == "decimer":
        from .decimer_recognizer import DecimerRecognizer
        return DecimerRecognizer(cfg.decimer)
    raise ValueError(f"Unknown tool '{tool}'. Choose from {AVAILABLE}.")


__all__ = ["AVAILABLE", "Prediction", "Recognizer", "build_recognizer"]
