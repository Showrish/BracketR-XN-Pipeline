"""DECIMER Image Transformer (Rajan et al., 2023): the second-opinion recognizer.

Pretrained model from the ``decimer`` package, used as-is. It is weaker than
MolScribe overall in our Phase 1 benchmark, but it was the stronger tool on
chiral images (0.53 against 0.47), so stage 4 can use the two together and flag
crops where they disagree. It gives no confidence score.

Needs TensorFlow, so it runs in its own environment (envs/stage3_decimer.yml).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..config import DecimerConfig
from .base import Recognizer


class DecimerRecognizer(Recognizer):
    name = "DECIMER"

    def __init__(self, cfg: DecimerConfig):
        from DECIMER import predict_SMILES

        self._predict_smiles = predict_SMILES

    def _predict(self, image_path: Path) -> Tuple[str, Optional[float]]:
        return self._predict_smiles(str(image_path)), None
