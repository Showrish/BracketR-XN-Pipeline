"""Common interface for the recognition models used in stage 3a."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Prediction:
    smiles: str = ""
    confidence: Optional[float] = None   # None when the model gives no confidence
    runtime_sec: float = 0.0
    error: str = ""


class Recognizer(ABC):
    """A model that reads one molecule image and returns SMILES."""

    name: str = "recognizer"

    @abstractmethod
    def _predict(self, image_path: Path) -> Tuple[str, Optional[float]]:
        """Return (smiles, confidence) for one prepared crop."""

    def predict(self, image_path: Path) -> Prediction:
        """Time one prediction and turn any failure into an error message,
        so one unreadable crop does not stop the whole run."""
        start = time.perf_counter()
        try:
            smiles, confidence = self._predict(Path(image_path))
            error = ""
        except Exception as exc:  # noqa: BLE001 - every model failure is recorded, not raised
            smiles, confidence, error = "", None, f"{type(exc).__name__}: {exc}"
        return Prediction(smiles=smiles or "", confidence=confidence,
                          runtime_sec=round(time.perf_counter() - start, 3), error=error)
