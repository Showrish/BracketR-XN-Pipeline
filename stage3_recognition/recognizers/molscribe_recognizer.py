"""MolScribe (Qian et al., 2023): the main recognizer.

Pretrained checkpoint ``swin_base_char_aux_1m680k.pth`` (Swin-B image encoder,
trained by the authors on about 1M rendered PubChem molecules and 680K USPTO
patent images), downloaded from Hugging Face and used as-is: no fine-tuning.
It was chosen because it was clearly the strongest reader in our Phase 1
benchmark: 0.46 exact match on 100 BioVista images against 0.30 for DECIMER
(McNemar p = 0.0015), and it returns a confidence score.

Needs PyTorch, so it runs in its own environment (envs/stage3_molscribe.yml).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from ..config import MolScribeConfig
from .base import Recognizer


class MolScribeRecognizer(Recognizer):
    name = "MolScribe"

    def __init__(self, cfg: MolScribeConfig):
        # MolScribe imports matplotlib. Notebooks (e.g. Colab) set a notebook-only
        # plotting backend that breaks that import, so use the headless one.
        os.environ["MPLBACKEND"] = "Agg"
        import torch
        from huggingface_hub import hf_hub_download
        from molscribe import MolScribe

        if cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)
        checkpoint = hf_hub_download(repo_id=cfg.hf_repo, filename=cfg.checkpoint)
        self.model = MolScribe(checkpoint, device=device)

    def _predict(self, image_path: Path) -> Tuple[str, Optional[float]]:
        output = self.model.predict_image_file(str(image_path), return_confidence=True)
        return output.get("smiles", ""), output.get("confidence")
