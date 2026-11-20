"""Stage 3 of the BracketR XN pipeline: recognition (3a) and the Markush branch (3b).

Reads the manifest written by stage 1-2 and turns each accepted crop into SMILES.
"""
from .config import Stage3Config, load_config
from .handoff import MARKUSH_COLUMNS, PREDICTION_COLUMNS, CropTask, read_stage2_manifest
from .pipeline import run_stage3

__all__ = ["Stage3Config", "load_config", "CropTask", "read_stage2_manifest", "run_stage3",
           "PREDICTION_COLUMNS", "MARKUSH_COLUMNS"]
