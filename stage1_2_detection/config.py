"""Configuration for stage 1-2 (structure detection + molecule gate).

All sizes are given in inches so the settings do not depend on the render
resolution. Defaults can be overridden from a YAML file (see
``default_config.yaml``) or from the command line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RenderConfig:
    dpi: int = 300                      # page render resolution
    first_page: Optional[int] = None    # 1-based, inclusive; None = from start
    last_page: Optional[int] = None     # 1-based, inclusive; None = to the end


@dataclass
class DetectConfig:
    backend: str = "classical"          # "classical" or "decimer"
    merge_gap_in: float = 0.07          # strokes closer than this are merged
    pad_in: float = 0.06                # padding added around each region
    min_side_in: float = 0.30           # smaller regions are discarded
    max_page_fraction: float = 0.60     # larger regions are discarded
    label_attach_in: float = 0.12       # pull in atom labels (text layer) this close
    # --- splitting of merged panels (molecule glued to a chart/other panel) ---
    split_enabled: bool = True
    split_gap_in: float = 0.02          # finer merge gap used inside a suspicious region
    split_min_letters: int = 20         # many letters inside an accepted crop = possibly mixed
    split_min_area_in2: float = 4.0     # or a very large accepted crop
    split_min_area_rejected_in2: float = 1.0  # rejected chart-like crops this big are re-checked
    join_gap_in: float = 0.05           # ink this close = same molecule when re-joining
    decimer_expand: bool = True         # passed to DECIMER Segmentation
    decimer_match_threshold: float = 0.80


@dataclass
class GateConfig:
    # --- hard rejection rules -------------------------------------------
    max_ink_ratio: float = 0.30         # filled areas, heat maps
    max_midtone_fraction: float = 0.35  # photographs, gels, shaded plots
    max_color_fraction: float = 0.10    # coloured content (plots, kinome trees); molecules are black
    max_text_cc_fraction: float = 0.60  # share of components that sit in text rows
    text_layer_min_words: int = 4       # prose words from the PDF text layer
    axis_line_fraction: float = 0.80    # a line this long (vs crop) = axis/grid
    min_bond_segments: int = 4          # fewer straight strokes -> not a molecule
    min_diagonal_fraction: float = 0.15 # share of stroke length not at 0/90 deg
    max_curved_fraction: float = 0.40   # joints bending 8-40 deg = curves/circles
    max_letters_per_bond: float = 0.80  # more letters than this per stroke = chart/diagram
    label_heavy_min_letters: int = 15
    # --- stroke analysis ------------------------------------------------
    min_bond_len_in: float = 0.05       # shorter strokes are ignored (journal bonds ~0.09 in)
    angle_tolerance_deg: float = 10.0   # distance from a multiple of 30 deg
    ring_min_area_in2: float = 0.01
    ring_max_area_in2: float = 0.60
    # --- score ----------------------------------------------------------
    min_score: float = 0.50
    w_segments: float = 0.30
    w_angle: float = 0.25
    w_diagonal: float = 0.20
    w_ring: float = 0.25
    text_penalty: float = 0.30
    # --- Markush hint (text-layer / OCR evidence) -----------------------
    markush_pad_in: float = 0.25        # look this far around the box
    use_ocr_fallback: bool = False      # try pytesseract if no text layer


@dataclass
class Config:
    render: RenderConfig = field(default_factory=RenderConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update(section: Any, values: Dict[str, Any], name: str) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in (values or {}).items():
        if key not in valid:
            raise ValueError(f"Unknown setting '{key}' in section '{name}'")
        setattr(section, key, value)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Return the default configuration, optionally overridden by a YAML file."""
    cfg = Config()
    if path is None:
        return cfg
    import yaml  # imported lazily so the module works without PyYAML

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for name in ("render", "detect", "gate"):
        _update(getattr(cfg, name), data.get(name, {}), name)
    unknown = set(data) - {"render", "detect", "gate"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return cfg
