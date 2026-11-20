"""Configuration for stage 3 (recognition + Markush branch).

Defaults live in the dataclasses below and can be overridden from a YAML file
(see ``default_config.yaml``) or from the command line. Sizes are in inches,
like stage 1-2, so they do not depend on the render resolution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessConfig:
    pad_in: float = 0.05             # white margin added around each crop
    background: int = 255            # grey level used for the margin and for transparency
    save_prepared: bool = True       # keep the prepared crops so they can be checked by eye


@dataclass
class MolScribeConfig:
    hf_repo: str = "yujieq/MolScribe"                  # Hugging Face repository of the weights
    checkpoint: str = "swin_base_char_aux_1m680k.pth"  # pretrained checkpoint, used as-is
    device: str = "auto"                               # auto | cpu | cuda


@dataclass
class DecimerConfig:
    pass                             # DECIMER needs no settings; kept for symmetry


@dataclass
class MarkushConfig:
    enabled: bool = True
    scaffold_tool: str = "molscribe"  # recognizer that reads the scaffold drawing
    table_source: str = "text_layer"  # text_layer | none (a CSV can always be given on the CLI)
    search_next_page: bool = True     # R-group tables often sit on the page after the scaffold
    from_molecule_crops: bool = True  # also expand "molecule" crops whose reading has attachment points


@dataclass
class Stage3Config:
    tools: List[str] = field(default_factory=lambda: ["molscribe"])
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    molscribe: MolScribeConfig = field(default_factory=MolScribeConfig)
    decimer: DecimerConfig = field(default_factory=DecimerConfig)
    markush: MarkushConfig = field(default_factory=MarkushConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SECTIONS = ("preprocess", "molscribe", "decimer", "markush")


def _update(section: Any, values: Dict[str, Any], name: str) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in (values or {}).items():
        if key not in valid:
            raise ValueError(f"Unknown setting '{key}' in section '{name}'")
        setattr(section, key, value)


def load_config(path: Optional[str | Path] = None) -> Stage3Config:
    """Return the default configuration, optionally overridden by a YAML file."""
    cfg = Stage3Config()
    if path is None:
        return cfg
    import yaml  # imported lazily so the module works without PyYAML

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "tools" in data:
        cfg.tools = list(data["tools"])
    for name in SECTIONS:
        _update(getattr(cfg, name), data.get(name, {}), name)
    unknown = set(data) - set(SECTIONS) - {"tools"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return cfg
