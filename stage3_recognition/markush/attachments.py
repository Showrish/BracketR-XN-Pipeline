"""Attachment points in a scaffold SMILES.

DECIMER writes a variable group as a placeholder such as [R] or [R1], which RDKit
cannot parse. In our Phase 1 benchmark this caused 10 of DECIMER's 12 invalid
outputs on Markush images. Rewriting them as SMILES attachment points
([*], [1*]) turns the scaffold into a valid molecule; the number becomes the
isotope label, which ``enumerate.py`` uses to match R1, R2, ... to table columns.
"""
from __future__ import annotations

import re
from typing import List, Optional

R_PLACEHOLDER = re.compile(r"\[R(\d*)\]")          # [R], [R1] but not [Ru], [Rh], [Rb]
ATTACHMENT = re.compile(r"\[(\d*)\*\]|\*")           # [*], [1*] or a bare *


def normalize_attachment_points(smiles: str) -> str:
    """Replace R-group placeholders with attachment points RDKit can parse."""
    if not isinstance(smiles, str):
        return ""
    return R_PLACEHOLDER.sub(lambda m: f"[{m.group(1)}*]" if m.group(1) else "[*]", smiles)


def attachment_labels(smiles: str) -> List[Optional[int]]:
    """Number of each attachment point in order of appearance (None = unnumbered)."""
    if not isinstance(smiles, str):
        return []
    return [int(m.group(1)) if m.group(1) else None for m in ATTACHMENT.finditer(smiles)]
