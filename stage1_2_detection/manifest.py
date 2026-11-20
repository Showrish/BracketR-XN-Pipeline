"""Hand-off format between stage 1-2 and the later stages.

One row per candidate region found on a page. Column names for the position
(``crop_path, source_pdf, page, x, y, width, height``) follow the convention of
the course example pipeline, so other modules can read the file unchanged.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

LABEL_MOLECULE = "molecule"
LABEL_MARKUSH = "markush"
LABEL_REJECTED = "not_molecule"
ACCEPTED_LABELS = (LABEL_MOLECULE, LABEL_MARKUSH)


@dataclass
class Candidate:
    paper_id: str
    crop_id: str                 # e.g. p003_c02 (page 3, region 2)
    crop_path: str               # relative to the output folder
    source_pdf: str
    page: int                    # 1-based
    x: Optional[int]             # pixel box on the rendered page (None if unknown)
    y: Optional[int]
    width: Optional[int]
    height: Optional[int]
    dpi: int
    label: str                   # molecule | markush | not_molecule
    gate_score: float            # 0-1 confidence that this is a molecule drawing
    reason: str                  # why the gate decided this way
    markush_evidence: str        # R-group tokens found near the drawing
    detector: str                # classical | decimer


COLUMNS: List[str] = [f.name for f in fields(Candidate)]


def write_manifest(rows: Iterable[Candidate], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def _to_int(value: str) -> Optional[int]:
    return int(value) if value not in ("", None) else None


def read_manifest(path: str | Path) -> List[Candidate]:
    rows: List[Candidate] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            rows.append(Candidate(
                paper_id=rec["paper_id"], crop_id=rec["crop_id"],
                crop_path=rec["crop_path"], source_pdf=rec["source_pdf"],
                page=int(rec["page"]), x=_to_int(rec["x"]), y=_to_int(rec["y"]),
                width=_to_int(rec["width"]), height=_to_int(rec["height"]),
                dpi=int(rec["dpi"]), label=rec["label"],
                gate_score=float(rec["gate_score"]), reason=rec["reason"],
                markush_evidence=rec["markush_evidence"], detector=rec["detector"]))
    return rows


def select_labels(rows: Sequence[Candidate], labels: Sequence[str]) -> List[Candidate]:
    """Filter rows by gate label, e.g. ``("molecule",)`` for stage 3a."""
    return [r for r in rows if r.label in labels]
