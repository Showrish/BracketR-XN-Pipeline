"""Hand-off files: the stage 1-2 manifest this stage reads, and the files it writes.

Input columns follow ``stage1_2_detection/manifest.py`` (one row per candidate
region). Output columns are the formats agreed in the Assignment 1.4 plan:

* ``predictions_<tool>.csv`` (3a -> stage 4)
* ``markush_compounds.csv``  (3b -> stage 4)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

LABEL_MOLECULE = "molecule"
LABEL_MARKUSH = "markush"
STAGE2_REQUIRED = ("paper_id", "crop_id", "crop_path", "source_pdf", "page", "dpi", "label")

PREDICTION_COLUMNS = ["paper_id", "page", "crop_id", "tool", "predicted_smiles",
                      "confidence", "runtime_sec", "error"]
MARKUSH_COLUMNS = ["paper_id", "page", "crop_id", "compound_label", "smiles", "method"]


@dataclass
class CropTask:
    """One accepted crop from stage 1-2, with its paths resolved."""
    paper_id: str
    crop_id: str
    page: int
    dpi: int
    label: str                       # molecule | markush
    image_path: Path
    source_pdf: Optional[Path]       # None if the PDF cannot be found
    markush_evidence: str = ""

    @property
    def key(self) -> str:
        return f"{self.paper_id}/{self.crop_id}"


def _find_pdf(name: str, manifest_dir: Path, pdf_dir: Optional[Path]) -> Optional[Path]:
    """The manifest stores the PDF name; look for it next to the manifest or in pdf_dir."""
    if not name:
        return None
    candidates = [Path(name), manifest_dir / name]
    if pdf_dir is not None:
        candidates.append(Path(pdf_dir) / Path(name).name)
    return next((p for p in candidates if p.is_file()), None)


def read_stage2_manifest(path: str | Path, pdf_dir: Optional[str | Path] = None,
                         labels: Sequence[str] = (LABEL_MOLECULE, LABEL_MARKUSH)) -> List[CropTask]:
    """Read stage 1-2's manifest.csv and keep the crops stage 3 should process."""
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(STAGE2_REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing the columns {sorted(missing)}")
        records = list(reader)

    tasks: List[CropTask] = []
    for rec in records:
        if rec["label"] not in labels:
            continue                  # not_molecule rows were dropped by the gate
        tasks.append(CropTask(
            paper_id=rec["paper_id"], crop_id=rec["crop_id"], page=int(rec["page"]),
            dpi=int(rec["dpi"]), label=rec["label"],
            image_path=(path.parent / rec["crop_path"]).resolve(),
            source_pdf=_find_pdf(rec["source_pdf"], path.parent, Path(pdf_dir) if pdf_dir else None),
            markush_evidence=rec.get("markush_evidence", "") or ""))
    return tasks


def write_rows(path: str | Path, columns: Sequence[str], rows: Iterable[Dict]) -> Path:
    """Write rows with a fixed column order (missing values are left empty)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    return path
