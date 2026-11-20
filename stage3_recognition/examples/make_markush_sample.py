"""Build the Markush test case: a scaffold drawing plus its R-group table.

Writes ``examples/sample_input/markush_sample/``:
* ``markush_paper.pdf``  one page with a para-disubstituted benzene scaffold
  (R1, R2) and a table of four compounds, typed as real text like a journal PDF;
* ``stage12/manifest.csv`` and ``stage12/crops/...``: the scaffold crop in the
  stage 1-2 manifest format, labelled ``markush``;
* ``expected.csv``  the four compounds the table defines (the answers).

Needs reportlab and poppler (pdftoppm). Usage, from the repository root:
    python -m stage3_recognition.examples.make_markush_sample
"""
from __future__ import annotations

import csv
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "sample_input" / "markush_sample"
DPI = 300
PAD_IN = 0.06                     # same padding stage 1-2 adds around its boxes
PAGE_H = letter[1]

# compound label -> (R1, R2, IC50) and the full compound it stands for
TABLE: List[Tuple[str, str, str, str]] = [
    ("5a", "H", "Cl", "120"),
    ("5b", "Me", "F", "85"),
    ("5c", "OMe", "Br", "40"),
    ("5d", "CF3", "OH", "310"),
]
EXPECTED: Dict[str, str] = {
    "5a": "Clc1ccccc1",
    "5b": "Cc1ccc(F)cc1",
    "5c": "COc1ccc(Br)cc1",
    "5d": "Oc1ccc(C(F)(F)F)cc1",
}
SCAFFOLD_TRUTH = "[1*]c1ccc([2*])cc1"


def _draw_scaffold(c: canvas.Canvas, cx: float, cy: float, r: float = 24) -> Tuple[float, ...]:
    """Benzene ring with R1 on top and R2 at the bottom; returns its box in points."""
    xs, ys = [], []

    def line(x1, y1, x2, y2):
        c.line(x1, y1, x2, y2)
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    v = [(cx + r * math.cos(math.radians(90 + 60 * k)), cy + r * math.sin(math.radians(90 + 60 * k)))
         for k in range(6)]
    c.setLineWidth(1.0)
    for k in range(6):
        (xa, ya), (xb, yb) = v[k], v[(k + 1) % 6]
        line(xa, ya, xb, yb)
        if k in (0, 2, 4):        # inner line of the alternating double bonds
            f = 0.78
            line(cx + (xa - cx) * f, cy + (ya - cy) * f, cx + (xb - cx) * f, cy + (yb - cy) * f)
    line(v[0][0], v[0][1], v[0][0], v[0][1] + r)          # bond to R1
    line(v[3][0], v[3][1], v[3][0], v[3][1] - r)          # bond to R2
    c.setFont("Helvetica", 11)
    c.drawString(v[0][0] - 6, v[0][1] + r + 2, "R1")
    c.drawString(v[3][0] - 6, v[3][1] - r - 11, "R2")
    xs.extend([v[0][0] + 8, v[0][0] - 6])
    ys.extend([v[0][1] + r + 12, v[3][1] - r - 13])
    return min(xs), min(ys), max(xs), max(ys)


def _draw_table(c: canvas.Canvas, x: float, y_top: float) -> None:
    """The R-group table as real text, one row per compound."""
    columns = (x, x + 50, x + 95, x + 140)
    c.setFont("Helvetica-Bold", 10)
    for col, text in zip(columns, ("Cmpd", "R1", "R2", "IC50 (nM)")):
        c.drawString(col, y_top, text)
    c.setFont("Helvetica", 10)
    for row, values in enumerate(TABLE, start=1):
        for col, text in zip(columns, values):
            c.drawString(col, y_top - 16 * row, text)


def build_pdf(pdf: Path) -> Tuple[float, ...]:
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.setFont("Helvetica", 10)
    c.drawString(72, 740, "Table 1. Substituent effects on kinase inhibition.")
    box = _draw_scaffold(c, 150, 620)
    _draw_table(c, 300, 660)
    c.showPage()
    c.save()
    return box


def render_crop(pdf: Path, box_pt: Tuple[float, ...], target: Path) -> Tuple[int, int, int, int]:
    """Render the page at DPI and cut out the scaffold, padded like stage 1-2 does."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "1", "-l", "1", "-png", "-singlefile",
                        str(pdf), str(prefix)], check=True)
        page = Image.open(f"{prefix}.png").convert("RGB")
        s, pad = DPI / 72.0, round(PAD_IN * DPI)
        x0, y0, x1, y1 = box_pt
        left, top = int(x0 * s) - pad, int((PAGE_H - y1) * s) - pad
        right, bottom = int(x1 * s) + pad, int((PAGE_H - y0) * s) + pad
        target.parent.mkdir(parents=True, exist_ok=True)
        page.crop((left, top, right, bottom)).save(target)
    return left, top, right - left, bottom - top


def build(out_dir: Path = DEFAULT_DIR) -> Path:
    """Write the sample and return the path of its stage 1-2 manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "markush_paper.pdf"
    box = build_pdf(pdf)
    stage12 = out_dir / "stage12"
    crop_rel = "crops/paper_markush_paper/p001_c01.png"
    x, y, w, h = render_crop(pdf, box, stage12 / crop_rel)

    manifest = stage12 / "manifest.csv"
    columns = ["paper_id", "crop_id", "crop_path", "source_pdf", "page", "x", "y", "width", "height",
               "dpi", "label", "gate_score", "reason", "markush_evidence", "detector"]
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({"paper_id": "markush_paper", "crop_id": "p001_c01", "crop_path": crop_rel,
                         "source_pdf": "../markush_paper.pdf", "page": 1, "x": x, "y": y, "width": w,
                         "height": h, "dpi": DPI, "label": "markush", "gate_score": 1.0,
                         "reason": "hand-made test case", "markush_evidence": "R1 R2",
                         "detector": "synthetic"})
    with open(out_dir / "expected.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["compound_label", "smiles"])
        writer.writerows(EXPECTED.items())
    return manifest


if __name__ == "__main__":
    print(f"Wrote {build()}")
