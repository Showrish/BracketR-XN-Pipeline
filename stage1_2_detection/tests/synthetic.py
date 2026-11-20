"""Synthetic two-page "paper" with known positions, used for tests and the sample run.

It mixes things a real paper contains: running text, drawn molecules, a
variable-group (Markush) drawing, a reaction arrow with conditions, a bar chart,
a line chart, and a table. The molecules are line art drawn with reportlab (no
chemistry library needed); they are NOT real compounds from the course papers.
Ground-truth boxes are returned in PDF points with the origin at the TOP-left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = letter           # 612 x 792 points
FONT = "Helvetica"
LOREM = ("The kinase inhibitor was dissolved in buffer and incubated with the enzyme "
         "for thirty minutes before the reaction was quenched and analysed by liquid "
         "chromatography. Results were averaged over three independent experiments and "
         "compared with the untreated control samples.").split()


@dataclass
class Truth:
    kind: str            # molecule | markush | text | chart | table | arrow
    page: int            # 1-based
    box: Tuple[float, float, float, float]   # x0, y0, x1, y1 in points, y from top


class Pen:
    """Small drawing helper that also records the bounding box of what it draws."""

    def __init__(self, c: canvas.Canvas):
        self.c = c
        self.reset()

    def reset(self):
        self.x0 = self.y0 = 1e9
        self.x1 = self.y1 = -1e9

    def _grow(self, x, y):
        self.x0, self.x1 = min(self.x0, x), max(self.x1, x)
        self.y0, self.y1 = min(self.y0, y), max(self.y1, y)

    def line(self, x1, y1, x2, y2, width=1.0):
        self.c.setLineWidth(width)
        self.c.line(x1, y1, x2, y2)
        self._grow(x1, y1)
        self._grow(x2, y2)

    def text(self, x, y, s, size=10):
        self.c.setFont(FONT, size)
        self.c.drawString(x, y, s)
        self._grow(x, y - 0.25 * size)
        self._grow(x + stringWidth(s, FONT, size), y + 0.8 * size)

    def box(self, page: int, kind: str) -> Truth:
        return Truth(kind, page, (self.x0, PAGE_H - self.y1, self.x1, PAGE_H - self.y0))


def _hex_vertices(cx, cy, r):
    return [(cx + r * math.cos(math.radians(90 + 60 * k)),
             cy + r * math.sin(math.radians(90 + 60 * k))) for k in range(6)]


def _ring(pen: Pen, cx, cy, r, double=(0, 2, 4)):
    v = _hex_vertices(cx, cy, r)
    for k in range(6):
        (xa, ya), (xb, yb) = v[k], v[(k + 1) % 6]
        pen.line(xa, ya, xb, yb)
        if k in double:                           # inner line of a double bond
            f = 0.78
            pen.line(cx + (xa - cx) * f, cy + (ya - cy) * f,
                     cx + (xb - cx) * f, cy + (yb - cy) * f)
    return v


def _bond(pen: Pen, p, angle_deg, length):
    q = (p[0] + length * math.cos(math.radians(angle_deg)),
         p[1] + length * math.sin(math.radians(angle_deg)))
    pen.line(p[0], p[1], q[0], q[1])
    return q


def draw_phenol(pen: Pen, cx, cy, r=24):
    v = _ring(pen, cx, cy, r)
    end = _bond(pen, v[3], 270, r)
    pen.text(end[0] - 8, end[1] - 11, "OH", 11)


def draw_naphthalene(pen: Pen, cx, cy, r=24):
    _ring(pen, cx, cy, r)
    _ring(pen, cx + math.sqrt(3) * r, cy, r, double=(1, 3, 5))


def draw_amide_chain(pen: Pen, x, y, b=26):
    """Zig-zag chain  CH3-CH2-C(=O)-NH2  (no ring)."""
    pts = [(x, y)]
    for i in range(1, 5):
        a = 30 if i % 2 else -30
        pts.append(_bond(pen, pts[-1], a, b))
    top = _bond(pen, pts[3], 90, b)
    pen.line(top[0] + 4, pts[3][1] + 4, top[0] + 4, top[1] - 1)   # second line of C=O
    pen.text(top[0] - 4, top[1] + 2, "O", 11)
    pen.text(pts[4][0] + 3, pts[4][1] - 4, "NH2", 11)


def draw_markush(pen: Pen, cx, cy, r=24):
    v = _ring(pen, cx, cy, r)
    top = _bond(pen, v[0], 90, r)
    pen.text(top[0] - 6, top[1] + 2, "R1", 11)
    right = _bond(pen, v[5], -30, r)
    pen.text(right[0] + 3, right[1] - 10, "R2", 11)
    left = _bond(pen, v[2], 210, r)
    pen.text(left[0] - 12, left[1] - 10, "X", 11)


def draw_paragraph(pen: Pen, x, y_top, lines=7, width=470):
    y, i = y_top, 0
    for _ in range(lines):
        line = []
        while stringWidth(" ".join(line + [LOREM[i % len(LOREM)]]), FONT, 10) < width:
            line.append(LOREM[i % len(LOREM)])
            i += 1
        pen.text(x, y, " ".join(line), 10)
        y -= 13


def draw_arrow(pen: Pen, x, y, length=70):
    pen.line(x, y, x + length, y, 1.2)
    pen.line(x + length, y, x + length - 7, y + 4, 1.2)
    pen.line(x + length, y, x + length - 7, y - 4, 1.2)
    pen.text(x + 4, y + 10, "NaOH, 80 C", 9)


def draw_bar_chart(pen: Pen, x, y, w=190, h=120):
    pen.line(x, y, x, y + h)
    pen.line(x, y, x + w, y)
    for i, height in enumerate((50, 85, 65, 100, 40)):
        left = x + 12 + i * 34
        pen.c.setFillGray(0.55)
        pen.c.rect(left, y, 22, height, stroke=0, fill=1)
        pen.c.setFillGray(0)
        pen._grow(left, y)
        pen._grow(left + 22, y + height)
        pen.text(left + 4, y - 12, f"S{i + 1}", 8)
    for t in (0, 50, 100):
        pen.text(x - 22, y + t - 3, str(t), 8)
    pen.text(x + 60, y - 28, "Sample", 9)


def draw_line_chart(pen: Pen, x, y, w=190, h=120):
    pen.line(x, y, x, y + h)
    pen.line(x, y, x + w, y)
    pts = [(x + 10, y + 20), (x + 50, y + 70), (x + 90, y + 45), (x + 130, y + 95),
           (x + 175, y + 60)]
    for a, b in zip(pts, pts[1:]):
        pen.line(a[0], a[1], b[0], b[1], 1.5)
    for t in (0, 50, 100):
        pen.text(x - 22, y + t - 3, str(t), 8)
    pen.text(x + 60, y - 22, "Time (h)", 9)


def draw_table(pen: Pen, x, y_top, cols=4, rows=5, cw=62, rh=16):
    for r in range(rows + 1):
        pen.line(x, y_top - r * rh, x + cols * cw, y_top - r * rh, 0.6)
    for c in range(cols + 1):
        pen.line(x + c * cw, y_top, x + c * cw, y_top - rows * rh, 0.6)
    for r in range(rows):
        for c in range(cols):
            pen.text(x + c * cw + 5, y_top - (r + 1) * rh + 5,
                     "Cmpd" if r == 0 else f"{(r * 3 + c) / 10:.2f}", 8)


def build_sample_pdf(path: str | Path) -> List[Truth]:
    """Write the two-page sample PDF and return its ground-truth boxes."""
    c = canvas.Canvas(str(path), pagesize=letter)
    pen = Pen(c)
    truth: List[Truth] = []

    def place(page, kind, fn, *args, **kw):
        pen.reset()
        fn(pen, *args, **kw)
        truth.append(pen.box(page, kind))

    # ---- page 1 ----------------------------------------------------------
    place(1, "text", draw_paragraph, 71, 720)
    place(1, "molecule", draw_phenol, 130, 560)
    place(1, "arrow", draw_arrow, 215, 560)
    place(1, "molecule", draw_naphthalene, 360, 560)
    place(1, "chart", draw_bar_chart, 100, 330)
    place(1, "molecule", draw_amide_chain, 380, 380)
    c.showPage()
    # ---- page 2 ----------------------------------------------------------
    place(2, "markush", draw_markush, 150, 640)
    place(2, "table", draw_table, 280, 700)
    place(2, "chart", draw_line_chart, 100, 380)
    place(2, "text", draw_paragraph, 71, 220, 4)
    c.showPage()
    c.save()
    return truth


def truth_to_pixels(truth: List[Truth], dpi: int) -> List[Dict]:
    s = dpi / 72.0
    return [{"kind": t.kind, "page": t.page,
             "box": tuple(int(round(v * s)) for v in
                          (t.box[0], t.box[1], t.box[2] - t.box[0], t.box[3] - t.box[1]))}
            for t in truth]


def iou(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def evaluate(rows, truth_px) -> Dict:
    """Score detection + gate against the known synthetic ground truth.

    * recall: share of true molecule/Markush drawings matched (IoU >= 0.5) by a
      region the gate accepted;
    * false_accepts: accepted regions that overlap a non-molecule item;
    * markush_ok: the Markush drawing carries the ``markush`` label.
    """
    accepted = [r for r in rows if r.label != "not_molecule" and r.x is not None]
    wanted = [t for t in truth_px if t["kind"] in ("molecule", "markush")]
    unwanted = [t for t in truth_px if t["kind"] not in ("molecule", "markush")]
    found, markush_ok = 0, True
    for t in wanted:
        hits = [r for r in accepted if r.page == t["page"]
                and iou((r.x, r.y, r.width, r.height), t["box"]) >= 0.5]
        found += bool(hits)
        if t["kind"] == "markush" and not any(h.label == "markush" for h in hits):
            markush_ok = False
    false_accepts = sum(
        1 for t in unwanted for r in accepted
        if r.page == t["page"] and iou((r.x, r.y, r.width, r.height), t["box"]) > 0.1)
    return {"true_molecules": len(wanted), "found": found,
            "recall": found / len(wanted) if wanted else 0.0,
            "non_molecule_items": len(unwanted), "false_accepts": false_accepts,
            "markush_ok": markush_ok}
