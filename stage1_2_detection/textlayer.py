"""Words from the PDF text layer, used as extra evidence by the gate.

Vector figures usually keep their atom labels (``OH``, ``R1``) as real text,
and running text is obviously text. Scanned or raster-only figures have no
text layer, in which case this module simply returns nothing.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

Word = Tuple[float, float, float, float, str]   # x0, y0, x1, y1 (pixels), text

_PAGE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)
_WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')


def parse_bbox_html(markup: str, dpi: int) -> Dict[int, List[Word]]:
    """Parse ``pdftotext -bbox`` output; coordinates are converted to pixels."""
    scale = dpi / 72.0
    pages: Dict[int, List[Word]] = {}
    for number, match in enumerate(_PAGE.finditer(markup), start=1):
        words = []
        for w in _WORD.finditer(match.group(3)):
            x0, y0, x1, y1 = (float(v) * scale for v in w.groups()[:4])
            words.append((x0, y0, x1, y1, html.unescape(w.group(5))))
        pages[number] = words
    return pages


def load_text_layer(pdf_path: str | Path, dpi: int) -> Dict[int, List[Word]]:
    """Return ``{page: [words]}`` or an empty dict if poppler is unavailable."""
    if shutil.which("pdftotext") is None or Path(pdf_path).suffix.lower() != ".pdf":
        return {}
    result = subprocess.run(
        ["pdftotext", "-bbox", str(pdf_path), "-"],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}
    return parse_bbox_html(result.stdout, dpi)


def words_in_box(words: List[Word], x: int, y: int, w: int, h: int,
                 pad: int = 0) -> List[str]:
    """Texts of words whose centre lies inside the (padded) box."""
    out = []
    for x0, y0, x1, y1, text in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if x - pad <= cx <= x + w + pad and y - pad <= cy <= y + h + pad:
            out.append(text)
    return out


_LABEL_MAX_CHARS = 4
_PROSE = re.compile(r"^[a-z]{4,}[.,;:)]?$")


def attach_label_words(box: Tuple[int, int, int, int], words: List[Word],
                       gap_px: int) -> Tuple[int, int, int, int]:
    """Grow a box to include short text-layer tokens (atom labels) just outside it.

    Atom labels such as ``X`` or ``OH`` are often drawn slightly apart from the
    bond they belong to, so pixel-based merging can leave them out of the crop.
    Long or prose-like words are ignored so running text is not pulled in.
    """
    x, y, w, h = box
    x1, y1 = x + w, y + h
    changed = True
    while changed:
        changed = False
        for wx0, wy0, wx1, wy1, text in words:
            if len(text) > _LABEL_MAX_CHARS or _PROSE.match(text):
                continue
            touches = (wx0 < x1 + gap_px and wx1 > x - gap_px
                       and wy0 < y1 + gap_px and wy1 > y - gap_px)
            inside = wx0 >= x and wx1 <= x1 and wy0 >= y and wy1 <= y1
            if touches and not inside:
                x, y = min(x, int(wx0)), min(y, int(wy0))
                x1, y1 = max(x1, int(wx1) + 1), max(y1, int(wy1) + 1)
                changed = True
    return (x, y, x1 - x, y1 - y)
