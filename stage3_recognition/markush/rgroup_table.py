"""Find and read the R-group table that defines a Markush scaffold's compounds.

Two sources:
* the PDF text layer (``pdftotext -layout``) of the scaffold's page, and the next
  page if needed, for papers whose tables are real text;
* a CSV file (compound_label, R1, R2, ...), for tables that are images or that
  were checked by hand.

A header line is any line with R-group column names (R1, R2, R', X, Ar ...).
Rows below it that start with a compound label (5a, 12, 10b ...) are read until
the table ends. Words before the first R column are treated as the label column,
and a header that repeats its label word ("Compound R ... Compound R ...") is
read as two tables printed side by side. Rows need plausible cells (short, no
links) and a table needs at least two rows, so running text is not mistaken for one.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .substituents import clean_token

R_COLUMN = re.compile(r"^(R\d*'{0,2}|R[a-z]|X\d*|Y\d*|Z\d*|Ar\d*)$")
COMPOUND_LABEL = re.compile(r"^\d{1,3}[a-z]{0,2}$")
LABEL_HEADERS = {"compound", "compd", "compd.", "cmpd", "cmpd.", "entry", "no", "no.", "#", "id"}
LABEL_NOTE = re.compile(r"^\([^()]*\)$")       # e.g. "(VMF)" written after a compound number
MIN_ROWS = 2                                     # one row is too easily a coincidence in running text
MAX_CELL_CHARS = 25

Row = Tuple[str, Dict[str, str]]   # (compound label, {column: cell text})


@dataclass
class RGroupTable:
    columns: List[str]             # R-group columns in table order, e.g. ["R1", "R2"]
    rows: List[Row]
    source: str                    # text_layer | csv


@dataclass
class _Segment:
    """One block of a header. Journals often print a table twice side by side
    ("Compound R ... Compound R ..."), which gives one segment per half."""
    width: int                                  # header tokens in the block
    r_positions: List[Tuple[int, str]]          # (position in the block, column name)
    label_words: int                            # header words naming the label column ("Compd no." = 2)

    def _index(self, position: int) -> int:
        # a row has one label cell however many words its header used
        return position - self.label_words + 1

    def cells(self, chunk: List[str]) -> Dict[str, str]:
        return {name: chunk[self._index(pos)] for pos, name in self.r_positions}

    @property
    def needed(self) -> int:
        return self._index(self.r_positions[-1][0]) + 1

    @property
    def step(self) -> int:
        return self._index(self.width)


def _header_segments(tokens: List[str]) -> List[_Segment]:
    """Split a header at each label-column word (Compound, Entry ...)."""
    starts = [i for i, tok in enumerate(tokens)
              if tok.lower() in LABEL_HEADERS and (i == 0 or tokens[i - 1].lower() not in LABEL_HEADERS)]
    bounds = (starts or [0]) + [len(tokens)]
    segments = []
    for a, b in zip(bounds, bounds[1:]):
        block = tokens[a:b]
        r_positions = [(i, tok) for i, tok in enumerate(block) if R_COLUMN.match(tok)]
        label_words = 0
        while label_words < len(block) and block[label_words].lower() in LABEL_HEADERS:
            label_words += 1
        if r_positions:
            segments.append(_Segment(b - a, r_positions, label_words))
    return segments


def _plausible(cells: Dict[str, str]) -> bool:
    return all(cell and len(cell) <= MAX_CELL_CHARS and "http" not in cell and "doi" not in cell
               for cell in cells.values())


def _row_chunks(tokens: List[str], segments: List[_Segment]) -> List[Tuple[str, Dict[str, str]]]:
    """Read one line as one row per header segment."""
    tokens = [tok for i, tok in enumerate(tokens)
              if not (i > 0 and LABEL_NOTE.match(tok) and COMPOUND_LABEL.match(tokens[i - 1]))]
    rows, position = [], 0
    for segment in segments:
        chunk = tokens[position:position + segment.step]
        position += segment.step
        if len(chunk) >= segment.needed and COMPOUND_LABEL.match(chunk[0]):
            cells = segment.cells(chunk)
            if _plausible(cells):
                rows.append((chunk[0], cells))
    return rows


def _read_rows(lines: Sequence[str], segments: List[_Segment]) -> List[Row]:
    rows: List[Row] = []
    for line in lines:
        tokens = [clean_token(c) for c in line.split()]
        if not tokens:
            continue                              # blank lines inside a table are common
        found = _row_chunks(tokens, segments)
        if not found:
            if rows:
                break                             # first non-row line after the rows ends the table
            continue
        rows.extend(found)
    return rows


def parse_table_lines(lines: Sequence[str], source: str = "text_layer") -> Optional[RGroupTable]:
    """Return the largest R-group table found in the lines, or None."""
    best: Optional[RGroupTable] = None
    for index, line in enumerate(lines):
        segments = _header_segments([clean_token(t) for t in line.split()])
        if not segments:
            continue
        rows = _read_rows(lines[index + 1:], segments)
        if len(rows) < MIN_ROWS:
            continue                              # e.g. an "R1" label next to a drawing
        columns = list(dict.fromkeys(name for seg in segments for _, name in seg.r_positions))
        table = RGroupTable(columns, rows, source)
        if best is None or (len(table.rows), len(table.columns)) > (len(best.rows), len(best.columns)):
            best = table
    return best


def page_lines(pdf: Path, page: int) -> List[str]:
    """Text of one page with its layout kept, or [] if poppler is not installed."""
    if shutil.which("pdftotext") is None:
        return []
    result = subprocess.run(["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
                            capture_output=True, text=True, check=False)
    return result.stdout.splitlines() if result.returncode == 0 else []


def table_from_pdf_text(pdf: Path, page: int, search_next_page: bool = True) -> Optional[RGroupTable]:
    """Look for an R-group table on the scaffold's page, then on the next page."""
    for number in ([page, page + 1] if search_next_page else [page]):
        table = parse_table_lines(page_lines(pdf, number), source="text_layer")
        if table is not None:
            return table
    return None


def table_from_csv(path: Path, paper_id: str, crop_id: str) -> Optional[RGroupTable]:
    """Rows of a CSV table for this crop. Optional paper_id and crop_id columns
    restrict a row to one scaffold, so one file can hold tables for many papers."""
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in (reader.fieldnames or []) if R_COLUMN.match(c)]
        rows: List[Row] = []
        for rec in reader:
            if rec.get("paper_id") and rec["paper_id"] != paper_id:
                continue
            if rec.get("crop_id") and rec["crop_id"] != crop_id:
                continue
            rows.append((rec["compound_label"], {c: rec.get(c, "") for c in columns}))
    return RGroupTable(columns, rows, "csv") if columns and rows else None
