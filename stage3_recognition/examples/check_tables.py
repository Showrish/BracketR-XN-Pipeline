"""Run the R-group table reader (stage 3b) over every page of real PDFs.

Needs no recognition model, so it shows what 3b can read from a paper's text
layer on its own: which tables it finds, their rows, and how many cells it can
turn into a substituent structure.

    python -m stage3_recognition.examples.check_tables --pdf-dir path/to/course_pdfs
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from stage3_recognition.markush.rgroup_table import page_lines, parse_table_lines
from stage3_recognition.markush.substituents import cell_text, to_fragment


def page_count(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(next(line for line in info.splitlines() if line.startswith("Pages:")).split()[1])


def understood(cell: str) -> bool:
    return cell_text(cell) == "H" or to_fragment(cell) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", default="table_check.json")
    args = parser.parse_args()

    report = []
    for pdf in sorted(Path(args.pdf_dir).glob("*.pdf"), key=lambda p: (len(p.stem), p.stem)):
        for page in range(1, page_count(pdf) + 1):
            table = parse_table_lines(page_lines(pdf, page))
            if table is None:
                continue
            cells = [c for _, row in table.rows for c in row.values()]
            entry = {"paper": pdf.stem, "page": page, "columns": table.columns, "rows": len(table.rows),
                     "cells_understood": sum(understood(c) for c in cells), "cells": len(cells),
                     "table": [{"label": label, **row} for label, row in table.rows]}
            report.append(entry)
            print(f"paper {pdf.stem} page {page}: {entry['rows']} rows, columns {table.columns}, "
                  f"{entry['cells_understood']}/{entry['cells']} cells understood")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
