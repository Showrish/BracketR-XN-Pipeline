"""Stage 1a: turn PDF pages (or single images) into numpy arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def count_pages(pdf_path: str | Path) -> int:
    from pdf2image import pdfinfo_from_path

    return int(pdfinfo_from_path(str(pdf_path))["Pages"])


def render_pages(pdf_path: str | Path, dpi: int = 300,
                 first_page: Optional[int] = None,
                 last_page: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(page_number, RGB array)`` one page at a time (low memory).

    Image files are treated as a single-page document so the same code path
    works for BioVista-style cropped images.
    """
    path = Path(pdf_path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        yield 1, np.array(Image.open(path).convert("RGB"))
        return

    from pdf2image import convert_from_path

    total = count_pages(path)
    start = max(1, first_page or 1)
    stop = min(total, last_page or total)
    for page in range(start, stop + 1):
        images = convert_from_path(str(path), dpi=dpi, first_page=page, last_page=page)
        yield page, np.array(images[0].convert("RGB"))
