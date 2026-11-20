"""Stage 1b: propose candidate regions on a rendered page.

Two interchangeable backends return the same thing, a list of boxes
``(x, y, w, h)`` in page pixels:

* ``classical`` (default): Otsu binarisation + morphological dilation +
  connected components. No model download, runs anywhere.
* ``decimer``: DECIMER Segmentation (pretrained Mask R-CNN). It returns
  cropped images only, so each crop is located on the page by template
  matching to recover its position.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectConfig

Box = Tuple[int, int, int, int]  # x, y, w, h


def to_gray(page_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255, background = 0 (Otsu threshold)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _inches(value_in: float, dpi: int) -> int:
    return max(1, int(round(value_in * dpi)))


def _overlap(a: Box, b: Box) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def merge_overlapping(boxes: Sequence[Box]) -> List[Box]:
    """Repeatedly merge boxes that overlap until none do."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        merged: List[Box] = []
        for box in boxes:
            for i, other in enumerate(merged):
                if _overlap(box, other):
                    merged[i] = _union(box, other)
                    changed = True
                    break
            else:
                merged.append(box)
        boxes = merged
    return boxes


def detect_classical(page_rgb: np.ndarray, dpi: int, cfg: DetectConfig) -> List[Box]:
    """Group nearby ink into blobs and return padded bounding boxes."""
    gray = to_gray(page_rgb)
    height, width = gray.shape
    ink = binarize(gray)

    gap = _inches(cfg.merge_gap_in, dpi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    blobs = cv2.dilate(ink, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)

    min_side = _inches(cfg.min_side_in, dpi)
    pad = _inches(cfg.pad_in, dpi)
    page_area = width * height
    boxes: List[Box] = []
    for i in range(1, count):                       # 0 is the background
        x, y, w, h, _area = stats[i]
        if w < min_side or h < min_side:
            continue
        if w * h > cfg.max_page_fraction * page_area:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return merge_overlapping(boxes)


def locate_crop(page_gray: np.ndarray, crop_rgb: np.ndarray,
                threshold: float) -> Optional[Box]:
    """Find where a crop sits on the page (template matching), or None."""
    crop_gray = to_gray(crop_rgb)
    h, w = crop_gray.shape
    if h > page_gray.shape[0] or w > page_gray.shape[1]:
        return None
    result = cv2.matchTemplate(page_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(result)
    if best < threshold:
        return None
    return (int(loc[0]), int(loc[1]), int(w), int(h))


def detect_decimer(page_rgb: np.ndarray, cfg: DetectConfig,
                   segmenter: Optional[Callable] = None
                   ) -> List[Tuple[Optional[Box], np.ndarray]]:
    """Run DECIMER Segmentation; returns ``[(box or None, crop_rgb), ...]``.

    ``segmenter`` can be injected (used by the tests); by default the real
    ``decimer_segmentation.segment_chemical_structures`` is imported.
    """
    if segmenter is None:
        from decimer_segmentation import segment_chemical_structures as segmenter
    crops = segmenter(page_rgb, expand=cfg.decimer_expand)
    gray = to_gray(page_rgb)
    out = []
    for crop in crops:
        crop = np.asarray(crop)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        out.append((locate_crop(gray, crop, cfg.decimer_match_threshold), crop))
    return out
