"""Stage 1c: split a merged panel into separate drawings.

Pixel merging at the normal gap (0.07 in) can glue a molecule to a neighbouring
chart or panel, and the gate may still accept the combined crop because the
molecules dominate it. For such suspicious regions this module:

1. re-detects blobs at a much finer gap (0.02 in),
2. attaches small pieces (atom labels, specks) to the nearest large blob,
3. gates every piece separately,
4. re-joins pieces of the same molecule: weak rejected fragments (too few bond
   lines, low score) next to an accepted piece are absorbed, and accepted
   pieces that almost touch are merged.

5. reject accepted pieces that sit inside a rejected piece (a data line inside
   chart axes).

Charts, text and tables stay rejected; molecules come out as separate crops.
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectConfig
from .detect import Box, binarize, to_gray
from .gate import GateResult
from .manifest import ACCEPTED_LABELS

Part = Tuple[Box, GateResult]
Classifier = Callable[[Box], GateResult]      # box in PAGE coordinates -> result

# Rejection reasons that mean "not enough drawing here", i.e. possibly a fragment
# of a molecule rather than a different kind of content.
WEAK_REASONS = ("too_few_bond_lines", "low_score", "no_diagonal_bond_angles", "too_small")


def box_distance(a: Box, b: Box) -> float:
    """Gap in pixels between two boxes (0 if they overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return math.hypot(dx, dy)


def union_box(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def fine_boxes(crop_rgb: np.ndarray, dpi: int, gap_in: float) -> List[Box]:
    """Boxes of ink blobs at a fine merge gap (relative to the crop)."""
    ink = binarize(to_gray(crop_rgb))
    k = max(1, int(round(gap_in * dpi)))
    dilated = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    return [(int(s[0]), int(s[1]), int(s[2]), int(s[3]))
            for s in stats[1:] if s[4] >= 4]


def attach_small(boxes: Sequence[Box], min_side_px: int, attach_px: int) -> List[Box]:
    """Merge small blobs (labels, specks) into the nearest large blob within reach."""
    large = [b for b in boxes if max(b[2], b[3]) >= min_side_px]
    small = [b for b in boxes if max(b[2], b[3]) < min_side_px]
    for piece in small:
        if not large:
            break
        index = min(range(len(large)), key=lambda i: box_distance(piece, large[i]))
        if box_distance(piece, large[index]) <= attach_px:
            large[index] = union_box(large[index], piece)
    return large


def ink_close(page_rgb: np.ndarray, a: Box, b: Box, px: int) -> bool:
    """True if some ink of box ``a`` lies within ``px`` pixels of ink of box ``b``.

    Bounding boxes of neighbouring molecules often touch even when their
    drawings do not, so closeness is measured on the ink itself.
    """
    if box_distance(a, b) > px:
        return False
    x0, y0, w, h = union_box(a, b)
    ink = binarize(to_gray(page_rgb[y0:y0 + h, x0:x0 + w])) > 0
    mask_a = np.zeros_like(ink)
    mask_b = np.zeros_like(ink)
    mask_a[a[1] - y0:a[1] - y0 + a[3], a[0] - x0:a[0] - x0 + a[2]] = True
    mask_b[b[1] - y0:b[1] - y0 + b[3], b[0] - x0:b[0] - x0 + b[2]] = True
    only_a = ink & mask_a & ~mask_b
    only_b = ink & mask_b & ~mask_a
    if not only_a.any() or not only_b.any():
        return True                          # one box lies inside the other
    k = 2 * px + 1
    grown = cv2.dilate(only_a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return bool((grown & only_b).any())


def _is_weak(result: GateResult) -> bool:
    return result.reason.startswith(WEAK_REASONS)


def _rejoin(parts: List[Part], classify: Classifier, join_px: int,
            page_rgb: np.ndarray) -> List[Part]:
    """Absorb weak fragments into, and merge, accepted parts that are close together."""
    changed = True
    while changed:
        changed = False
        accepted = [p for p in parts if p[1].label in ACCEPTED_LABELS]
        for i, (box_a, _) in enumerate(accepted):
            for other in parts:
                if other is accepted[i]:
                    continue
                box_b, res_b = other
                joinable = res_b.label in ACCEPTED_LABELS or _is_weak(res_b)
                if joinable and ink_close(page_rgb, box_a, box_b, join_px):
                    merged = union_box(box_a, box_b)
                    result = classify(merged)
                    if result.label in ACCEPTED_LABELS:
                        parts = [p for p in parts if p is not accepted[i] and p is not other]
                        parts.append((merged, result))
                        changed = True
                        break
            if changed:
                break
    return parts


def _inside(inner: Box, outer: Box, share: float = 0.8) -> bool:
    ix = max(0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    return ix * iy >= share * inner[2] * inner[3] and outer[2] * outer[3] > inner[2] * inner[3]


def _drop_enclosed(parts: List[Part]) -> List[Part]:
    """An accepted piece sitting inside a rejected piece (e.g. a data line inside
    chart axes) belongs to that figure, not to a molecule."""
    out: List[Part] = []
    for box, result in parts:
        if result.label in ACCEPTED_LABELS and any(
                r.label not in ACCEPTED_LABELS and _inside(box, b) for b, r in parts):
            result = GateResult("not_molecule", result.score, "inside_other_figure",
                                "", result.features)
        out.append((box, result))
    return out


def split_region(page_rgb: np.ndarray, box: Box, dpi: int, cfg: DetectConfig,
                 classify: Classifier) -> List[Part]:
    """Return ``[(page_box, GateResult), ...]`` for the pieces of a suspicious region."""
    x0, y0, w, h = box
    crop = page_rgb[y0:y0 + h, x0:x0 + w]
    min_side = max(1, int(round(cfg.min_side_in * dpi)))
    attach_px = max(1, int(round(cfg.label_attach_in * dpi)))
    pieces = attach_small(fine_boxes(crop, dpi, cfg.split_gap_in), min_side, attach_px)

    tiny = max(1, int(round(0.10 * dpi)))
    parts: List[Part] = []
    for bx, by, bw, bh in pieces:
        page_box = (x0 + bx, y0 + by, bw, bh)
        if min(bw, bh) < tiny:              # a thin rule or a short line of text
            parts.append((page_box, GateResult("not_molecule", 0.0, "too_small")))
        else:
            parts.append((page_box, classify(page_box)))
    join_px = max(1, int(round(cfg.join_gap_in * dpi)))
    return _drop_enclosed(_rejoin(parts, classify, join_px, page_rgb))
