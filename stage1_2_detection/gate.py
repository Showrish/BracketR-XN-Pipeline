"""Stage 2: the molecule gate.

Decides, for each candidate crop, whether it is a drawn molecule, a
variable-group (Markush) drawing, or something else (text, chart, table,
photo, arrow, ...). Decisions are rule-based and transparent on purpose:
every rejection carries a short reason code that is saved in the manifest, so
mistakes can be inspected and thresholds tuned (see ``config.GateConfig``).

Evidence used
-------------
* Stroke geometry: chemical bonds are straight strokes of similar length at
  multiples of 30 degrees. Letters are too short, charts are rectilinear or
  use arbitrary angles, photographs have no clean strokes.
* Rings: closed 5-6 sided holes (benzene-like rings) are strong evidence.
* Text rows: many small components lined up in rows (and few strokes) means
  running text; many letters per stroke means a labelled chart or diagram.
* Colour: drawn molecules are black; coloured content is a plot or diagram.
* PDF text layer (when present): prose words reject a crop, ``R1``/``R2``
  style tokens near the drawing flag it as Markush.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .config import GateConfig
from .detect import binarize, to_gray
from .manifest import LABEL_MARKUSH, LABEL_MOLECULE, LABEL_REJECTED

Segment = Tuple[int, int, int, int]

_PROSE_WORD = re.compile(r"^[a-z]{4,}[.,;:)]?$")
_R_TOKEN = re.compile(r"^R(\d{1,2}|[a-d]|['’′])?[,;:=]?$")
_OTHER_TOKEN = re.compile(r"^(X|Y|Z|Ar|Het|Alk)\d?[,;:=]?$")


@dataclass
class GateResult:
    label: str
    score: float
    reason: str
    markush_evidence: str = ""
    features: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Stroke analysis
# --------------------------------------------------------------------------
def _angle_deg(seg: Segment) -> float:
    x1, y1, x2, y2 = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _length(seg: Segment) -> float:
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def _point_line_distance(px: float, py: float, seg: Segment) -> Tuple[float, float]:
    """Return (perpendicular distance, position along the segment in px)."""
    x1, y1, x2, y2 = seg
    length = _length(seg) or 1.0
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    along = (px - x1) * ux + (py - y1) * uy
    perp = abs(-(px - x1) * uy + (py - y1) * ux)
    return perp, along


def dedupe_segments(segments: Sequence[Segment], dist_px: float = 4.0,
                    angle_tol: float = 7.0) -> List[Segment]:
    """Drop Hough duplicates: near-parallel segments lying on an accepted one."""
    kept: List[Segment] = []
    for seg in sorted(segments, key=_length, reverse=True):
        duplicate = False
        for other in kept:
            diff = abs(_angle_deg(seg) - _angle_deg(other))
            diff = min(diff, 180.0 - diff)
            if diff > angle_tol:
                continue
            length = _length(other)
            ends = [_point_line_distance(seg[0], seg[1], other),
                    _point_line_distance(seg[2], seg[3], other)]
            if all(p <= dist_px and -dist_px <= a <= length + dist_px for p, a in ends):
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
    return kept


def extract_segments(ink: np.ndarray, dpi: int, cfg: GateConfig) -> List[Segment]:
    """Straight strokes (>= min_bond_len_in) found on the skeleton of the ink."""
    min_len = max(8, int(round(cfg.min_bond_len_in * dpi)))
    skeleton = skeletonize(ink > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=max(10, min_len // 2),
                            minLineLength=min_len, maxLineGap=max(2, dpi // 50))
    if lines is None:
        return []
    # OpenCV returns (N, 1, 4) in 4.x but (N, 4) in 5.x; reshape handles both.
    raw = [tuple(int(v) for v in line) for line in np.asarray(lines).reshape(-1, 4)]
    return dedupe_segments(raw)


def angle_statistics(segments: Sequence[Segment], tol: float) -> Tuple[float, float]:
    """(share of stroke length aligned to a multiple of 30 deg,
        share of stroke length at 30/60/120/150 deg i.e. non-rectilinear bonds)."""
    total = sum(_length(s) for s in segments)
    if total == 0:
        return 0.0, 0.0
    aligned = diagonal = 0.0
    for seg in segments:
        angle = _angle_deg(seg)
        nearest = round(angle / 30.0) * 30.0
        if abs(angle - nearest) <= tol:
            aligned += _length(seg)
            if round(nearest) % 90 != 0:
                diagonal += _length(seg)
    return aligned / total, diagonal / total


def curved_joint_fraction(segments: Sequence[Segment], join_px: float = 4.0) -> Tuple[float, int]:
    """Share of segment joints that bend by only 8-40 degrees.

    Bonds meet at about 60/120 degrees; a circle or curve drawn as a chain of
    short straight pieces bends a little at every joint.
    """
    ends = [((s[0], s[1]), (s[2], s[3])) for s in segments]
    joints = shallow = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if not any(math.dist(p, q) <= join_px for p in ends[i] for q in ends[j]):
                continue
            diff = abs(_angle_deg(segments[i]) - _angle_deg(segments[j]))
            diff = min(diff, 180.0 - diff)
            if diff < 3.0:                     # collinear pieces of one stroke
                continue
            joints += 1
            shallow += 8.0 <= diff <= 40.0
    return (shallow / joints if joints else 0.0), joints


def has_long_axes(segments: Sequence[Segment], width: int, height: int,
                  fraction: float, tol: float = 5.0) -> Tuple[bool, bool]:
    """Is there a near-horizontal / near-vertical line spanning most of the crop?"""
    horizontal = vertical = False
    for seg in segments:
        angle, length = _angle_deg(seg), _length(seg)
        if min(angle, 180.0 - angle) <= tol and length >= fraction * width:
            horizontal = True
        if abs(angle - 90.0) <= tol and length >= fraction * height:
            vertical = True
    return horizontal, vertical


def count_rings(ink: np.ndarray, dpi: int, cfg: GateConfig) -> int:
    """Count closed, convex 5-7 sided holes of ring-like size and roundness."""
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    rings = 0
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] == -1:                       # not a hole
            continue
        area_in2 = cv2.contourArea(contour) / float(dpi * dpi)
        if not cfg.ring_min_area_in2 <= area_in2 <= cfg.ring_max_area_in2:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        circularity = 4 * math.pi * cv2.contourArea(contour) / (perimeter ** 2)
        if 5 <= len(approx) <= 7 and cv2.isContourConvex(approx) and circularity > 0.75:
            rings += 1
    return rings


def text_row_fraction(ink: np.ndarray, dpi: int) -> Tuple[float, int]:
    """Share of components that look like letters lined up in rows, and their count."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    letters = []
    total = 0
    for i in range(1, count):
        w, h, area = stats[i][2], stats[i][3], stats[i][4]
        if area < 4:                            # specks
            continue
        total += 1
        if 0.03 <= h / dpi <= 0.30 and w / dpi <= 0.40:
            letters.append((float(centroids[i][1]), float(h)))
    if total == 0 or not letters:
        return 0.0, 0
    letters.sort()
    median_h = float(np.median([h for _, h in letters]))
    rows: List[List[float]] = [[letters[0][0]]]
    for cy, _ in letters[1:]:
        if cy - rows[-1][-1] <= 0.6 * median_h:
            rows[-1].append(cy)
        else:
            rows.append([cy])
    in_rows = sum(len(r) for r in rows if len(r) >= 4)
    return in_rows / total, in_rows


def midtone_fraction(gray: np.ndarray) -> float:
    return float(np.mean((gray > 40) & (gray < 215)))


def color_fraction(crop_rgb: np.ndarray, gray: np.ndarray) -> float:
    """Share of non-white pixels that are clearly coloured (not black/grey)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2HSV)
    coloured = (hsv[..., 1] > 80) & (hsv[..., 2] > 60)
    non_white = gray < 235
    return float((coloured & non_white).sum() / max(1, int(non_white.sum())))


# --------------------------------------------------------------------------
# Markush evidence
# --------------------------------------------------------------------------
def ocr_words(crop_rgb: np.ndarray) -> List[str]:
    """Optional OCR fallback for figures without a text layer."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(crop_rgb, config="--psm 11")
    except Exception:                           # missing package/binary, bad image
        return []
    return text.split()


def markush_evidence(words: Sequence[str]) -> List[str]:
    """R-group style tokens (R1, R', X, Ar, ...). One R token or two others suffice."""
    r_tokens = [w for w in words if _R_TOKEN.match(w)]
    other = [w for w in words if _OTHER_TOKEN.match(w)]
    if r_tokens:
        return sorted(set(r_tokens + other))
    return sorted(set(other)) if len(other) >= 2 else []


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------
def compute_features(crop_rgb: np.ndarray, dpi: int, words_inside: Sequence[str],
                     cfg: GateConfig) -> Dict[str, float]:
    gray = to_gray(crop_rgb)
    ink = binarize(gray)
    height, width = gray.shape
    segments = extract_segments(ink, dpi, cfg)
    angle_frac, diagonal_frac = angle_statistics(segments, cfg.angle_tolerance_deg)
    h_axis, v_axis = has_long_axes(segments, width, height, cfg.axis_line_fraction)
    text_frac, text_cc = text_row_fraction(ink, dpi)
    curved, joints = curved_joint_fraction(segments)
    return {
        "width_in": width / dpi, "height_in": height / dpi,
        "ink_ratio": float(np.mean(ink > 0)),
        "midtone_fraction": midtone_fraction(gray),
        "color_fraction": color_fraction(crop_rgb, gray),
        "n_segments": len(segments),
        "angle_fraction": angle_frac, "diagonal_fraction": diagonal_frac,
        "n_rings": count_rings(ink, dpi, cfg),
        "curved_fraction": curved, "n_joints": joints,
        "text_cc_fraction": text_frac, "text_cc_count": text_cc,
        "h_axis": float(h_axis), "v_axis": float(v_axis),
        "prose_words": sum(1 for w in words_inside if _PROSE_WORD.match(w)),
    }


def molecule_score(f: Dict[str, float], cfg: GateConfig) -> float:
    score = (cfg.w_segments * min(1.0, f["n_segments"] / 8.0)
             + cfg.w_angle * f["angle_fraction"]
             + cfg.w_diagonal * min(1.0, f["diagonal_fraction"] / 0.30)
             + cfg.w_ring * (1.0 if f["n_rings"] >= 1 else 0.0)
             - cfg.text_penalty * f["text_cc_fraction"])
    return float(min(1.0, max(0.0, score)))


def decide(f: Dict[str, float], cfg: GateConfig) -> Tuple[bool, float, str]:
    """Return (accepted, score, reason). Hard rules first, then the score."""
    score = molecule_score(f, cfg)
    if f["prose_words"] >= cfg.text_layer_min_words:
        return False, score, f"text_block_text_layer(words={int(f['prose_words'])})"
    if f["midtone_fraction"] > cfg.max_midtone_fraction or f["ink_ratio"] > cfg.max_ink_ratio:
        return False, score, "photo_or_filled_area"
    if f["color_fraction"] > cfg.max_color_fraction:
        return False, score, f"colored_figure({f['color_fraction']:.2f})"
    if (f["text_cc_fraction"] >= cfg.max_text_cc_fraction and f["text_cc_count"] >= 12
            and f["text_cc_count"] > cfg.max_letters_per_bond * f["n_segments"]):
        return False, score, f"text_block_rows(letters={int(f['text_cc_count'])})"
    if f["h_axis"] and f["v_axis"]:
        return False, score, "chart_or_table_axes"
    if (f["text_cc_count"] >= cfg.label_heavy_min_letters
            and f["text_cc_count"] > cfg.max_letters_per_bond * f["n_segments"]):
        return False, score, (f"label_heavy_figure(letters={int(f['text_cc_count'])}, "
                              f"strokes={int(f['n_segments'])})")
    if f["n_joints"] >= 6 and f["curved_fraction"] >= cfg.max_curved_fraction:
        return False, score, f"curved_lines({f['curved_fraction']:.2f})"
    if f["n_segments"] < cfg.min_bond_segments:
        return False, score, f"too_few_bond_lines(n={int(f['n_segments'])})"
    if f["n_rings"] == 0 and f["diagonal_fraction"] < cfg.min_diagonal_fraction:
        return False, score, "no_diagonal_bond_angles"
    if score < cfg.min_score:
        return False, score, f"low_score({score:.2f})"
    return True, score, (f"ok(bond_lines={int(f['n_segments'])}, "
                         f"rings={int(f['n_rings'])})")


def classify_crop(crop_rgb: np.ndarray, dpi: int, cfg: GateConfig,
                  words_inside: Sequence[str] = (),
                  words_near: Sequence[str] = ()) -> GateResult:
    """Classify one crop as molecule / markush / not_molecule."""
    height, width = crop_rgb.shape[:2]
    if min(width, height) / dpi < 0.10:
        return GateResult(LABEL_REJECTED, 0.0, "too_small")
    features = compute_features(crop_rgb, dpi, words_inside, cfg)
    accepted, score, reason = decide(features, cfg)
    if not accepted:
        return GateResult(LABEL_REJECTED, score, reason, "", features)

    nearby = list(words_near)
    if not nearby and cfg.use_ocr_fallback:
        nearby = ocr_words(crop_rgb)
    evidence = markush_evidence(nearby)
    label = LABEL_MARKUSH if evidence else LABEL_MOLECULE
    return GateResult(label, score, reason, " ".join(evidence), features)
