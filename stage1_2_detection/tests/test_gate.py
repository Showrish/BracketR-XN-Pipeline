import cv2
import numpy as np
import pytest

from stage1_2_detection.config import GateConfig
from stage1_2_detection.gate import (angle_statistics, classify_crop, dedupe_segments,
                                     markush_evidence)
from stage1_2_detection.render import render_pages

DPI = 300
CFG = GateConfig()


def crop_of(sample, kind, index=0):
    """Cut a ground-truth region out of the rendered page."""
    items = [t for t in sample["truth_px"] if t["kind"] == kind]
    item = items[index]
    page = dict(render_pages(sample["pdf"], DPI))[item["page"]]
    x, y, w, h = item["box"]
    pad = 18
    return page[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_molecules_pass(sample, index):
    result = classify_crop(crop_of(sample, "molecule", index), DPI, CFG)
    assert result.label == "molecule", result.reason


def test_markush_flagged_with_text_evidence(sample):
    crop = crop_of(sample, "markush")
    result = classify_crop(crop, DPI, CFG, words_near=["R1", "R2", "X"])
    assert result.label == "markush" and "R1" in result.markush_evidence


def test_markush_without_evidence_is_plain_molecule(sample):
    result = classify_crop(crop_of(sample, "markush"), DPI, CFG)
    assert result.label == "molecule"


@pytest.mark.parametrize("kind", ["chart", "table", "text"])
def test_non_molecules_rejected(sample, kind):
    result = classify_crop(crop_of(sample, kind), DPI, CFG)
    assert result.label == "not_molecule", (kind, result.reason)
    assert result.reason


def test_arrow_rejected(sample):
    result = classify_crop(crop_of(sample, "arrow"), DPI, CFG)
    assert result.label == "not_molecule"


def test_blank_and_tiny_crops_rejected():
    assert classify_crop(np.full((300, 300, 3), 255, np.uint8), DPI, CFG).label == "not_molecule"
    assert classify_crop(np.full((10, 10, 3), 255, np.uint8), DPI, CFG).reason == "too_small"


def test_photo_like_image_rejected():
    rng = np.random.default_rng(0)
    photo = rng.integers(60, 200, (400, 400, 3), dtype=np.uint8)
    assert classify_crop(photo, DPI, CFG).reason == "photo_or_filled_area"


def test_rectilinear_drawing_rejected():
    img = np.full((400, 400, 3), 255, np.uint8)
    for x in (80, 160, 240):                       # a few boxes: only 0/90 degree lines
        cv2.rectangle(img, (x, 100), (x + 60, 300), (0, 0, 0), 2)
    result = classify_crop(img, DPI, CFG)
    assert result.label == "not_molecule"
    assert result.reason == "no_diagonal_bond_angles"


def test_text_layer_prose_rejects_crop():
    img = np.full((300, 600, 3), 255, np.uint8)
    cv2.line(img, (50, 50), (150, 100), (0, 0, 0), 2)
    words = ["the", "enzyme", "reaction", "quenched", "analysed"]
    result = classify_crop(img, DPI, CFG, words_inside=words)
    assert result.reason.startswith("text_block_text_layer")


def test_markush_evidence_rules():
    assert markush_evidence(["R1", "OH"]) == ["R1"]
    assert markush_evidence(["R'", "NaOH"]) == ["R'"]
    assert markush_evidence(["X", "Y"]) == ["X", "Y"]       # two non-R tokens suffice
    assert markush_evidence(["X", "OH"]) == []               # a lone X is not enough
    assert markush_evidence(["Rate", "Run"]) == []


def test_angle_statistics_on_hexagon_like_segments():
    segs = [(0, 0, 100, 0), (100, 0, 150, 87), (150, 87, 100, 173)]   # 0, 60, 120 deg
    aligned, diagonal = angle_statistics(segs, tol=5)
    assert aligned == pytest.approx(1.0, abs=0.02)
    assert 0.5 < diagonal < 1.0


def test_dedupe_removes_overlapping_duplicates():
    segs = [(0, 0, 100, 0), (2, 1, 98, 1), (0, 50, 100, 50)]
    assert len(dedupe_segments(segs)) == 2


# --- look-alikes from other kinds of figures (drawn with OpenCV) --------------
def _blank(w=700, h=600):
    return np.full((h, w, 3), 255, np.uint8)


def _scatter():
    rng = np.random.default_rng(1)
    img = _blank()
    cv2.line(img, (60, 540), (60, 60), (0, 0, 0), 3)
    cv2.line(img, (60, 540), (650, 540), (0, 0, 0), 3)
    for _ in range(40):
        cv2.circle(img, (int(rng.integers(90, 630)), int(rng.integers(80, 520))), 7, (0, 0, 0), -1)
    return img


def _pie():
    img = _blank()
    cv2.circle(img, (350, 300), 200, (0, 0, 0), 3)
    for a in (10, 100, 170, 260):
        cv2.line(img, (350, 300), (350 + int(200 * np.cos(np.radians(a))),
                                   300 - int(200 * np.sin(np.radians(a)))), (0, 0, 0), 3)
    return img


def _flowchart():
    img = _blank()
    for x, y in [(60, 60), (400, 60), (230, 380)]:
        cv2.rectangle(img, (x, y), (x + 180, y + 100), (0, 0, 0), 3)
    cv2.arrowedLine(img, (240, 110), (400, 110), (0, 0, 0), 3)
    cv2.arrowedLine(img, (490, 160), (330, 380), (0, 0, 0), 3)
    return img


def _heatmap():
    rng = np.random.default_rng(1)
    img = _blank()
    for r in range(6):
        for c in range(6):
            color = tuple(int(v) for v in rng.integers(0, 255, 3))
            cv2.rectangle(img, (60 + c * 90, 40 + r * 90), (148 + c * 90, 128 + r * 90), color, -1)
    return img


def _equation():
    img = _blank(700, 200)
    cv2.putText(img, "y = a / (b + c) x 100", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


def _heading():
    img = _blank(900, 200)
    cv2.putText(img, "RESULTS", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
    return img


@pytest.mark.parametrize("make", [_scatter, _pie, _flowchart, _heatmap, _equation, _heading])
def test_other_figure_types_rejected(make):
    assert classify_crop(make(), DPI, CFG).label == "not_molecule"


@pytest.mark.xfail(strict=True, reason="Known limitation: a honeycomb of hexagons looks like "
                                       "a ring system to a geometry-only gate.")
def test_honeycomb_pattern_is_rejected():
    img = _blank()
    for r in range(3):
        for c in range(4):
            cx, cy = 100 + c * 140 + (70 if r % 2 else 0), 120 + r * 120
            pts = np.array([(cx + 60 * np.cos(np.radians(30 + 60 * k)),
                             cy + 60 * np.sin(np.radians(30 + 60 * k))) for k in range(6)], np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    assert classify_crop(img, DPI, CFG).label == "not_molecule"
