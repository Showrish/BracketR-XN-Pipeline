"""Splitting of merged panels (a molecule glued to a chart), drawn with OpenCV at 300 dpi."""
import math

import cv2
import numpy as np
from PIL import Image

from stage1_2_detection import load_config, run_stage1_2
from stage1_2_detection.gate import classify_crop
from stage1_2_detection.split import ink_close, split_region

DPI = 300
BLACK = (0, 0, 0)


def draw_ring_system(img, x, y, bond=30, rings=3, label_gap=8):
    """Fused hexagons in a row plus an 'OH' label drawn slightly apart from its bond."""
    cy = y
    for k in range(rings):
        cx = x + int(k * math.sqrt(3) * bond)
        pts = [(int(cx + bond * math.cos(math.radians(90 + 60 * i))),
                int(cy + bond * math.sin(math.radians(90 + 60 * i)))) for i in range(6)]
        cv2.polylines(img, [np.array(pts, np.int32)], True, BLACK, 2, cv2.LINE_AA)
    tail = (x, cy + bond)
    end = (tail[0], tail[1] + bond)
    cv2.line(img, tail, end, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, "OH", (end[0] - 12, end[1] + label_gap + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)


def draw_labelled_scatter(img, x, y, w=420, h=320):
    rng = np.random.default_rng(3)
    cv2.line(img, (x, y), (x, y + h), BLACK, 2)
    cv2.line(img, (x, y + h), (x + w, y + h), BLACK, 2)
    for t in range(5):
        cv2.putText(img, str(50 + 10 * t), (x - 40, y + h - t * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    for i in range(14):
        px, py = int(x + 40 + rng.integers(0, w - 120)), int(y + 20 + rng.integers(0, h - 60))
        cv2.circle(img, (px, py), 6, (230, 90, 90), -1)
        cv2.putText(img, f"K{i}", (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 90, 90), 1, cv2.LINE_AA)


def merged_page():
    """A page where a molecule sits 0.04 in from a labelled scatter plot."""
    page = np.full((3300, 2550, 3), 255, np.uint8)
    draw_ring_system(page, 400, 800)
    # ring system ends near x = 400 + 2*52 + 30 = 534; the plot axis starts 12 px later
    draw_labelled_scatter(page, 546, 600)
    return page


def test_merged_panel_is_split(tmp_path):
    path = tmp_path / "merged.png"
    Image.fromarray(merged_page()).save(path)
    rows = run_stage1_2(path, tmp_path / "out", load_config())
    accepted = [r for r in rows if r.label != "not_molecule"]
    assert len(accepted) == 1, [(r.reason, r.x, r.width) for r in rows]
    mol = accepted[0]
    assert mol.x + mol.width <= 560                # the plot is not in the crop
    assert mol.y + mol.height >= 800 + 60 + 18     # the detached OH label is in the crop
    assert "[split]" in mol.reason


def test_without_split_the_panel_is_one_region(tmp_path):
    path = tmp_path / "merged.png"
    Image.fromarray(merged_page()).save(path)
    cfg = load_config()
    cfg.detect.split_enabled = False
    rows = run_stage1_2(path, tmp_path / "out", cfg)
    assert sum(r.label != "not_molecule" for r in rows) <= 1
    assert not any("[split]" in r.reason for r in rows)


def test_single_large_molecule_stays_whole():
    page = np.full((1400, 1400, 3), 255, np.uint8)
    draw_ring_system(page, 300, 500, bond=70, rings=4, label_gap=10)
    cfg = load_config()

    def classify(b):
        x, y, w, h = b
        return classify_crop(page[y:y + h, x:x + w], DPI, cfg.gate)

    parts = split_region(page, (200, 380, 900, 420), DPI, cfg.detect, classify)
    accepted = [p for p in parts if p[1].label != "not_molecule"]
    assert len(accepted) == 1
    (x, y, w, h), _ = accepted[0]
    assert x <= 245 and x + w >= 720 and y + h > 500 + 140   # all 4 rings and the OH label


def test_ink_close_uses_ink_not_boxes():
    page = np.full((200, 400, 3), 255, np.uint8)
    cv2.line(page, (10, 10), (190, 190), BLACK, 2)      # diagonal: its box is a big square
    cv2.line(page, (210, 10), (390, 10), BLACK, 2)      # box 20 px away, ink ~180 px away
    cv2.line(page, (212, 190), (390, 190), BLACK, 2)    # ink ~22 px away
    diagonal = (9, 9, 183, 183)
    assert not ink_close(page, diagonal, (209, 9, 183, 3), 25)
    assert ink_close(page, diagonal, (211, 189, 181, 3), 25)
