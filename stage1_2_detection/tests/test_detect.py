import numpy as np

from stage1_2_detection.config import DetectConfig
from stage1_2_detection.detect import (detect_classical, detect_decimer, locate_crop,
                                       merge_overlapping, to_gray)
from stage1_2_detection.render import render_pages
from stage1_2_detection.tests.synthetic import evaluate


def test_merge_overlapping_boxes():
    merged = merge_overlapping([(0, 0, 10, 10), (5, 5, 10, 10), (100, 100, 5, 5)])
    assert len(merged) == 2 and (0, 0, 15, 15) in merged


def test_blank_page_has_no_regions():
    page = np.full((800, 600, 3), 255, np.uint8)
    assert detect_classical(page, 300, DetectConfig()) == []


def test_every_molecule_is_found_and_non_molecules_not_accepted(sample):
    metrics = evaluate(sample["rows"], sample["truth_px"])
    assert metrics["recall"] == 1.0, metrics
    assert metrics["false_accepts"] == 0, metrics
    assert metrics["markush_ok"], metrics


def test_locate_crop_recovers_position(sample):
    page = dict(render_pages(sample["pdf"], 300))[1]
    crop = page[800:1200, 400:900]
    assert locate_crop(to_gray(page), crop, 0.9) == (400, 800, 500, 400)


def test_decimer_backend_with_fake_segmenter(sample):
    """The DECIMER adapter should recover boxes for crops that are page slices."""
    page = dict(render_pages(sample["pdf"], 300))[1]
    slices = [page[800:1250, 400:700], page[800:1150, 1350:1850]]
    out = detect_decimer(page, DetectConfig(), segmenter=lambda img, expand=True: slices)
    assert [box for box, _ in out] == [(400, 800, 300, 450), (1350, 800, 500, 350)]
