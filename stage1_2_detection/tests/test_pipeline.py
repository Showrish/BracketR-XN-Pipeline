import json
from pathlib import Path

import numpy as np
from PIL import Image

from stage1_2_detection import (load_config, read_manifest, run_many, run_stage1_2,
                                select_labels)
from stage1_2_detection.manifest import COLUMNS


def test_manifest_columns_and_files(sample):
    out = Path(sample["out"])
    rows = read_manifest(out / "manifest.csv")
    assert list(rows[0].__dict__) == COLUMNS
    for row in rows:
        assert (out / row.crop_path).exists()
        assert row.dpi == 300 and row.source_pdf == "sample.pdf"
    assert (out / "overlays" / "paper_sample" / "p001.png").exists()


def test_labels_counts_and_routing(sample):
    rows = read_manifest(Path(sample["out"]) / "manifest.csv")
    assert len(select_labels(rows, ("molecule",))) == 3
    assert len(select_labels(rows, ("markush",))) == 1
    assert all(r.crop_path.startswith("crops/") for r in select_labels(rows, ("molecule", "markush")))
    assert all(r.crop_path.startswith("rejected/") for r in select_labels(rows, ("not_molecule",)))
    assert all(r.reason for r in rows)


def test_positions_are_inside_the_page(sample):
    for r in sample["rows"]:
        assert 0 <= r.x and r.x + r.width <= 2551 and r.y + r.height <= 3301


def test_page_range_limits_processing(sample, tmp_path):
    cfg = load_config()
    cfg.render.first_page = cfg.render.last_page = 2
    rows = run_stage1_2(sample["pdf"], tmp_path, cfg)
    assert {r.page for r in rows} == {2}


def test_no_rejected_crops_option(sample, tmp_path):
    run_stage1_2(sample["pdf"], tmp_path, save_rejected=False)
    assert not list((tmp_path / "rejected").glob("**/*.png"))


def test_single_image_input(tmp_path):
    """A BioVista-style cropped image is treated as a one-page document."""
    page = np.full((900, 900, 3), 255, np.uint8)
    Image.fromarray(page).save(tmp_path / "blank.png")
    rows = run_stage1_2(tmp_path / "blank.png", tmp_path / "o")
    assert rows == []


def test_run_many_writes_summary(sample, tmp_path):
    rows = run_many([sample["pdf"]], tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["total_regions"] == len(rows)
    assert summary["by_label"]["molecule"] == 3 and summary["by_label"]["markush"] == 1
    assert (tmp_path / "manifest.csv").exists()
