"""Shared fixtures: one synthetic PDF, rendered and processed once per test session."""
import sys
from pathlib import Path

import pytest

# Allow ``pytest`` to be run from inside the folder or from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_2_detection import load_config, run_stage1_2          # noqa: E402
from stage1_2_detection.tests.synthetic import build_sample_pdf, truth_to_pixels  # noqa: E402


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sample")
    pdf = folder / "sample.pdf"
    truth = build_sample_pdf(pdf)
    cfg = load_config()
    out = folder / "out"
    rows = run_stage1_2(pdf, out, cfg, overlay=True)
    return {"pdf": pdf, "out": out, "rows": rows, "cfg": cfg,
            "truth_px": truth_to_pixels(truth, cfg.render.dpi)}
