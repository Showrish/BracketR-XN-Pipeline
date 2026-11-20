"""Shared fixtures. The tests use a stand-in recognizer, so they run without the
models (no GPU, no downloads); the real models are exercised by examples/run_sample.py."""
import sys
from pathlib import Path

import pytest

# Allow ``pytest`` to be run from inside the folder or from the repo root.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from stage3_recognition.recognizers.base import Recognizer  # noqa: E402

STAGE12_SAMPLE = REPO / "stage1_2_detection" / "examples"


class StubRecognizer(Recognizer):
    """Answers from a dict keyed by crop file name; unknown crops raise, like a model failure."""

    name = "Stub"

    def __init__(self, answers):
        self.answers = answers

    def _predict(self, image_path):
        if image_path.stem not in self.answers:
            raise ValueError("unreadable crop")
        return self.answers[image_path.stem], 0.9


@pytest.fixture
def stub_factory():
    def make(answers):
        return lambda tool, cfg: StubRecognizer(answers)
    return make


@pytest.fixture
def stage12_manifest():
    path = STAGE12_SAMPLE / "sample_output" / "manifest.csv"
    if not path.exists():
        pytest.skip("stage 1-2 sample output not found")
    return path
