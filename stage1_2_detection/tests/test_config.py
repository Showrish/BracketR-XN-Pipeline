import pytest

from stage1_2_detection import load_config


def test_defaults_are_sane():
    cfg = load_config()
    assert cfg.render.dpi == 300
    assert cfg.detect.backend == "classical"
    assert 0 < cfg.gate.min_score < 1


def test_yaml_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("render:\n  dpi: 150\ngate:\n  min_score: 0.7\n")
    cfg = load_config(path)
    assert cfg.render.dpi == 150 and cfg.gate.min_score == 0.7
    assert cfg.detect.merge_gap_in == 0.07           # untouched values keep defaults


def test_unknown_setting_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("gate:\n  not_a_setting: 1\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_shipped_default_yaml_loads():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "default_config.yaml")
    assert cfg.render.dpi == 300
