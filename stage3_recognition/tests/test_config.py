import pytest

from stage3_recognition.config import load_config


def test_defaults_match_the_documented_choices():
    cfg = load_config()
    assert cfg.tools == ["molscribe"]
    assert cfg.molscribe.checkpoint == "swin_base_char_aux_1m680k.pth"
    assert cfg.preprocess.pad_in == 0.05
    assert cfg.markush.scaffold_tool == "molscribe"


def test_yaml_overrides_and_rejects_unknown_keys(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("tools: [molscribe, decimer]\npreprocess:\n  pad_in: 0.0\n", encoding="utf-8")
    cfg = load_config(good)
    assert cfg.tools == ["molscribe", "decimer"] and cfg.preprocess.pad_in == 0.0

    bad = tmp_path / "bad.yaml"
    bad.write_text("preprocess:\n  padding: 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
