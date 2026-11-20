"""End to end with a stand-in recognizer: manifest in, hand-off files out."""
import csv
import json
import shutil

import pytest

from stage3_recognition import MARKUSH_COLUMNS, PREDICTION_COLUMNS, load_config, run_stage3


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_stage12_sample_runs_end_to_end(tmp_path, stage12_manifest, stub_factory):
    answers = {"p001_c02": "Oc1ccccc1", "p001_c03": "c1ccc2ccccc2c1",       # p001_c05 "fails"
               "p002_c02": "[1*]c1ccc([2*])c(Cl)c1"}
    summary = run_stage3(stage12_manifest, tmp_path, load_config(),
                         pdf_dir=stage12_manifest.parents[1] / "sample_input",
                         recognizer_factory=stub_factory(answers))

    rows = read(tmp_path / "predictions_molscribe.csv")
    assert list(rows[0]) == PREDICTION_COLUMNS and len(rows) == 3
    failed = [r for r in rows if r["error"]]
    assert len(failed) == 1 and failed[0]["crop_id"] == "p001_c05"        # one bad crop, run continues

    markush = read(tmp_path / "markush_compounds.csv")
    assert list(markush[0]) == MARKUSH_COLUMNS
    assert markush[0]["method"] == "scaffold_only:no_table"               # that page has no R-group table
    assert summary["crops"] == {"molecule": 3, "markush": 1}
    assert json.loads((tmp_path / "summary.json").read_text())["tools"]["molscribe"]["errors"] == 1
    assert (tmp_path / "prepared" / "paper_sample_paper" / "p001_c02.png").is_file()


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_markush_sample_is_expanded_into_full_compounds(tmp_path, stub_factory):
    Chem = pytest.importorskip("rdkit.Chem")
    pytest.importorskip("reportlab")
    from stage3_recognition.examples.make_markush_sample import EXPECTED, build

    manifest = build(tmp_path / "input")
    run_stage3(manifest, tmp_path / "out", load_config(),
               recognizer_factory=stub_factory({"p001_c01": "[R1]c1ccc([R2])cc1"}))   # DECIMER-style placeholders

    rows = {r["compound_label"]: r for r in read(tmp_path / "out" / "markush_compounds.csv")}
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))           # noqa: E731
    assert {k: r["smiles"] for k, r in rows.items()} == {k: canon(v) for k, v in EXPECTED.items()}
    assert all(r["method"] == "enumerated_text_layer" for r in rows.values())


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_scaffold_missed_by_the_gate_is_still_expanded(tmp_path, stub_factory):
    """A crop labelled 'molecule' whose reading has attachment points goes to 3b too."""
    pytest.importorskip("rdkit.Chem")
    pytest.importorskip("reportlab")
    from stage3_recognition.examples.make_markush_sample import build

    manifest = build(tmp_path / "input")
    text = manifest.read_text(encoding="utf-8").replace(",markush,", ",molecule,")
    manifest.write_text(text, encoding="utf-8")
    run_stage3(manifest, tmp_path / "out", load_config(),
               recognizer_factory=stub_factory({"p001_c01": "[1*]c1ccc([2*])cc1"}))

    audit = read(tmp_path / "out" / "markush_scaffolds.csv")
    assert audit[0]["gate_label"] == "molecule" and audit[0]["outcome"].startswith("enumerated(4/4)")
    assert len(read(tmp_path / "out" / "markush_compounds.csv")) == 4
