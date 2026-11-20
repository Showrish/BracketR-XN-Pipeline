import csv

import pytest

from stage3_recognition.handoff import PREDICTION_COLUMNS, read_stage2_manifest, write_rows


def test_reads_the_stage12_manifest_and_keeps_accepted_crops(stage12_manifest):
    tasks = read_stage2_manifest(stage12_manifest)
    labels = sorted(t.label for t in tasks)
    assert labels == ["markush", "molecule", "molecule", "molecule"]
    assert all(t.image_path.is_file() for t in tasks)          # crop paths resolve
    assert {t.crop_id for t in tasks} == {"p001_c02", "p001_c03", "p001_c05", "p002_c02"}


def test_finds_the_source_pdf_through_pdf_dir(stage12_manifest):
    pdf_dir = stage12_manifest.parents[1] / "sample_input"
    tasks = read_stage2_manifest(stage12_manifest, pdf_dir=pdf_dir)
    assert all(t.source_pdf is not None and t.source_pdf.name == "sample_paper.pdf" for t in tasks)


def test_missing_columns_are_reported(tmp_path):
    bad = tmp_path / "manifest.csv"
    bad.write_text("paper_id,crop_id\n1,p001_c01\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_stage2_manifest(bad)


def test_outputs_keep_the_agreed_column_order(tmp_path):
    path = write_rows(tmp_path / "p.csv", PREDICTION_COLUMNS,
                      [{"paper_id": "1", "crop_id": "p001_c01", "predicted_smiles": "CCO", "confidence": None}])
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == PREDICTION_COLUMNS
    assert rows[1][PREDICTION_COLUMNS.index("confidence")] == ""   # None is written as empty
