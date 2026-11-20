"""Markush helpers that need no RDKit: placeholders, substituents, table reading."""
import shutil

import pytest

from stage3_recognition.markush import (attachment_labels, match_columns, normalize_attachment_points,
                                        parse_table_lines, table_from_csv, table_from_pdf_text, to_fragment)


def test_decimer_placeholders_become_attachment_points():
    # real DECIMER outputs from our Phase 1 BioVista run, which RDKit rejected
    assert normalize_attachment_points("CN1CC(CC1=O)[R]") == "CN1CC(CC1=O)[*]"
    assert normalize_attachment_points("Cc1ccc(cc1)[R1]") == "Cc1ccc(cc1)[1*]"
    assert normalize_attachment_points("[Ru]C[Rh]C[Rb]") == "[Ru]C[Rh]C[Rb]"   # real elements untouched
    assert normalize_attachment_points(None) == ""


def test_attachment_labels_in_order():
    assert attachment_labels("[1*]c1ccc([2*])cc1") == [1, 2]
    assert attachment_labels("*c1ccccc1") == [None]
    assert attachment_labels("CCO") == []


def test_substituent_lookup():
    assert to_fragment("Me") == "*C"
    assert to_fragment("CH\u2083") == "*C"              # subscript three, as copied from a PDF
    assert to_fragment(" OMe, ") == "*OC"
    assert to_fragment("*CC#N") == "*CC#N"              # already a SMILES fragment
    assert to_fragment("morpholino") is None            # unknown: reported, not guessed


TABLE_TEXT = """Table 1. Substituent effects on kinase inhibition.
                R1
                                Cmpd   R1    R2   IC50 (nM)
                                5a     H     Cl   120

                                5b     Me    F    85
                                5c     OMe   Br   40
                R2
Figure 2. Something else.
"""


def test_reads_a_table_with_a_label_column_and_extra_columns():
    table = parse_table_lines(TABLE_TEXT.splitlines())
    assert table.columns == ["R1", "R2"]
    assert table.rows == [("5a", {"R1": "H", "R2": "Cl"}), ("5b", {"R1": "Me", "R2": "F"}),
                          ("5c", {"R1": "OMe", "R2": "Br"})]


def test_reads_a_table_whose_header_has_no_label_column():
    table = parse_table_lines(["R1   R2", "12a  Cl  H", "12b  F   Me"])
    assert table.rows == [("12a", {"R1": "Cl", "R2": "H"}), ("12b", {"R1": "F", "R2": "Me"})]


def test_drawing_labels_alone_are_not_a_table():
    assert parse_table_lines(["R1", "", "R2", "Some running text 12 words long."]) is None


def test_csv_table_is_filtered_by_paper_and_crop(tmp_path):
    path = tmp_path / "tables.csv"
    path.write_text("paper_id,crop_id,compound_label,R1,R2\n"
                    "19,p004_c01,12a,H,Cl\n19,p004_c01,12b,Me,F\n4,p002_c01,3a,OMe,H\n", encoding="utf-8")
    table = table_from_csv(path, "19", "p004_c01")
    assert table.source == "csv" and [label for label, _ in table.rows] == ["12a", "12b"]
    assert table_from_csv(path, "6", "p001_c01") is None


def test_matching_attachment_points_to_columns():
    assert match_columns([1, 2], ["R1", "R2"]) == (["R1", "R2"], "numbered")
    assert match_columns([2, 1], ["R1", "R2"]) == (["R2", "R1"], "numbered")
    assert match_columns([0, 0], ["R1", "R2"]) == (["R1", "R2"], "table_order")
    assert match_columns([], ["R1"])[0] is None
    assert match_columns([0], ["R1", "R2"])[1].startswith("attachment_mismatch")


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler not installed")
def test_table_is_found_in_a_real_pdf_text_layer(tmp_path):
    pytest.importorskip("reportlab")
    from stage3_recognition.examples.make_markush_sample import build
    manifest = build(tmp_path)
    table = table_from_pdf_text(manifest.parents[1] / "markush_paper.pdf", page=1)
    assert table is not None and table.columns == ["R1", "R2"]
    assert [label for label, _ in table.rows] == ["5a", "5b", "5c", "5d"]
    assert table.rows[3][1] == {"R1": "CF3", "R2": "OH"}


# Layouts seen in the course papers' R-group tables (values made up here)
SIDE_BY_SIDE = """   Compound     R          MKK4    BRaf      Compound     R                  MKK4    BRaf
      7a       -CH3          16      93        7d        -phenyl              3.3     22
      7b       nC3H7         14      16        7e        -CH2-(p)F-phenyl     15      95
      7c       c-hexyl       21      95
"""


def test_side_by_side_halves_are_both_read():
    table = parse_table_lines(SIDE_BY_SIDE.splitlines())
    assert table.columns == ["R"]
    assert [(label, cells["R"]) for label, cells in table.rows] == [
        ("7a", "-CH3"), ("7d", "-phenyl"), ("7b", "nC3H7"), ("7e", "-CH2-(p)F-phenyl"), ("7c", "c-hexyl")]


def test_columns_before_r_and_names_after_labels_are_handled():
    lines = ["Compound   A      B      R       IC50",
             "1 (VMF)    -CH=   -CH=   -H      14",
             "2          -N=    -CH=   -OCH3   12"]
    table = parse_table_lines(lines)
    assert table.rows == [("1", {"R": "-H"}), ("2", {"R": "-OCH3"})]


def test_linker_cells_with_brackets_are_kept_whole():
    table = parse_table_lines(["Entry  R           IC50", "3a     (CO)-(CO)   100", "3b     CH2         94"])
    assert table.rows[0] == ("3a", {"R": "(CO)-(CO)"})


def test_running_text_is_not_a_table():
    lines = ["such as R2 groups in the text", "394 https://doi.org/10.1021/acs.jmedchem.5c02620 more text"]
    assert parse_table_lines(lines) is None                # one implausible row only


def test_condensed_formulas_from_journal_tables():
    assert to_fragment("-CH2-CH3") == "*CC"
    assert to_fragment("nC4H9") == "*CCCC"
    assert to_fragment("-(CH2)2-CF3") == "*CCC(F)(F)F"
    assert to_fragment("-CH2-C(O)-CH3") == "*CC(=O)C"
    assert to_fragment("-CH2-(p)F-phenyl") == "*Cc1ccc(F)cc1"
    assert to_fragment("-CH2-(o)F-phenyl") == "*Cc1ccccc1F"
    assert to_fragment("cyclo-prop") == "*C1CC1"
    assert to_fragment("-ethylphenyl") is None             # ambiguous name: not guessed
    assert to_fragment("CO") is None                       # a linker, not an end group
