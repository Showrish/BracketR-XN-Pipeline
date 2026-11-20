"""Building full compounds from a scaffold and a table (needs RDKit)."""
import pytest

Chem = pytest.importorskip("rdkit.Chem")

from stage3_recognition.markush import RGroupTable, enumerate_compounds  # noqa: E402


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


TABLE = RGroupTable(["R1", "R2"], [("5a", {"R1": "H", "R2": "Cl"}), ("5b", {"R1": "Me", "R2": "F"}),
                                   ("5c", {"R1": "OMe", "R2": "Br"}), ("5d", {"R1": "CF3", "R2": "OH"})],
                    "text_layer")
EXPECTED = {"5a": "Clc1ccccc1", "5b": "Cc1ccc(F)cc1", "5c": "COc1ccc(Br)cc1", "5d": "Oc1ccc(C(F)(F)F)cc1"}


def test_numbered_scaffold_gives_the_expected_compounds():
    compounds = enumerate_compounds("[1*]c1ccc([2*])cc1", TABLE)
    assert [c.status for c in compounds] == ["ok"] * 4
    assert {c.label: c.smiles for c in compounds} == {k: canon(v) for k, v in EXPECTED.items()}


def test_numbering_wins_over_the_order_of_the_atoms():
    compounds = enumerate_compounds("[2*]c1ccc([1*])cc1", TABLE)     # R2 written first
    assert {c.label: c.smiles for c in compounds} == {k: canon(v) for k, v in EXPECTED.items()}


def test_unnumbered_points_follow_the_table_order_and_say_so():
    compounds = enumerate_compounds("*c1ccc(*)cc1", TABLE)
    assert all(c.status == "ok_assumed_column_order" for c in compounds)
    assert compounds[1].smiles == canon("Cc1ccc(F)cc1")


def test_problems_are_reported_not_guessed():
    odd = RGroupTable(["R1", "R2"], [("7a", {"R1": "morpholino", "R2": "H"})], "csv")
    assert enumerate_compounds("[1*]c1ccc([2*])cc1", odd)[0].status.startswith("unknown_substituent")
    assert enumerate_compounds("c1ccccc1", TABLE)[0].status == "no_attachment_points"
    assert enumerate_compounds("[1*]c1ccccc1", TABLE)[0].status.startswith("attachment_mismatch")
    assert enumerate_compounds("not a smiles", TABLE)[0].status == "invalid_scaffold"
