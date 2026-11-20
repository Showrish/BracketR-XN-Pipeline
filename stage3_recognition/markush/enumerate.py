"""Build full compounds from a scaffold and its R-group table.

Each attachment point in the scaffold is paired with a table column: by number
when the scaffold carries numbered points ([1*] -> R1), otherwise by order when
the counts match (reported as an assumption). Each substituent becomes a small
fragment, and RDKit's ``molzip`` joins fragment and scaffold at matching
atom-map numbers. An H cell turns the attachment point itself into a hydrogen,
which RDKit then removes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .rgroup_table import RGroupTable
from .substituents import clean_token, to_fragment


@dataclass
class Compound:
    label: str
    smiles: Optional[str]
    status: str          # ok | ok_assumed_column_order | reason it failed


def _column_number(column: str) -> Optional[int]:
    match = re.match(r"^R(\d+)$", column)
    return int(match.group(1)) if match else None


def match_columns(point_labels: List[int], columns: List[str]) -> Tuple[Optional[List[str]], str]:
    """Pair attachment points (isotope numbers, 0 = unnumbered) with table columns."""
    if not point_labels:
        return None, "no_attachment_points"
    if all(label > 0 for label in point_labels):
        numbered = {_column_number(c): c for c in columns if _column_number(c) is not None}
        if all(label in numbered for label in point_labels):
            return [numbered[label] for label in point_labels], "numbered"
    if len(point_labels) == len(columns):
        return list(columns), "table_order"
    return None, f"attachment_mismatch({len(point_labels)} points, {len(columns)} columns)"


def _build_one(Chem, scaffold, columns: List[str], label: str, cells, how: str) -> Compound:
    base = Chem.RWMol(scaffold)
    points = [a.GetIdx() for a in base.GetAtoms() if a.GetAtomicNum() == 0]
    fragments = None
    for map_number, (atom_index, column) in enumerate(zip(points, columns), start=1):
        token = cells.get(column, "")
        if clean_token(token) == "H":
            # A hydrogen: turn the attachment point itself into H, which RemoveHs drops below.
            point = base.GetAtomWithIdx(atom_index)
            point.SetAtomicNum(1)
            point.SetIsotope(0)          # an isotope label would make RemoveHs keep it
            continue
        smiles = to_fragment(token)
        fragment = Chem.MolFromSmiles(smiles) if smiles else None
        if fragment is None:
            return Compound(label, None, f"unknown_substituent({column}={token})")
        ends = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
        if len(ends) != 1:
            return Compound(label, None, f"substituent_needs_one_attachment({column}={token})")
        ends[0].SetAtomMapNum(map_number)
        base.GetAtomWithIdx(atom_index).SetAtomMapNum(map_number)
        fragments = fragment if fragments is None else Chem.CombineMols(fragments, fragment)
    try:
        mol = base.GetMol()
        if fragments is not None:          # None when every group was H
            mol = Chem.molzip(mol, fragments)
        joined = Chem.RemoveHs(mol)
    except Exception as exc:  # noqa: BLE001 - reported per compound, the run continues
        return Compound(label, None, f"join_failed({type(exc).__name__})")
    status = "ok" if how == "numbered" else "ok_assumed_column_order"
    return Compound(label, Chem.MolToSmiles(joined), status)


def enumerate_compounds(scaffold_smiles: str, table: RGroupTable) -> List[Compound]:
    """One Compound per table row, or a single failed Compound explaining why none could be built."""
    from rdkit import Chem  # imported here so the rest of stage 3 works without RDKit

    scaffold = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None
    if scaffold is None:
        return [Compound("", None, "invalid_scaffold")]
    labels = [a.GetIsotope() for a in scaffold.GetAtoms() if a.GetAtomicNum() == 0]
    columns, how = match_columns(labels, table.columns)
    if columns is None:
        return [Compound("", None, how)]
    return [_build_one(Chem, scaffold, columns, label, cells, how) for label, cells in table.rows]
