"""Turn the text in an R-group table cell into a SMILES fragment.

Each fragment starts with ``*``, the attachment point joined to the scaffold.
Three ways a cell is understood, in this order:
1. a common abbreviation or name (Me, OMe, CF3, phenyl, cyclo-prop ...);
2. a straight-chain alkyl formula such as nC4H9;
3. a short condensed formula written as groups joined by dashes, such as
   CH2-CH3, (CH2)2-CF3, CH2-C(O)-CH3 or CH2-(p)F-phenyl (the style of the
   R-group tables in course paper 19).
Anything else is reported as unknown rather than guessed. A leading dash
(the bond to the scaffold, as in "-CH3") is ignored.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

SUBSTITUENTS: Dict[str, str] = {
    # hydrogen and halogens
    "H": "*[H]", "F": "*F", "Cl": "*Cl", "Br": "*Br", "I": "*I",
    # alkyl groups
    "Me": "*C", "CH3": "*C", "methyl": "*C", "Et": "*CC", "C2H5": "*CC", "ethyl": "*CC",
    "Pr": "*CCC", "nPr": "*CCC", "n-Pr": "*CCC", "propyl": "*CCC",
    "iPr": "*C(C)C", "i-Pr": "*C(C)C", "iprop": "*C(C)C", "isopropyl": "*C(C)C",
    "cPr": "*C1CC1", "c-Pr": "*C1CC1", "cyclo-prop": "*C1CC1", "c-prop": "*C1CC1", "cyclopropyl": "*C1CC1",
    "Bu": "*CCCC", "nBu": "*CCCC", "n-Bu": "*CCCC", "butyl": "*CCCC",
    "tBu": "*C(C)(C)C", "t-Bu": "*C(C)(C)C", "tert-butyl": "*C(C)(C)C",
    "c-hexyl": "*C1CCCCC1", "cyclohexyl": "*C1CCCCC1",
    "CF3": "*C(F)(F)F", "CHF2": "*C(F)F", "CH2OH": "*CO",
    # aryl and benzyl
    "Ph": "*c1ccccc1", "phenyl": "*c1ccccc1", "Bn": "*Cc1ccccc1", "benzyl": "*Cc1ccccc1",
    # oxygen, sulfur and nitrogen groups
    "OH": "*O", "OMe": "*OC", "OCH3": "*OC", "OEt": "*OCC", "OCF3": "*OC(F)(F)F",
    "OPh": "*Oc1ccccc1", "SH": "*S", "SMe": "*SC",
    "NH2": "*N", "NHMe": "*NC", "NMe2": "*N(C)C", "NHAc": "*NC(C)=O",
    "NO2": "*[N+](=O)[O-]", "CN": "*C#N",
    # carbonyl groups
    "CHO": "*C=O", "Ac": "*C(C)=O", "COMe": "*C(C)=O",
    "COOH": "*C(=O)O", "CO2H": "*C(=O)O", "COOMe": "*C(=O)OC", "CO2Me": "*C(=O)OC",
    "CO2Et": "*C(=O)OCC", "CONH2": "*C(N)=O", "SO2Me": "*S(C)(=O)=O",
}

# Groups inside a condensed formula. Chain groups can sit anywhere; end groups only last.
CHAIN_GROUPS = {"CH2": "C", "C(O)": "C(=O)", "CO": "C(=O)", "C(OH)": "C(O)", "CH(OH)": "C(O)",
                "O": "O", "NH": "N", "S": "S"}
END_GROUPS = {"CH3": "C", "CF3": "C(F)(F)F", "iprop": "C(C)C", "iPr": "C(C)C", "tBu": "C(C)(C)C",
              "phenyl": "c1ccccc1", "Ph": "c1ccccc1", "OH": "O", "OCH3": "OC", "OMe": "OC",
              "NH2": "N", "CN": "C#N", "F": "F", "Cl": "Cl", "Br": "Br"}
# (o)/(m)/(p) position of a substituent on a phenyl ring attached by its first atom
RING_POSITION = {"o": "c1ccccc1{x}", "m": "c1cccc({x})c1", "p": "c1ccc({x})cc1"}
RING_SUBSTITUENT = {"F": "F", "Cl": "Cl", "Br": "Br", "OH": "O", "OMe": "OC", "Me": "C", "CH3": "C", "CF3": "C(F)(F)F"}

N_ALKYL = re.compile(r"^n-?C(\d{1,2})H(\d{1,2})$")                 # nC4H9
REPEATED = re.compile(r"^\(CH2\)(\d)$")                            # (CH2)2
POSITIONAL_ARYL = re.compile(r"^\((o|m|p)\)-?(\w+)-phenyl$")       # (p)F-phenyl


def clean_token(token: str) -> str:
    """Normalise how a cell is typed: subscript digits, unicode dashes, spaces."""
    text = unicodedata.normalize("NFKC", token or "")
    for dash in ("\u2212", "\u2013", "\u2014", "\u2010"):
        text = text.replace(dash, "-")
    return text.strip().strip(",;")


def cell_text(token: str) -> str:
    """Cell text with the leading bond dash removed ("-CH3" -> "CH3")."""
    return clean_token(token).lstrip("-")


def _split_groups(text: str) -> List[str]:
    """Split a condensed formula at dashes that are not inside brackets."""
    groups, depth, current = [], 0, ""
    for char in text:
        depth += char == "("
        depth -= char == ")"
        if char == "-" and depth == 0:
            groups.append(current)
            current = ""
        else:
            current += char
    return groups + [current]


def _group_smiles(group: str, last: bool) -> Optional[str]:
    """SMILES for one group. The last group must close the chain (CH3, CF3, phenyl ...),
    so a bare linker such as CO or CH2 is never read as a complete substituent."""
    if last:
        return END_GROUPS.get(group)
    if group in CHAIN_GROUPS:
        return CHAIN_GROUPS[group]
    repeated = REPEATED.match(group)
    return "C" * int(repeated.group(1)) if repeated else None


def condensed_to_fragment(text: str) -> Optional[str]:
    """Fragment for a short condensed formula, or None if any group is not understood."""
    # a positional aryl group may be written with a dash inside, e.g. (o)-F-phenyl
    aryl = None
    for position in RING_POSITION:
        marker = f"({position})"
        if marker in text:
            head, _, tail = text.partition(marker)
            match = POSITIONAL_ARYL.match(marker + tail)
            if not match or match.group(2) not in RING_SUBSTITUENT:
                return None
            aryl = RING_POSITION[position].format(x=RING_SUBSTITUENT[match.group(2)])
            text = head.rstrip("-")
            break
    groups = [g for g in _split_groups(text) if g] if text else []
    parts = []
    for i, group in enumerate(groups):
        smiles = _group_smiles(group, last=(i == len(groups) - 1 and aryl is None))
        if smiles is None:
            return None
        parts.append(smiles)
    if aryl:
        parts.append(aryl)
    return "*" + "".join(parts) if parts else None


def to_fragment(token: str) -> Optional[str]:
    """SMILES fragment for a table cell, or None if it is not recognised.

    A cell already written as a SMILES fragment (contains *) is used as is.
    """
    text = cell_text(token)
    if not text:
        return None
    if text in SUBSTITUENTS:
        return SUBSTITUENTS[text]
    if "*" in text:
        return text
    alkyl = N_ALKYL.match(text)
    if alkyl and int(alkyl.group(2)) == 2 * int(alkyl.group(1)) + 1:
        return "*" + "C" * int(alkyl.group(1))
    return condensed_to_fragment(text)
