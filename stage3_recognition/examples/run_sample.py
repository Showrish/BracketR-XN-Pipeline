"""Reproduce the sample run of stage 3 and check it against known answers.

Two test cases, both with known answers:
1. Stage 1-2's own sample output (``stage1_2_detection/examples/sample_output``):
   three molecule crops (phenol, naphthalene, butanamide) and one Markush crop
   with no R-group table, from Jirapa's synthetic two-page paper.
2. The Markush case from ``make_markush_sample.py``: a scaffold (R1, R2) whose
   R-group table defines four compounds.

Usage, from the repository root (in the MolScribe environment):
    python -m stage3_recognition.examples.run_sample

Writes ``examples/sample_output/`` (predictions, Markush files, summaries, metrics.json).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from stage3_recognition import load_config, run_stage3
from stage3_recognition.examples.make_markush_sample import DEFAULT_DIR, build

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE12_SAMPLE = REPO / "stage1_2_detection" / "examples"
OUTPUT = HERE / "sample_output"

# What Jirapa's synthetic paper actually draws (stage1_2_detection/tests/synthetic.py)
EXPECTED_STAGE12 = {"p001_c02": "Oc1ccccc1",          # phenol
                    "p001_c03": "c1ccc2ccccc2c1",     # naphthalene
                    "p001_c05": "CCCC(N)=O"}          # butanamide chain


def canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(rows, key: str, smiles_column: str, expected: Dict[str, str]) -> Dict:
    """Exact match of canonical SMILES against the known answers."""
    results = []
    for row in rows:
        truth = expected.get(row[key])
        if truth is None:
            continue
        predicted = row[smiles_column]
        results.append({key: row[key], "predicted": predicted, "expected": truth,
                        "exact_match": canonical(predicted) is not None and canonical(predicted) == canonical(truth)})
    return {"cases": len(results), "exact_matches": sum(r["exact_match"] for r in results), "details": results}


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    cfg = load_config()

    # 1. stage 1-2 sample output
    out1 = OUTPUT / "stage12_sample"
    run_stage3(STAGE12_SAMPLE / "sample_output" / "manifest.csv", out1, cfg,
               pdf_dir=STAGE12_SAMPLE / "sample_input")
    recognition = check(read_csv(out1 / "predictions_molscribe.csv"), "crop_id", "predicted_smiles",
                        EXPECTED_STAGE12)

    # 2. Markush case
    manifest = build(DEFAULT_DIR)
    out2 = OUTPUT / "markush_sample"
    run_stage3(manifest, out2, cfg)
    expected = {r["compound_label"]: r["smiles"] for r in read_csv(DEFAULT_DIR / "expected.csv")}
    markush = check(read_csv(out2 / "markush_compounds.csv"), "compound_label", "smiles", expected)
    scaffold = read_csv(out2 / "markush_scaffolds.csv")

    metrics = {"recognition_stage12_sample": recognition, "markush_sample": markush,
               "markush_scaffold_read": scaffold}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: {"cases": v["cases"], "exact_matches": v["exact_matches"]}
                      for k, v in metrics.items() if isinstance(v, dict)}, indent=2))
    for row in scaffold:
        print(f"Markush scaffold read as {row['raw_prediction']!r} -> outcome {row['outcome']}")


if __name__ == "__main__":
    main()
