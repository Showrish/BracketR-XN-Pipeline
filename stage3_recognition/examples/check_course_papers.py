"""Quick check of stage 3 output on the six course papers against the answer key.

This is a sanity check of what stage 3 produces on real input, not the group's
official scoring (that is stage 5). A prediction counts only if its canonical
SMILES exactly matches a structure in the paper's answer key; each structure is
counted once.

    python -m stage3_recognition.examples.check_course_papers \
        --stage3 out/stage3 --answer-key data/ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def canonical(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_answer_key(path: Path):
    truth = defaultdict(set)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            smiles = canonical(row["canonical_SMILES"])
            if smiles:
                truth[str(row["paperID"]).strip()].add(smiles)
    return truth


def load_predictions(stage3: Path):
    predicted = defaultdict(list)
    for path in sorted(stage3.glob("predictions_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                predicted[(row["tool"], row["paper_id"])].append(row["predicted_smiles"])
    markush = stage3 / "markush_compounds.csv"
    if markush.exists():
        with open(markush, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                predicted[("Markush branch", row["paper_id"])].append(row["smiles"])
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage3", required=True, help="stage 3 output folder")
    parser.add_argument("--answer-key", required=True, help="course answer key CSV")
    args = parser.parse_args()

    truth = load_answer_key(Path(args.answer_key))
    results = []
    for (tool, paper), smiles_list in sorted(load_predictions(Path(args.stage3)).items()):
        valid = [c for c in (canonical(s) for s in smiles_list) if c]
        found = set(valid) & truth.get(paper, set())
        results.append({"tool": tool, "paper_id": paper, "outputs": len(smiles_list), "valid": len(valid),
                        "unique_valid": len(set(valid)), "answer_key_structures": len(truth.get(paper, set())),
                        "exact_matches": len(found)})
    out = Path(args.stage3) / "course_paper_check.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'tool':15} {'paper':>5} {'outputs':>7} {'valid':>5} {'matches':>7} {'in key':>6}")
    for r in results:
        print(f"{r['tool']:15} {r['paper_id']:>5} {r['outputs']:>7} {r['valid']:>5} "
              f"{r['exact_matches']:>7} {r['answer_key_structures']:>6}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
