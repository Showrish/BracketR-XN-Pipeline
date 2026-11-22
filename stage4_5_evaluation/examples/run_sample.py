"""Score the stage 3 Markush sample against known answers."""
import argparse
import csv
from pathlib import Path
from stage4_5_evaluation.pipeline import run, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='stage4_5_evaluation/examples/sample_output')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = root / 'stage3_recognition/examples/sample_input/markush_sample/expected.csv'
    with expected.open() as handle:
        rows = list(csv.DictReader(handle))
    truth = out / 'answer_key.csv'
    write_csv(truth, [dict(paperID='markush_paper', canonical_SMILES=r['smiles']) for r in rows],
              ['paperID', 'canonical_SMILES'])
    print(run(root / 'stage3_recognition/examples/sample_output/markush_sample', out, truth))


if __name__ == '__main__':
    main()
