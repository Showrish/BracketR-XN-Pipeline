"""Run from the repository root with python -m stage4_5_evaluation.cli."""
import argparse
import json
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage3', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--answer-key', help='CSV with paperID, canonical_SMILES')
    parser.add_argument('--min-confidence', type=float, default=0.2)
    parser.add_argument('--keep-fragments', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.stage3, args.out, args.answer_key,
                         args.min_confidence, not args.keep_fragments), indent=2))


if __name__ == '__main__':
    main()
