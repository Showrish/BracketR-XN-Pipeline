"""Compare confidence thresholds on one answer key (exploratory)."""
import argparse
from pathlib import Path

from stage4_5_evaluation.pipeline import combine, evaluate, validate, write_csv


def analyze(stage3, answer_key, out):
    audit = validate(stage3, min_confidence=0)
    results = []
    for threshold in (0, 0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
        filtered = [dict(row, status='low_confidence')
                    if row['status'] == 'accepted' and row['confidence'] != ''
                    and row['confidence'] < threshold else row for row in audit]
        scores = evaluate(combine(filtered), answer_key)[-1]
        results.append(dict(min_confidence=threshold, **scores))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, results, list(results[0]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage3', required=True)
    parser.add_argument('--answer-key', required=True)
    parser.add_argument('--out', required=True, help='Output CSV path')
    args = parser.parse_args()
    analyze(args.stage3, args.answer_key, args.out)
    print(f'Saved exploratory threshold results to {args.out}')


if __name__ == '__main__':
    main()
