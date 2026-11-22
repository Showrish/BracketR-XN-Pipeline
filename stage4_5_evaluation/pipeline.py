"""Validate, combine, and score stage 3 predictions."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .chemistry import normalize

KEY = ('paper_id', 'page', 'crop_id')
AUDIT_COLUMNS = [*KEY, 'tool', 'compound_label', 'raw_smiles', 'canonical_smiles',
                 'confidence', 'method', 'status', 'error', 'source_file']
COMBINED_COLUMNS = [*AUDIT_COLUMNS, 'disagreement']


def read_csv(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    for row in rows:
        for column in required:
            if row[column] is None:
                raise ValueError(f'{path}: incomplete CSV row')
    return rows


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def confidence_value(value):
    if not value:
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f'Confidence must be between 0 and 1: {value}')
    return number


def validate(stage3, min_confidence=0.2, largest_fragment=True):
    """Validate predictions and retain all rows in the audit."""
    stage3 = Path(stage3)
    paths = sorted(stage3.glob('predictions_*.csv'))
    markush = stage3 / 'markush_compounds.csv'
    if markush.exists():
        paths.append(markush)
    if not paths:
        raise ValueError(f'No stage 3 prediction files in {stage3}')
    audit = []
    seen = set()
    for path in paths:
        is_markush = path == markush
        required = [*KEY, 'smiles', 'method'] if is_markush else [*KEY, 'tool', 'predicted_smiles']
        for row in read_csv(path, required):
            if any(not row[k].strip() for k in KEY):
                raise ValueError(f'{path}: blank crop identity')
            raw = row.get('smiles' if is_markush else 'predicted_smiles', '')
            canonical = normalize(raw, largest_fragment)
            confidence = confidence_value(row.get('confidence', ''))
            method = row.get('method', '')
            status = 'accepted'
            if row.get('error'):
                status = 'recognition_error'
            elif is_markush and not method.startswith('enumerated_'):
                status = 'unresolved_markush'
            elif canonical is None:
                status = 'invalid_smiles'
            elif '*' in canonical:
                status = 'unresolved_scaffold'
            elif confidence is not None and confidence < min_confidence:
                status = 'low_confidence'
            result = {k: row[k].strip() for k in KEY}
            result.update(tool='Markush' if is_markush else row['tool'],
                          compound_label=row.get('compound_label', ''), raw_smiles=raw,
                          canonical_smiles=canonical or '', confidence='' if confidence is None else confidence,
                          method=method, status=status, error=row.get('error', ''), source_file=path.name)
            identity = (*[result[k] for k in KEY], result['tool'], result['compound_label'])
            if identity in seen:
                raise ValueError(f'Duplicate prediction identity: {identity}')
            seen.add(identity)
            audit.append(result)
    return audit


def combine(audit):
    """Prefer valid MolScribe, then DECIMER; keep all successful Markush compounds.

    Confidence is not compared between tools. Disagreements remain in the audit.
    Expanded compounds replace scaffold predictions for the same crop.
    """
    groups = defaultdict(list)
    for row in audit:
        groups[tuple(row[k] for k in KEY)].append(row)
    combined = []
    for key, rows in sorted(groups.items()):
        accepted = [r for r in rows if r['status'] == 'accepted']
        expanded = [r for r in accepted if r['tool'] == 'Markush']
        readings = [r for r in accepted if r['tool'] != 'Markush']
        disagreement = len({r['canonical_smiles'] for r in readings}) > 1
        selected = expanded or sorted(readings, key=lambda r: (
            {'molscribe': 0, 'decimer': 1}.get(r['tool'].lower(), 2), r['tool']))[:1]
        combined.extend(dict(r, disagreement=disagreement) for r in selected)
    return combined


def evaluate(combined, answer_key, largest_fragment=True):
    """Unique molecular-set precision/recall per paper, then micro totals.

    Invalid answer-key entries are errors. Missing predictions still incur FN.
    Predictions for papers absent from the key are reported as unscored.
    """
    truth = defaultdict(set)
    for row in read_csv(answer_key, ['paperID', 'canonical_SMILES']):
        paper = row['paperID'].strip()
        canonical = normalize(row['canonical_SMILES'], largest_fragment)
        if not paper or canonical is None:
            raise ValueError(f'Invalid answer-key entry: {row}')
        truth[paper].add(canonical)
    if not truth:
        raise ValueError('Answer key is empty')
    predicted = defaultdict(set)
    for row in combined:
        predicted[row['paper_id']].add(row['canonical_smiles'])
    results = []
    totals = [0, 0, 0]
    for paper in sorted(set(truth) | set(predicted)):
        if paper not in truth:
            results.append(dict(paper_id=paper, status='no_answer_key', predictions=len(predicted[paper]),
                                truth='', tp='', fp='', fn='', precision='', recall='', f1=''))
            continue
        p, t = predicted[paper], truth[paper]
        tp, fp, fn = len(p & t), len(p - t), len(t - p)
        totals = [a + b for a, b in zip(totals, (tp, fp, fn))]
        results.append(metric_row(paper, tp, fp, fn))
    results.append(metric_row('__micro__', *totals))
    return results


def metric_row(paper, tp, fp, fn):
    return dict(paper_id=paper, status='scored', predictions=tp + fp, truth=tp + fn,
                tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def run(stage3, out, answer_key=None, min_confidence=0.2, largest_fragment=True):
    if not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence must be between 0 and 1')
    audit = validate(stage3, min_confidence, largest_fragment)
    combined = combine(audit)
    scores = evaluate(combined, answer_key, largest_fragment) if answer_key else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'validated_predictions.csv', audit, AUDIT_COLUMNS)
    write_csv(out / 'combined_predictions.csv', combined, COMBINED_COLUMNS)
    if scores:
        write_csv(out / 'evaluation.csv', scores, list(scores[0]))
    else:
        (out / 'evaluation.csv').unlink(missing_ok=True)
    summary = dict(input_rows=len(audit), accepted_rows=sum(r['status'] == 'accepted' for r in audit),
                   combined_rows=len(combined), min_confidence=min_confidence,
                   largest_fragment=largest_fragment, evaluated=bool(scores),
                   statuses={s: sum(r['status'] == s for r in audit) for s in sorted({r['status'] for r in audit})})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
