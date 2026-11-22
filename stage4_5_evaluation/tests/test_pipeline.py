import csv
from pathlib import Path

import pytest

from stage4_5_evaluation.chemistry import normalize
from stage4_5_evaluation.pipeline import combine, evaluate, run, validate

ROOT = Path(__file__).resolve().parents[2]


def save(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate(tool='MolScribe', smiles='CCO', confidence='0.9'):
    return dict(paper_id='1', page='1', crop_id='c1', tool=tool,
                predicted_smiles=smiles, confidence=confidence, error='')


def test_normalization():
    assert normalize('CCO.[He]') == normalize('OCC')
    assert normalize('[1*]c1ccccc1') == normalize('[*:1]c1ccccc1')
    assert normalize('CCO.[He]', False) != normalize('CCO')
    assert normalize('bad_smiles') is None
    assert normalize('F[C@H](Cl)Br') != normalize('F[C@@H](Cl)Br')


def test_fallback_and_audit(tmp_path):
    save(tmp_path / 'predictions_molscribe.csv', [candidate(confidence='0.1')])
    save(tmp_path / 'predictions_decimer.csv', [candidate('DECIMER', 'CCN', '')])
    audit = validate(tmp_path)
    assert {r['status'] for r in audit} == {'accepted', 'low_confidence'}
    assert combine(audit)[0]['tool'] == 'DECIMER'


def test_disagreement_and_preference(tmp_path):
    save(tmp_path / 'predictions_molscribe.csv', [candidate()])
    save(tmp_path / 'predictions_decimer.csv', [candidate('DECIMER', 'CCN', '')])
    selected = combine(validate(tmp_path))
    assert selected[0]['tool'] == 'MolScribe'
    assert selected[0]['disagreement']


def test_markush_sample(tmp_path):
    stage3 = ROOT / 'stage3_recognition/examples/sample_output/markush_sample'
    answer = tmp_path / 'truth.csv'
    expected = ROOT / 'stage3_recognition/examples/sample_input/markush_sample/expected.csv'
    with expected.open() as handle:
        rows = list(csv.DictReader(handle))
    save(answer, [dict(paperID='markush_paper', canonical_SMILES=r['smiles']) for r in rows])
    summary = run(stage3, tmp_path / 'out', answer)
    assert summary['combined_rows'] == 4
    with (tmp_path / 'out/evaluation.csv').open() as handle:
        total = list(csv.DictReader(handle))[-1]
    assert (total['tp'], total['fp'], total['fn']) == ('4', '0', '0')


def test_missing_paper_duplicates_and_unknown(tmp_path):
    answer = tmp_path / 'truth.csv'
    save(answer, [dict(paperID='1', canonical_SMILES='CCO'),
                  dict(paperID='2', canonical_SMILES='CCN')])
    rows = [dict(paper_id='1', canonical_smiles='CCO')] * 2
    rows.append(dict(paper_id='unknown', canonical_smiles='CC'))
    results = evaluate(rows, answer)
    assert results[-1]['tp'] == 1
    assert results[-1]['fn'] == 1
    assert results[-1]['recall'] == 0.5
    assert results[-2]['status'] == 'no_answer_key'


def test_invalid_truth_rejected(tmp_path):
    answer = tmp_path / 'truth.csv'
    save(answer, [dict(paperID='1', canonical_SMILES='nonsense')])
    with pytest.raises(ValueError, match='Invalid answer-key'):
        evaluate([], answer)


def test_invalid_and_failed_markush(tmp_path):
    save(tmp_path / 'predictions_molscribe.csv', [candidate(smiles='invalid')])
    save(tmp_path / 'markush_compounds.csv', [dict(paper_id='1', page='1', crop_id='c1',
         compound_label='', smiles='*C', method='scaffold_only:no_table')])
    audit = validate(tmp_path)
    assert {r['status'] for r in audit} == {'invalid_smiles', 'unresolved_markush'}
    assert combine(audit) == []


def test_empty_file_schema_and_duplicate_checks(tmp_path):
    path = tmp_path / 'predictions_molscribe.csv'
    path.write_text('nonsense\n')
    with pytest.raises(ValueError, match='missing columns'):
        validate(tmp_path)
    save(path, [candidate(), candidate()])
    with pytest.raises(ValueError, match='Duplicate'):
        validate(tmp_path)
