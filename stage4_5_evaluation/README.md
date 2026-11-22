# Stages 4–5: Validation and Evaluation

Author: Hinal Sachpara

Validates stage 3 SMILES, combines recognition and Markush outputs, and scores
unique structures against the course answer key.

## Setup and usage

Run from the repository root with Python 3.10+:

```bash
python3 -m venv .venv-evaluation
source .venv-evaluation/bin/activate
pip install -r stage4_5_evaluation/requirements.txt

python -m stage4_5_evaluation.cli \
  --stage3 stage3_recognition/examples/course_run_output \
  --out out/stage45
```

For scoring, copy `ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv` from the course
`Dataset_2.zip` into `data/`, then run:

```bash
python -m stage4_5_evaluation.cli \
  --stage3 stage3_recognition/examples/course_run_output \
  --answer-key "data/ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv" \
  --out out/stage45
```

Options: `--min-confidence` defaults to `0.2`; `--keep-fragments` disables
largest-fragment cleanup. Paths are passed through the CLI or Python API:

```python
from stage4_5_evaluation.pipeline import run

summary = run('out/stage3', 'out/stage45', answer_key='data/ground_truth.csv')
```

## Inputs and outputs

Reads every `predictions_*.csv` and optional `markush_compounds.csv` in the input
folder. All rows use `paper_id,page,crop_id` to identify the source crop.

- Recognition: `tool,predicted_smiles`, with optional `confidence,error`.
- Markush: `smiles,method`, with `compound_label` for individual compounds.
- Answer key: `paperID,canonical_SMILES`; paper IDs must match the predictions.

Missing required columns, malformed confidence values, duplicate prediction
identities, and invalid answer-key SMILES raise errors.

| Output | Contents |
|---|---|
| `validated_predictions.csv` | All input rows, raw/canonical SMILES, status, confidence, errors, source file |
| `combined_predictions.csv` | Selected predictions and expanded compounds, with disagreement flags |
| `evaluation.csv` | Per-paper TP, FP, FN, precision, recall, F1, and micro totals; requires answer key |
| `summary.json` | Counts and processing settings |

## Processing

1. Parse and sanitize SMILES with RDKit. Keep invalid predictions and recognition
   errors in the audit, but exclude them from combined results.
2. Keep the fragment with the most heavy atoms; resolve ties lexically. This
   removes stray fragments reported by stage 3. Use `--keep-fragments` for salts
   or mixtures where all fragments matter.
3. Remove atom maps and dummy-atom isotope labels so `[1*]` and `[*:1]` compare
   equally. Preserve stereochemistry and isotopes on real atoms. Apply the same
   normalization to predictions and answers.
4. Filter predictions below the confidence threshold. Allow missing confidence
   because DECIMER may not provide it. The default follows stage 3's recommendation.
5. Prefer accepted MolScribe readings, then DECIMER. Flag disagreements and retain
   both readings in the audit. Confidence is not compared across tools.
6. Retain successful `enumerated_` Markush compounds in place of recognition
   predictions for their crop. Exclude failed expansions and unresolved scaffolds.

RDKit performs chemical parsing and canonicalization. This module uses no
pretrained model, training, transfer learning, image preprocessing, or augmentation.

Scoring uses unique canonical structures within each paper. TP is the intersection
of predictions and answers; FP is predicted-only; FN is answer-only.
Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
Zero denominators return 0. Duplicate structures count once. Papers with no
predictions still incur FN. Papers absent from the answer key are marked
`no_answer_key` and excluded from micro totals. The `__micro__` row uses summed
counts across scored papers.

## Tests and results

```bash
python -m pytest stage4_5_evaluation/tests -q
python -m stage4_5_evaluation.examples.run_sample
```

**8 tests passed** with Python 3.12 and RDKit 2026.09.2. Coverage includes
normalization, stereochemistry, fallback, disagreements, Markush integration,
invalid inputs, duplicate handling, and missing-paper scoring.

The sample uses stage 3's saved Markush predictions and
`stage3_recognition/examples/sample_input/markush_sample/expected.csv`.
All four compounds match: **4 TP, 0 FP, 0 FN**. Outputs and the adapted answer
key are in `examples/sample_output/`.

Course results are in `examples/course_validation_output/`. The 66 input rows
include recognition and Markush entries for overlapping crops. Validation keeps
29 rows, representing 28 unique structures within papers. The 222 answer-key
rows normalize to 220 unique structures within papers.

| Confidence threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 (baseline) | 17 | 11 | 203 | 60.7% | 7.7% | 13.7% |
| 0.85 (exploratory) | 17 | 2 | 203 | 89.5% | 7.7% | 14.2% |
| 0.90 (exploratory) | 14 | 1 | 206 | 93.3% | 6.4% | 11.9% |

The baseline audit contains 4 invalid SMILES, 7 low-confidence predictions,
14 unresolved Markush rows, and 12 unresolved scaffolds. Course inputs contain
MolScribe and Markush outputs; DECIMER fallback is tested with controlled inputs.

The threshold comparison uses the same answer key and is exploratory. A separate
validation/test split is needed to assess a chosen threshold independently.
The baseline remains 0.2. Reproduce `examples/threshold_analysis.csv` with:

```bash
python -m stage4_5_evaluation.examples.analyze_thresholds \
  --stage3 stage3_recognition/examples/course_run_output \
  --answer-key "data/ground_truth_SMILES_confirmed_1,2,4,6,16,19.csv" \
  --out out/threshold_analysis.csv
```

## Files and limitations

- `chemistry.py`: SMILES normalization.
- `pipeline.py`: validation, combination, scoring, and CSV output.
- `cli.py`: command-line entry point.
- `tests/`: chemistry and integration tests.
- `examples/`: sample runner, threshold analysis, and saved results.

Valid SMILES can still describe the wrong molecule. The combination rule cannot
resolve two plausible but different readings. Fragment cleanup can remove useful
information, and unresolved scaffolds reduce coverage. No tautomer, protonation,
or charge standardization is applied. These are compound-set scores, not
bounding-box or paired-image scores.
