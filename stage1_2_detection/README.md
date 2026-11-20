# Stage 1–2: Structure Detection + Molecule Gate

**Owner:** Jirapa Bangvak  ·  **Pipeline:** BracketR XN (molecular structures from scientific PDFs → SMILES)

Stage 1–2 takes a paper (PDF) and answers two questions for every drawing on every page:

1. **Where is it?** Render each page and propose candidate regions (stage 1).
2. **Is it a molecule?** A gate labels each region `molecule`, `markush` (a drawing with variable groups such as R1/R2) or `not_molecule` (text, chart, table, arrow, photo …) and logs *why* (stage 2).

Only accepted crops go on to stage 3 (recognition). This matters because the course paper (Gulluoglu et al., 2026) found the main bottleneck is figure segmentation and telling molecules from non-molecules, not the recognition model.

```
PDF ──► render pages ──► detect regions ──► molecule gate ──► split merged panels ──► crops/ + manifest.csv
        (300 dpi)        detect.py          gate.py           split.py               pipeline.py, manifest.py
        render.py
```

## Inputs and outputs

**Input:** one or more PDFs (or single images, treated as one-page documents).

**Output** (in the folder you pass as `--out`):

| Path | Content |
|---|---|
| `manifest.csv` | one row per candidate region (columns below) |
| `crops/paper_<id>/pNNN_cNN.png` | accepted crops (`molecule` + `markush`) → **stage 3 reads these** |
| `rejected/paper_<id>/pNNN_cNN.png` | rejected crops, kept so the gate can be audited (`--no-rejected` to skip) |
| `overlays/paper_<id>/pNNN.png` | page images with boxes: green = molecule, blue = markush, red = rejected (`--overlay`) |
| `summary.json` | counts per paper/label and rejection reasons |

`manifest.csv` columns:

| Column | Meaning |
|---|---|
| `paper_id`, `crop_id` | paper name and region id, e.g. `p003_c02` = page 3, region 2 |
| `crop_path` | image path relative to the output folder |
| `source_pdf`, `page` | where it came from (page is 1-based) |
| `x`, `y`, `width`, `height` | box on the rendered page, in pixels (empty if a backend cannot locate it) |
| `dpi` | render resolution the box refers to |
| `label` | `molecule` → stage 3a · `markush` → stage 3b · `not_molecule` → dropped |
| `gate_score` | 0–1 confidence that it is a molecule drawing |
| `reason` | why: `ok(bond_lines=11, rings=1)`, `label_heavy_figure(...)`, `chart_or_table_axes`, … (`[split]` = cut out of a merged panel) |
| `markush_evidence` | R-group tokens found near the drawing (e.g. `R1 R2 X`) |
| `detector` | `classical` or `decimer` |

The position columns (`crop_path, source_pdf, page, x, y, width, height`) follow the convention of the course example pipeline; the crop folder layout matches the one used by `run_decimer.py`.

## How to use it

```bash
# one-time setup (Python 3.10+; poppler is a system package)
sudo apt-get install -y poppler-utils        # macOS: brew install poppler
pip install -r stage1_2_detection/requirements.txt

# run from the repository root
python -m stage1_2_detection.cli --pdf-dir path/to/pdfs --out path/to/output --overlay
python -m stage1_2_detection.cli --pdf paper.pdf --out out --dpi 200 --config my_config.yaml
```

From Python (this is how `run_pipeline.py` can call it):

```python
from stage1_2_detection import load_config, run_stage1_2, read_manifest, select_labels

rows = run_stage1_2("paper.pdf", "out", load_config())            # list of Candidate
recognise = select_labels(rows, ("molecule",))                    # → stage 3a
markush   = select_labels(rows, ("markush",))                     # → stage 3b
# or later: rows = read_manifest("out/manifest.csv")
```

Google Colab:

```python
!apt-get -qq install -y poppler-utils
!pip -q install opencv-python-headless scikit-image pdf2image pyyaml
# after cloning the group repo and cd-ing into it:
!python -m stage1_2_detection.cli --pdf-dir "$PDF_FOLDER" --out "$OUT_FOLDER" --overlay
```

No paths or credentials are hard-coded; every location is a command-line argument or function parameter. All tunable numbers live in `config.py` / `default_config.yaml` (sizes are in inches, so they do not depend on the dpi).

## Code navigation

| File | Role |
|---|---|
| `render.py` | PDF page → RGB array, one page at a time (poppler via `pdf2image`) |
| `textlayer.py` | words + positions from the PDF text layer (`pdftotext -bbox`); used by the gate; attaches detached atom labels to a box |
| `detect.py` | candidate regions: `classical` (default) and `decimer` backends |
| `gate.py` | the molecule gate: stroke/ring/text/colour analysis, decision rules, Markush evidence |
| `split.py` | splits a merged panel (molecules glued to charts) into separate drawings and re-joins pieces of one molecule |
| `pipeline.py` | `run_stage1_2` / `run_many`: ties it together, saves crops, overlays, summary |
| `manifest.py` | `Candidate` dataclass, CSV read/write, `select_labels` |
| `cli.py`, `config.py`, `default_config.yaml` | command line and settings |
| `tests/` | 43 tests (42 pass, 1 expected failure) + the synthetic-paper generator; `examples/` sample input/output |

## Technical methods and choices

**Preprocessing.** Pages are rendered at 300 dpi (RGB) and converted to grayscale; ink is separated from background with Otsu thresholding. Sizes are expressed in inches and converted with the dpi. No resizing, normalisation or colour change is applied to the saved crops, so stage 3 receives the page pixels unchanged.

**Detection (classical, default).** Ink is dilated by 0.07 in so strokes of one drawing (bonds, atom labels) merge into one blob; connected components give boxes, which are padded by 0.06 in. Boxes smaller than 0.3 in or larger than 60 % of the page are dropped. For vector PDFs, short text-layer tokens (≤ 4 characters, e.g. `X`, `OH`) lying just outside an accepted box are pulled in, because labels are often drawn slightly apart from their bond. *Why classical:* it needs no model download or GPU, runs on any machine, is fast (≈ 0.5 s per synthetic page) and is easy to explain and debug.

**Detection (optional DECIMER Segmentation backend).** `--backend decimer` uses the pretrained DECIMER Segmentation Mask R-CNN (Rajan et al., 2023), the same tool used for my benchmark runs. It returns crops only, so each crop is located on the page by template matching to recover its box. **Pretrained, used as-is: no fine-tuning, no transfer learning and no data augmentation are done in this module.** (The adapter is unit-tested with a stand-in segmenter; the real model needs Python 3.10 + TensorFlow and was not run in the environment used to write this.)

**Gate (stage 2).** A transparent rule-based classifier rather than a trained model, because (a) we have no labelled non-molecule examples yet and (b) every rejection can then be explained in the manifest. Features: straight strokes (skeleton + Hough lines, de-duplicated) at ≥ 0.05 in (bonds in journal figures are about 0.09 in); their alignment to multiples of 30° (typical bond angles); closed 5–7-sided convex holes (rings); letters lined up in text rows; letters per stroke (labelled charts have many, molecules few); share of coloured pixels (drawn molecules are black); joints that bend only slightly (curves, circles); ink ratio and mid-tone share (photos, heat maps); long axis-like lines (charts, tables); and prose words from the PDF text layer. Hard rules reject first (text, photo/filled area, coloured figure, label-heavy figure, chart/table axes, curves, too few bond lines, only horizontal/vertical lines); the rest are scored 0–1 and kept if the score ≥ 0.5.

**Splitting merged panels (stage 1c).** In real papers a molecule often sits so close to a chart that pixel merging puts both in one crop. In my first run on a real *J. Med. Chem.* figure, three structures and four scatter plots came out as one accepted box. Any large or letter-rich crop (and any big crop rejected as a chart or diagram) is therefore re-examined: blobs are found again at a 0.02 in gap, small blobs (atom labels) are attached to the nearest large one, each piece is gated, and pieces of one molecule are re-joined when their *ink* (not their boxes) comes within 0.05 in. A piece that sits inside a rejected piece, such as a data line inside chart axes, is rejected (`inside_other_figure`). If no molecule is found inside, the region is kept as one rejected crop, so the manifest does not fill up with fragments. Text blocks and photos are never split. Accepted drawings with R-group tokens nearby (`R1`, `R'`, `Ra`, or two of `X Y Z Ar`) are labelled `markush`. If a figure has no text layer, `use_ocr_fallback: true` looks for those tokens with Tesseract OCR.

## Testing evidence

`python -m pytest stage1_2_detection/tests -q` → **42 passed, 1 expected failure** (a documented limitation, see below).

**Sample test case** (`python -m stage1_2_detection.examples.run_sample`). Input: `examples/sample_input/sample_paper.pdf`, a synthetic 2-page paper with running text, four line-art molecules (one is a Markush drawing with R1/R2/X), a reaction arrow, a bar chart, a line chart and a table. Output: `examples/sample_output/` (manifest, crops, rejected crops, overlays). Page 1 overlay: `examples/sample_output/overlays/paper_sample_paper/p001.png`.

| Region | Truth | Gate label | Reason |
|---|---|---|---|
| p001_c01 | text paragraph | not_molecule | text_block_text_layer(words=66) |
| p001_c02 | molecule (phenol-like) | **molecule** | ok(bond_lines=16, rings=1) |
| p001_c03 | molecule (naphthalene-like) | **molecule** | ok(bond_lines=22, rings=2) |
| p001_c04 | bar chart | not_molecule | no_diagonal_bond_angles |
| p001_c05 | molecule (ring-free chain) | **molecule** | ok(bond_lines=13, rings=0) |
| p002_c01 | table | not_molecule | text_block_rows(letters=64) |
| p002_c02 | Markush drawing | **markush** | ok(…); evidence `R1 R2 X` |
| p002_c03 | line chart | not_molecule | chart_or_table_axes |
| p002_c04 | text paragraph | not_molecule | text_block_text_layer(words=38) |

Result: **4 / 4 molecule drawings found (IoU ≥ 0.5) and 0 / 6 non-molecule items accepted**; the Markush drawing carries the `markush` label. The reaction arrow with its conditions text is smaller than the 0.3 in minimum and is dropped at the detection step (if it were kept, the gate would reject it as `too_few_bond_lines`). The same 4/4, 0 false-accept result holds at 100, 150, 200, 300 and 400 dpi. On a JPEG-compressed, text-layer-free (raster-only) version of the sample, all 4 are still found, but the Markush label needs the OCR fallback.

**Real-figure check.** Two figures from the course papers (*J. Med. Chem.* kinase-inhibitor figures), taken from my first real run, where each came out as one merged box:

| Figure | Content | Before the split step | After |
|---|---|---|---|
| 1 | LXH254, AMG925, Ponatinib + 4 scatter plots | 1 box around everything | **3 separate molecule crops**; all 4 plots rejected (`label_heavy_figure`, `chart_or_table_axes`) |
| 2 | GW296115X, GW830263A, TPKI-39 + kinome trees + 2 plots | 1 box around everything | **3 separate molecule crops**; trees, plots and labels rejected |

These were rebuilt from screenshots of the overlay (figure 2 at only about 120 dpi), so they show the method works on real layouts. They are not a full evaluation: the thresholds were adjusted using these two figures. `tests/test_split.py` reproduces the merged-panel case with a drawn molecule and plot.

### Run on the six course papers

Run in Google Colab on the six benchmark PDFs (300 dpi, default settings, `--overlay`), after the splitting fix. All tests passed in Colab (`42 passed, 1 xfailed`).

| Paper | Pages | Regions found | Accepted crops | True molecules (ground truth) | Crops from the course example pipeline (no gate) |
|---|---|---|---|---|---|
| 1 | 13 | 91 | 12 | 17 | 66 |
| 2 | 19 | 96 | 12 | 34 | 78 |
| 4 | 16 | 72 | 10 | 61 | 96 |
| 6 | 25 | 74 | 2 | 1 | 198 |
| 16 | 13 | 81 | 9 | 4 | 62 |
| 19 | 18 | 99 | 7 | 105 | 89 |
| **Total** | 104 | **513** | **52** | 282 | 589 |

Rejection reasons (461 regions): text layer 240, coloured figure 106, text rows 53, only horizontal/vertical lines 21, photo/filled area 17, curved lines 11, chart/table axes 5, label-heavy figure 4, inside another figure 3, low score 1.

How to read this:
- **The gate removes most noise.** 513 regions were cut down to 52 crops. Paper 6, with one true molecule, gives 2 crops instead of the 198 the course example pipeline produced without a gate.
- **Crops are not molecules.** One crop can hold several structures (a grid of compounds), and a molecule can be drawn more than once, so crop counts cannot be compared one-to-one with the ground-truth counts. The real test is the SMILES scoring in stage 4–5.
- **Papers 4 and 19 need a closer look.** They have 61 and 105 true molecules but only 10 and 7 crops. This could be grids of compounds in one crop, compounds defined by R-group tables, or molecules rejected by mistake. The `rejected/` folders, especially `colored_figure` (the second most common reason), should be checked for real structures.
- **No region was labelled `markush`.** Most figures are images without a text layer, so R-group labels are not seen. The next run should set `use_ocr_fallback: true`.
- Overlays were checked by eye. The only miss found is the small TPKI-39 drawing inside a graphical abstract (see Limitations).

## Limitations (please read before trusting the numbers)

- **The tests use a synthetic paper I generated, not the six course papers.** Real figures are messier (anti-aliasing, colour, shading, tightly packed panels), so the thresholds in `GateConfig` are sensible starting values, not tuned values. Run it on the real PDFs and inspect the overlays before relying on it.
- **No localisation ground truth exists** for the real papers (only SMILES), so detection quality on them can only be checked by eye or indirectly through downstream SMILES scores.
- **Known false positive:** a pattern of hexagons (e.g. a honeycomb icon) passes, because the gate looks at geometry, not chemistry (`test_honeycomb_pattern_is_rejected` is marked as an expected failure).
- **Merged panels** are now split (see above), but splitting can still fail. Two molecules whose ink is within 0.05 in stay as one crop, and a molecule with a very long gap inside (e.g. around a large label) may come out in two pieces. Tune `split_gap_in` / `join_gap_in`, or set `split_enabled: false` to compare.
- **Drawings set inside a text block are missed.** Example: the small TPKI-39 drawing in a graphical abstract (about 0.18 in tall) sits inside the abstract's text region, which is rejected as text. Splitting text blocks was tried, but it also accepted a black kinome-tree drawing, so it is off. In that paper the same molecule is found full-size in a figure.
- **Coloured molecules** (structures drawn with coloured atoms or highlighted parts) are rejected by the colour rule; raise `max_color_fraction` for papers that use them.
- **Tuned on little real data:** the real-figure thresholds come from two figures only.
- **Tiny molecules** (smaller than 0.3 in) and **hand-drawn or very low-resolution** drawings are missed or rejected.
- **Markush detection depends on text evidence** (text layer or OCR); a Markush drawing whose R-groups are only in a separate table is labelled plain `molecule`. Reading that table is stage 3b's job.
- **DECIMER backend** is not run in the writing environment (see above).

## Integration notes for the other stages

- Stage 3 should read `manifest.csv` and open `crop_path` for rows labelled `molecule` (recognition) or `markush` (variable-group branch).
- Stage 4–5 can use `page`, `x`, `y`, `width`, `height` to trace each SMILES back to its place in the paper, and `gate_score` / `reason` to explain dropped regions.
- If the group agrees on different column names, only `manifest.py` needs to change.
