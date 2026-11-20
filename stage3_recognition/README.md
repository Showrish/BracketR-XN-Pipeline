# Stage 3: Recognition + Markush Branch

**Owner:** Showrish Reddy Sampath  ·  **Pipeline:** BracketR XN (molecular structures from scientific PDFs → SMILES)

Stage 3 takes the crops that passed the stage 1–2 gate and turns each one into SMILES:

1. **Recognition (3a).** Every crop labelled `molecule` is read by a pretrained model: MolScribe as the main reader and DECIMER as a second opinion. Each tool writes its own predictions file.
2. **Markush branch (3b).** A crop labelled `markush` (a scaffold with variable groups R1, R2 …) is read as a scaffold, its R-group placeholders are fixed, the paper's R-group table is found, and every compound in the table is built as a full molecule. A `molecule` crop whose reading turns out to contain attachment points is sent here too, because the gate cannot see R-groups that only appear in a separate table.

```
stage 1-2 manifest.csv ──► keep molecule + markush ──► prepare crop ──► 3a  MolScribe / DECIMER ──► predictions_<tool>.csv
   + crops/                 handoff.py                  preprocess.py      recognizers/
                                                                       └─► 3b  fix [R] ─► find R-group table ─► build compounds ──► markush_compounds.csv
                                                                              markush/attachments.py  rgroup_table.py  substituents.py, enumerate.py
```

## Inputs and outputs

**Input:** the folder written by stage 1–2 (`manifest.csv` + `crops/`). Columns used: `paper_id, crop_id, crop_path, source_pdf, page, dpi, label, markush_evidence`. Optional: the source PDFs (`--pdf-dir`), used to read R-group tables, and a CSV of R-group tables (`--rgroup-table`) for tables that are images or were checked by hand.

**Output** (in the folder passed as `--out`):

| Path | Content |
|---|---|
| `predictions_<tool>.csv` | one row per `molecule` crop and tool → **stage 4 reads these** |
| `markush_compounds.csv` | one row per compound built from a scaffold, or one `scaffold_only` row when none could be built → **stage 4 reads this** |
| `markush_scaffolds.csv` | audit of every scaffold: raw reading, fixed scaffold, table found, outcome |
| `prepared/paper_<id>/pNNN_cNN.png` | the exact images the models read (`save_prepared: false` to skip) |
| `summary.json` | counts per tool, errors, mean confidence, Markush outcomes |

`predictions_<tool>.csv`: `paper_id, page, crop_id, tool, predicted_smiles, confidence` (blank if the tool gives none), `runtime_sec, error` (blank unless the model failed on that crop).

`markush_compounds.csv`: `paper_id, page, crop_id, compound_label` (e.g. `12a`), `smiles, method`. `method` is `enumerated_text_layer` or `enumerated_csv` for a built compound, `failed:<reason>` for a table row that could not be built, or `scaffold_only:<reason>` when no compound could be built (`no_table`, `no_attachment_points`, `attachment_mismatch`, `invalid_scaffold`, `recognition_error`).

These are the hand-off formats agreed in the Assignment 1.4 plan. `crop_id` is stage 1–2's id (`p003_c02`), so every row joins back to its box on the page.

## How to use

```bash
# one-time setup: one conda environment per recognizer (they pin conflicting frameworks)
conda env create -f stage3_recognition/envs/stage3_molscribe.yml      # MolScribe + RDKit + tests
conda env create -f stage3_recognition/envs/stage3_decimer.yml        # optional second opinion
sudo apt-get install -y poppler-utils                                  # pdftotext, for R-group tables

# run from the repository root, after stage 1-2 has written out/stage12/
conda run -n xn-stage3-molscribe python -m stage3_recognition.cli \
    --manifest out/stage12/manifest.csv --out out/stage3 --pdf-dir path/to/pdfs
conda run -n xn-stage3-decimer python -m stage3_recognition.cli \
    --manifest out/stage12/manifest.csv --out out/stage3 --tools decimer --no-markush
```

From Python (this is how `run_pipeline.py` can call it):

```python
from stage3_recognition import load_config, run_stage3

summary = run_stage3("out/stage12/manifest.csv", "out/stage3", load_config(), pdf_dir="papers")
```

Google Colab: `notebooks/run_stage3_colab.ipynb` sets up the environment, runs the tests and the sample, and can run the whole chain (stage 1–2 then stage 3) on the course papers.

No paths or credentials are hard-coded. Every location is a command-line argument or function parameter, and every tunable value lives in `config.py` / `default_config.yaml`.

## Code navigation

| File | Role |
|---|---|
| `handoff.py` | reads the stage 1–2 manifest (resolves crop and PDF paths) and writes the hand-off CSVs with fixed columns |
| `preprocess.py` | RGB conversion on a white background and the white margin |
| `recognizers/base.py` | `Recognizer` interface: timing, and turning a model failure into an `error` value |
| `recognizers/molscribe_recognizer.py`, `decimer_recognizer.py` | the two pretrained models, imported only when used |
| `markush/attachments.py` | `[R]`, `[R1]` placeholders → SMILES attachment points `[*]`, `[1*]` |
| `markush/rgroup_table.py` | finds R-group tables in the PDF text layer (or a CSV) |
| `markush/substituents.py` | table cell → SMILES fragment (abbreviations, alkyl formulas, short condensed formulas) |
| `markush/enumerate.py` | joins scaffold and fragments with RDKit to build each compound |
| `pipeline.py` | `run_stage3`: ties it together and writes `summary.json` |
| `cli.py`, `config.py`, `default_config.yaml` | command line and settings |
| `tests/` | unit and end-to-end tests (a stand-in recognizer, so they need no model or GPU) |
| `examples/` | sample inputs with known answers, `run_sample.py`, `check_tables.py`, `check_course_papers.py` |
| `envs/`, `requirements.txt`, `notebooks/` | environments, dependencies, Colab notebook |

## Technical methods and choices

**Pretrained models (transfer learning).** Both recognizers are pretrained models used as-is. **No fine-tuning and no transfer learning on our data are done in this module.**
- *MolScribe* (Qian et al., 2023), checkpoint `swin_base_char_aux_1m680k.pth` from Hugging Face: a Swin-B image encoder with a decoder that predicts atoms and bonds as a graph, trained by its authors on about 1M rendered PubChem molecules and 680K USPTO patent images. It is the main reader because it was clearly the strongest in our Phase 1 benchmark (0.46 exact match on 100 BioVista images against 0.30 for DECIMER, McNemar p = 0.0015), and it returns a confidence score that stage 4 can use.
- *DECIMER* (Rajan et al., 2023): an image-to-SMILES transformer. It is weaker overall but was slightly better on chiral images in Phase 1 (0.53 against 0.47), so it runs as a second opinion and stage 4 can flag crops where the two disagree.
- *Why no fine-tuning:* the only labelled real data we have is the course answer key (220 structures from six papers). It is our test set, so training on it would make our scores meaningless, and it is far too small to fine-tune a model of this size.

**Preprocessing** (`preprocess.py`). (1) Transparent pixels are flattened onto white and every crop becomes 3-channel RGB, which both models expect. (2) A white margin of 0.05 in (15 px at 300 dpi, converted with each crop's dpi) is added, because stage 1–2 cuts boxes tightly and an atom label that touches the edge of the image is easy to lose. The margin guarantees some white space around every drawing before the model's own preprocessing, and it changes nothing for a model that trims white borders itself. No resizing or normalisation is done here: each model resizes and normalises to its own input format internally, and doing it twice would only add blur. The prepared images are saved, so what the model saw can be checked.

**Data augmentation.** None. Augmentation is a training-time technique and this module does no training. Robustness comes from the pretrained models, whose authors trained them with image and molecule augmentation.

**Markush branch (3b).**
- *Placeholders.* DECIMER writes a variable group as `[R]` or `[R1]`, which RDKit rejects. In Phase 1 this caused 10 of DECIMER's 12 invalid outputs on Markush images. They are rewritten as attachment points (`[R1]` → `[1*]`), keeping the number.
- *Finding the table.* The page's text layer is read with `pdftotext -layout` (poppler, which stage 1–2 already uses), on the scaffold's page and then the next page. A header is any line with R-group column names (R1, R2, R, X, Ar …). Rows that start with a compound label (5a, 12, 10b …) are read until the table ends. A header that repeats its label word is read as two tables printed side by side, a note after a label such as `1 (VMF)` is skipped, and rows need short, link-free cells and a table at least two rows, so running text is not mistaken for a table. A CSV (`compound_label, R1, R2, …`, optionally per `paper_id`/`crop_id`) overrides the text layer.
- *Cells to structures.* A transparent dictionary of common groups and names (Me, OMe, CF3, phenyl, cyclo-prop …), straight-chain formulas (`nC4H9`), and short condensed formulas built from known groups (`CH2-CH3`, `(CH2)2-CF3`, `CH2-C(O)-CH3`, `CH2-(p)F-phenyl`). Anything else, such as the ambiguous `ethylphenyl`, is reported as unknown, not guessed.
- *Building compounds.* Each attachment point is paired with a table column, by number when the scaffold has numbered points (`[1*]` → R1), otherwise by order when the counts match (reported as `ok_assumed_column_order`). RDKit's `molzip` joins each fragment to the scaffold at matching atom-map numbers. An `H` cell turns the attachment point itself into a hydrogen, which RDKit removes.

**Errors are recorded, not hidden.** A crop the model cannot read gets an `error` value and the run continues. A compound that cannot be built gets a `failed:<reason>` row, so stage 4 can see what was lost and why.

## Testing evidence

All results below are saved in this folder: `examples/sample_output/` (sample run), `examples/course_run_output/` (course papers) and the test suite in `tests/`.

**Tests.** `conda run -n xn-stage3-molscribe python -m pytest stage3_recognition/tests -q` runs 30 tests with a stand-in recognizer, so they need no model download or GPU. They cover reading stage 1–2's real sample manifest, the output column order, the margin and RGB conversion, the placeholder fix on real DECIMER outputs from Phase 1, substituent conversion, table reading (a real PDF text layer, side-by-side tables, columns before R, notes after labels, rejection of running text), building compounds (numbered, by order, H cells, unknown cells, mismatches) and three end-to-end runs. In the environment used to write the module, which has no RDKit, 24 passed and the 6 that build molecules were skipped. The compound-building code also ran for real in the sample run below.

**Sample run with the real model** (`python -m stage3_recognition.examples.run_sample`, Colab T4 GPU, MolScribe checkpoint `swin_base_char_aux_1m680k.pth`). Two inputs whose answers are known:

| Input | Expected | Stage 3 output | Result |
|---|---|---|---|
| Stage 1–2 sample, crop `p001_c02` | phenol | `Oc1ccccc1` (confidence 0.88) | exact match |
| Stage 1–2 sample, crop `p001_c03` | naphthalene | `c1ccc2ccccc2c1` (0.91) | exact match |
| Stage 1–2 sample, crop `p001_c05` | butanamide | `CCCC(N)=O` (0.92) | exact match |
| Stage 1–2 sample, Markush crop `p002_c02` (R1, R2, X, no table) | flagged, nothing built | read as `*c1ccc(C)c[c]1[Rf]`, `scaffold_only:no_table` | as expected |
| Markush sample scaffold (R1, R2) + table of 5a–5d | 4 compounds | scaffold read as `[1*]c1ccc([2*])cc1`, table found in the text layer, 4 compounds built | **4 / 4 exact** |

MolScribe turned the R1 and R2 labels into numbered attachment points, so each column was paired with its point by number, and the four compounds came out as chlorobenzene, 4-fluorotoluene, 4-bromoanisole and 4-(trifluoromethyl)phenol. The one Markush drawing with no table was passed on as a flagged scaffold instead of a guess. That drawing also shows a weakness: the X label was read as the element Rf.

**Real input 1: the table reader on the six course papers** (`examples/check_tables.py`; run while writing the module, and again in Colab with identical results, saved in `examples/course_run_output/table_check.json`):

| Paper | Tables found | Rows | Cells turned into structures |
|---|---|---|---|
| 1, 2, 4, 6, 16 | none | – | – |
| 19, page 4 | Table 1 (columns Compound, A, B, R …) | 6 (compounds 1–6) | 6 / 6 (`-H`, `-OCH3`) |
| 19, page 5 | two linker tables printed side by side | 10 (1, 10a–10i) | 0 / 10 (linkers such as `SO2`, `(CO)-(CO)` join two parts of the scaffold, see Limitations) |
| 19, page 6 | two tables printed side by side | 19 (1, 12a–12r) | 18 / 19 (all but the ambiguous `-ethylphenyl`) |

Before the side-by-side and plausibility rules were added, the reader also "found" one-row tables in the running text of papers 2 and 16 (a page number followed by a DOI) and read only the left half of the page 6 table. Both problems are covered by tests now.

**Real input 2: the whole chain on the course papers** (Colab: stage 1–2 on the six PDFs, then stage 3, then `examples/check_course_papers.py`). A structure counts only if its canonical SMILES exactly matches the paper's answer key, each structure once. This checks what stage 3 produces on real input; it is not the group's official scoring (stage 5).

| Paper | Crops from stage 1–2 | Valid SMILES | Structures found | In answer key |
|---|---|---|---|---|
| 1 | 12 | 12 | 9 | 17 |
| 2 | 12 | 12 | 3 | 34 |
| 4 | 10 | 8 | 0 | 60 |
| 6 | 2 | 2 | 0 | 1 |
| 16 | 9 | 9 | 1 | 4 |
| 19 | 7 | 5 | 3 | 104 |
| **Total** | **52** | **48** | **16** | **220** |

How to read this:
- **Recognition is strong when the crop is a clean molecule.** In paper 1, 9 of 12 crops matched the answer key exactly. All 9 correct readings had confidence of about 0.90 or higher, and the 3 wrong ones had 0.18, 0.45 and 0.54.
- **Confidence separates junk well.** All 10 readings below 0.2 confidence are long carbon chains or broken fragments, the signature of an image that is not a molecule (a chart or table that passed the gate) or of a misread crop. Stage 4 can use this.
- **The ceiling is set by the crops.** Stage 1–2 kept 52 crops for 220 molecules, so most answer-key structures never reached stage 3. Paper 19's R-group tables (pages 4–6) were read correctly, but no scaffold crop came from those pages, so 3b had nothing to expand there.
- **The Markush routing works on real crops.** 14 `molecule` crops came back with attachment points (for example, a phenyl ring with R1–R5 in paper 2), were sent to 3b, and were passed on as flagged scaffolds, because none had an R-group table in the text layer of its page or the next one.
- **Two fixes for stage 4–5, found in this run.** (1) 17 readings have more than one fragment (e.g. `.[He]`, `.[Co]`, `CC.`), and keeping only the largest fragment would turn some into matches: paper 19's scaffold B9 was read correctly apart from a stray `[He]`. (2) The answer key writes attachment points as `[*:1]` while MolScribe writes `[1*]`, so scaffold entries should be compared with their labels removed.

## Limitations (please read before trusting the numbers)

- **Few scaffolds reach 3b on real papers.** Stage 1–2 labelled no course-paper crop `markush`, and its 52 crops did not include the scaffolds drawn above paper 19's tables, so on the course papers 3b could only flag scaffolds, not build compounds.
- **Scaffolds without attachment points cannot be expanded.** In Phase 1, MolScribe left the attachment point out of 31 of 43 Markush images, often drawing a carbon instead. Those crops come out as `scaffold_only:no_attachment_points`.
- **Image-only tables are not read.** The table reader needs a text layer. Tables pasted as images need the CSV option (OCR is not implemented).
- **Linkers and variable ring atoms are not supported.** A column whose groups join two parts of the scaffold (paper 19, page 5) or replace a ring atom (columns A and B on page 4) needs a different scaffold model. Those rows are reported, not built.
- **Several tables on one page:** the largest one is used, which can pair a scaffold with the wrong table. A CSV avoids this.
- **Table layout:** each header cell must be one word between R columns (a two-word header cell such as `IC50 (nM)` is fine after the last R column), and table cells must not contain spaces.
- **Pairing by order** (when the scaffold's attachment points carry no numbers) assumes the points appear in the same order as the table columns. Such compounds are marked `ok_assumed_column_order`.
- **Substituents** outside the dictionary and the simple formula patterns, including IUPAC names, are reported as unknown.
- **Misread labels.** MolScribe sometimes reads a text label as an element (an X label as `Rf`, stray `[He]` or `[Co]` fragments). These are passed on unchanged, so stage 4 can clean or flag them.
- **The DECIMER adapter** is a thin wrapper around `predict_SMILES` and was not run for these results (MolScribe only).

## Integration notes for the other stages

- Stage 4 should read every `predictions_<tool>.csv` plus `markush_compounds.csv`, and join on `paper_id, page, crop_id`. A scaffold that was expanded also appears in `predictions_molscribe.csv` with its attachment points (`*`), so stage 4 should drop predictions containing `*` once the compounds are built.
- Run stage 3 once per environment into the same `--out` folder. The MolScribe run also does 3b.
- From the course-paper run: keep the largest fragment of each SMILES before scoring (17 of 52 readings have more than one fragment), use MolScribe's `confidence` to flag or drop readings below about 0.2, and compare attachment points with their labels removed (`[1*]` here, `[*:1]` in the answer key).
- If the group changes column names, only `handoff.py` needs to change.

## References

Gulluoglu, H. S. A., et al. (2026). Practical use of advanced AI frameworks on real-life scientific problems: Three case studies [Preprint]. bioRxiv. https://doi.org/10.64898/2026.06.23.734132

Landrum, G., et al. (2026). RDKit: Open-source cheminformatics [Computer software]. https://www.rdkit.org

Qian, Y., Guo, J., Tu, Z., Li, Z., Coley, C. W., & Barzilay, R. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. Journal of Chemical Information and Modeling, 63(7), 1925–1934. https://doi.org/10.1021/acs.jcim.2c01480

Rajan, K., Brinkhaus, H. O., Agea, M. I., Zielesny, A., & Steinbeck, C. (2023). DECIMER.ai: An open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications. Nature Communications, 14(1), 5045. https://doi.org/10.1038/s41467-023-40782-0
