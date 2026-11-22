# BracketR-XN-Pipeline

Extract molecular structures from scientific PDFs, recognize SMILES, and evaluate
against known structures.

| Module | Owner | Responsibility |
|---|---|---|
| [stage1_2_detection](stage1_2_detection/README.md) | Jirapa | PDF rendering, segmentation, molecule filtering |
| [stage3_recognition](stage3_recognition/README.md) | Showrish | MolScribe/DECIMER recognition and Markush enumeration |
| [stage4_5_evaluation](stage4_5_evaluation/README.md) | Hinal | SMILES validation, combining predictions, evaluation |

Data flow: PDF → crops + manifest → recognition/Markush CSVs → validated and
combined predictions → paper-level precision, recall, and F1.

Each module README documents its environment, CLI, schemas, and testing evidence.
See the stage 4–5 README to run evaluation on the saved stage 3 outputs without
rerunning recognition models.