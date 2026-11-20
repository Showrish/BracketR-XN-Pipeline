"""Stage 3: recognise molecule crops (3a) and expand Markush scaffolds (3b).

    stage 1-2 manifest.csv
        -> keep crops labelled molecule / markush        (handoff.py)
        -> prepare each crop: RGB + white margin          (preprocess.py)
        -> 3a: every tool reads every molecule crop       (recognizers/)
               -> predictions_<tool>.csv
        -> 3b: scaffold tool reads each Markush crop (and any molecule crop whose
               reading has attachment points), [R] placeholders are fixed,
               the R-group table is found and full compounds are built (markush/)
               -> markush_compounds.csv (+ markush_scaffolds.csv for auditing)
        -> summary.json
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import Stage3Config, load_config
from .handoff import (LABEL_MARKUSH, LABEL_MOLECULE, MARKUSH_COLUMNS, PREDICTION_COLUMNS,
                      CropTask, read_stage2_manifest, write_rows)
from .markush import (RGroupTable, attachment_labels, enumerate_compounds,
                      normalize_attachment_points, table_from_csv, table_from_pdf_text)
from .preprocess import prepare_crop
from .recognizers import Prediction, Recognizer, build_recognizer

LOGGER = logging.getLogger(__name__)
RecognizerFactory = Callable[[str, Stage3Config], Recognizer]
SCAFFOLD_AUDIT_COLUMNS = ["paper_id", "page", "crop_id", "gate_label", "raw_prediction",
                          "scaffold_smiles", "table_source", "table_rows", "outcome"]


def prediction_row(task: CropTask, tool: str, prediction: Prediction) -> Dict:
    return {"paper_id": task.paper_id, "page": task.page, "crop_id": task.crop_id, "tool": tool,
            "predicted_smiles": prediction.smiles, "confidence": prediction.confidence,
            "runtime_sec": prediction.runtime_sec, "error": prediction.error}


def find_rgroup_table(task: CropTask, cfg: Stage3Config,
                      rgroup_csv: Optional[Path]) -> Optional[RGroupTable]:
    """A CSV given by the user wins; otherwise search the PDF text layer."""
    if rgroup_csv is not None:
        table = table_from_csv(rgroup_csv, task.paper_id, task.crop_id)
        if table is not None:
            return table
    if cfg.markush.table_source == "text_layer" and task.source_pdf is not None:
        return table_from_pdf_text(task.source_pdf, task.page, cfg.markush.search_next_page)
    return None


def expand_scaffold(task: CropTask, prediction: Prediction, table: Optional[RGroupTable]
                    ) -> Tuple[List[Dict], Dict]:
    """Rows for markush_compounds.csv and one audit row for one Markush crop."""
    scaffold = normalize_attachment_points(prediction.smiles)
    ids = {"paper_id": task.paper_id, "page": task.page, "crop_id": task.crop_id}
    audit = {**ids, "gate_label": task.label, "raw_prediction": prediction.smiles, "scaffold_smiles": scaffold,
             "table_source": table.source if table else "", "table_rows": len(table.rows) if table else 0}

    if prediction.error:
        outcome = "scaffold_only:recognition_error"
    elif table is None:
        outcome = "scaffold_only:no_table"
    else:
        compounds = enumerate_compounds(scaffold, table)
        built = [c for c in compounds if c.smiles]
        if built:
            audit["outcome"] = f"enumerated({len(built)}/{len(table.rows)})"
            rows = [{**ids, "compound_label": c.label, "smiles": c.smiles or "",
                     "method": f"enumerated_{table.source}" if c.smiles else f"failed:{c.status}"}
                    for c in compounds]
            return rows, audit
        outcome = f"scaffold_only:{compounds[0].status}"

    # Nothing could be built: pass the scaffold on so stage 4 can flag it.
    audit["outcome"] = outcome
    return [{**ids, "compound_label": "", "smiles": scaffold, "method": outcome}], audit


def run_markush_branch(tasks: Sequence[CropTask], predictions: Dict[str, Prediction],
                       cfg: Stage3Config, rgroup_csv: Optional[Path]) -> Tuple[List[Dict], List[Dict]]:
    """Expand every scaffold. ``predictions`` holds the scaffold tool's reading of each crop."""
    rows, audits = [], []
    for task in tasks:
        task_rows, audit = expand_scaffold(task, predictions[task.key], find_rgroup_table(task, cfg, rgroup_csv))
        rows.extend(task_rows)
        audits.append(audit)
    return rows, audits


def has_attachment_points(prediction: Prediction) -> bool:
    return bool(attachment_labels(normalize_attachment_points(prediction.smiles)))


def _tool_summary(rows: List[Dict]) -> Dict:
    confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
    return {"predictions": len(rows),
            "errors": sum(1 for r in rows if r["error"]),
            "empty": sum(1 for r in rows if not r["predicted_smiles"]),
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None}


def run_stage3(manifest: str | Path, out_dir: str | Path, cfg: Optional[Stage3Config] = None,
               tools: Optional[Sequence[str]] = None, pdf_dir: Optional[str | Path] = None,
               rgroup_csv: Optional[str | Path] = None,
               recognizer_factory: RecognizerFactory = build_recognizer) -> Dict:
    """Run stage 3 on a stage 1-2 manifest and write the hand-off files to ``out_dir``."""
    cfg = cfg or load_config()
    tools = list(tools or cfg.tools)
    out_dir = Path(out_dir)
    rgroup_csv = Path(rgroup_csv) if rgroup_csv else None

    tasks = read_stage2_manifest(manifest, pdf_dir)
    missing = [t.key for t in tasks if not t.image_path.is_file()]
    if missing:
        LOGGER.warning("%d crops listed in the manifest were not found and are skipped", len(missing))
        tasks = [t for t in tasks if t.image_path.is_file()]
    molecules = [t for t in tasks if t.label == LABEL_MOLECULE]
    scaffolds = [t for t in tasks if t.label == LABEL_MARKUSH]
    LOGGER.info("%d molecule crops, %d Markush crops", len(molecules), len(scaffolds))

    prepared_dir = out_dir / "prepared"
    prepared = {t.key: prepare_crop(t, cfg.preprocess, prepared_dir) for t in tasks}

    summary: Dict = {"manifest": str(manifest),
                     "crops": {LABEL_MOLECULE: len(molecules), LABEL_MARKUSH: len(scaffolds)},
                     "missing_crops": missing,
                     "tools": {}, "markush": None}
    for tool in tools:
        recognizer = recognizer_factory(tool, cfg)
        readings = {t.key: recognizer.predict(prepared[t.key]) for t in molecules}
        rows = [prediction_row(t, recognizer.name, readings[t.key]) for t in molecules]
        write_rows(out_dir / f"predictions_{tool}.csv", PREDICTION_COLUMNS, rows)
        summary["tools"][tool] = _tool_summary(rows)
        LOGGER.info("%s: %d predictions written", recognizer.name, len(rows))

        if cfg.markush.enabled and tool == cfg.markush.scaffold_tool:
            readings.update({t.key: recognizer.predict(prepared[t.key]) for t in scaffolds})
            # The gate cannot see R-groups that only appear in a table, so a "molecule"
            # crop whose reading has attachment points is treated as a scaffold too.
            extra = [t for t in molecules
                     if cfg.markush.from_molecule_crops and has_attachment_points(readings[t.key])]
            markush_rows, audits = run_markush_branch(scaffolds + extra, readings, cfg, rgroup_csv)
            write_rows(out_dir / "markush_compounds.csv", MARKUSH_COLUMNS, markush_rows)
            write_rows(out_dir / "markush_scaffolds.csv", SCAFFOLD_AUDIT_COLUMNS, audits)
            summary["markush"] = {
                "scaffolds": len(audits), "from_markush_crops": len(scaffolds), "from_molecule_crops": len(extra),
                "compounds_built": sum(1 for r in markush_rows if r["method"].startswith("enumerated")),
                "outcomes": dict(Counter(a["outcome"].split("(")[0] for a in audits)),
            }

    if not cfg.preprocess.save_prepared:
        shutil.rmtree(prepared_dir, ignore_errors=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
