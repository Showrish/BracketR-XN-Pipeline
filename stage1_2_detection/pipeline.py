"""Stage 1-2 driver: PDF -> candidate regions -> gate -> crops + manifest."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import Config, load_config
from .detect import Box, detect_classical, detect_decimer
from .gate import GateResult, classify_crop
from .manifest import (ACCEPTED_LABELS, LABEL_MARKUSH, LABEL_MOLECULE, Candidate,
                       write_manifest)
from .render import render_pages
from .split import split_region
from .textlayer import Word, attach_label_words, load_text_layer, words_in_box

log = logging.getLogger(__name__)

_COLORS = {LABEL_MOLECULE: (0, 160, 0), LABEL_MARKUSH: (0, 90, 220),
           "not_molecule": (220, 0, 0)}


def _reading_order(items: List[Tuple[Optional[Box], np.ndarray]]):
    """Sort regions top-to-bottom, then left-to-right (unlocated ones last)."""
    def key(item):
        box = item[0]
        return (box[1], box[0]) if box else (10 ** 9, 0)
    return sorted(items, key=key)


def find_regions(page_rgb: np.ndarray, cfg: Config) -> List[Tuple[Optional[Box], np.ndarray]]:
    """Run the configured detector; return ``[(box, crop_rgb), ...]``."""
    if cfg.detect.backend == "decimer":
        return _reading_order(detect_decimer(page_rgb, cfg.detect))
    if cfg.detect.backend != "classical":
        raise ValueError(f"Unknown detector backend: {cfg.detect.backend}")
    out = []
    for box in detect_classical(page_rgb, cfg.render.dpi, cfg.detect):
        x, y, w, h = box
        out.append((box, page_rgb[y:y + h, x:x + w]))
    return _reading_order(out)


def _clip(box: Box, page_rgb: np.ndarray) -> Box:
    """Keep a box inside the page."""
    height, width = page_rgb.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    return (x0, y0, x1 - x0, y1 - y0)


def _save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _draw_overlay(page_rgb: np.ndarray, rows: Sequence[Candidate], path: Path) -> None:
    canvas = page_rgb.copy()
    thickness = max(2, page_rgb.shape[1] // 400)
    for row in rows:
        if row.x is None:
            continue
        color = _COLORS.get(row.label, (128, 128, 128))
        cv2.rectangle(canvas, (row.x, row.y), (row.x + row.width, row.y + row.height),
                      color, thickness)
        cv2.putText(canvas, row.crop_id.split("_")[1], (row.x + 4, row.y + 4 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    scale = min(1.0, 1400.0 / canvas.shape[1])
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _save_png(canvas, path)


FIGURE_REASONS = ("colored_figure", "label_heavy_figure", "chart_or_table_axes", "curved_lines")
def _is_mixed(result: GateResult, box: Box, dpi: int, cfg: Config) -> bool:
    """Could this crop be several panels glued together (e.g. molecules + a chart)?

    * accepted crops that are very large or contain many letters (panels glued together),
    * crops rejected as a chart/diagram that are not tiny
    are re-examined piece by piece (see ``split.py``). Text blocks and photos are
    never split.
    """
    if not cfg.detect.split_enabled:
        return False
    area_in2 = (box[2] / dpi) * (box[3] / dpi)
    if result.label in ACCEPTED_LABELS:
        return (result.features.get("text_cc_count", 0) >= cfg.detect.split_min_letters
                or area_in2 >= cfg.detect.split_min_area_in2)
    return (result.reason.startswith(FIGURE_REASONS)
            and area_in2 >= cfg.detect.split_min_area_rejected_in2)


def _split_parts(page_rgb: np.ndarray, box: Box, result: GateResult, cfg: Config,
                 classify) -> List[Tuple[Box, GateResult, bool]]:
    """Split a suspicious region; keep the original if no molecule is found inside.

    Tiny rejected pieces (single words, specks) are not reported separately.
    """
    dpi = cfg.render.dpi
    parts = split_region(page_rgb, box, dpi, cfg.detect, classify)
    if not any(r.label in ACCEPTED_LABELS for _, r in parts):
        return [(box, result, False)]
    min_side = int(round(cfg.detect.min_side_in * dpi))
    return [(b, r, True) for b, r in parts
            if r.label in ACCEPTED_LABELS or min(b[2], b[3]) >= min_side]


def _grow_with_labels(box: Box, crop: np.ndarray, page_rgb: np.ndarray,
                      words: List[Word], cfg: Config):
    """Pull in detached atom labels from the text layer; returns (box, crop)."""
    grown = attach_label_words(box, words, int(cfg.detect.label_attach_in * cfg.render.dpi))
    if grown == box:
        return box, crop
    x, y, w, h = _clip(grown, page_rgb)
    return (x, y, w, h), page_rgb[y:y + h, x:x + w]


def process_page(page_rgb: np.ndarray, page: int, words: List[Word], cfg: Config,
                 paper_id: str, source_pdf: str, out_dir: Path,
                 save_rejected: bool, overlay: bool) -> List[Candidate]:
    dpi = cfg.render.dpi
    near_pad = int(cfg.gate.markush_pad_in * dpi)

    def classify(b: Box) -> GateResult:
        x, y, w, h = b
        return classify_crop(page_rgb[y:y + h, x:x + w], dpi, cfg.gate,
                             words_in_box(words, x, y, w, h),
                             words_in_box(words, x, y, w, h, pad=near_pad))

    # 1. detect regions and gate them; split suspicious accepted ones
    regions = []                      # (box or None, crop, GateResult, split_flag)
    for box, crop in find_regions(page_rgb, cfg):
        if box is None:               # backend could not locate the crop
            regions.append((None, crop, classify_crop(crop, dpi, cfg.gate), False))
            continue
        result = classify(box)
        if _is_mixed(result, box, dpi, cfg):
            for part_box, part_result, was_split in _split_parts(page_rgb, box, result, cfg,
                                                                  classify):
                x, y, w, h = part_box
                regions.append((part_box, page_rgb[y:y + h, x:x + w], part_result, was_split))
        else:
            regions.append((box, crop, result, False))
    regions.sort(key=lambda r: (r[0][1], r[0][0]) if r[0] else (10 ** 9, 0))

    # 2. save crops and build manifest rows
    rows: List[Candidate] = []
    for index, (box, crop, result, was_split) in enumerate(regions, start=1):
        if result.label in ACCEPTED_LABELS and box is not None and words:
            box, crop = _grow_with_labels(box, crop, page_rgb, words, cfg)
        x, y, w, h = box if box else (None, None, None, None)
        crop_id = f"p{page:03d}_c{index:02d}"
        accepted = result.label in ACCEPTED_LABELS
        rel_path = Path("crops" if accepted else "rejected") / f"paper_{paper_id}" / f"{crop_id}.png"
        if accepted or save_rejected:
            _save_png(crop, out_dir / rel_path)
        reason = result.reason + (" [split]" if was_split and accepted else "")
        rows.append(Candidate(
            paper_id=paper_id, crop_id=crop_id, crop_path=rel_path.as_posix(),
            source_pdf=source_pdf, page=page, x=x, y=y, width=w, height=h, dpi=dpi,
            label=result.label, gate_score=round(result.score, 3), reason=reason,
            markush_evidence=result.markush_evidence, detector=cfg.detect.backend))
    if overlay:
        _draw_overlay(page_rgb, rows,
                      out_dir / "overlays" / f"paper_{paper_id}" / f"p{page:03d}.png")
    return rows


def run_stage1_2(pdf_path: str | Path, out_dir: str | Path,
                 config: Optional[Config] = None, paper_id: Optional[str] = None,
                 save_rejected: bool = True, overlay: bool = False,
                 write_files: bool = True) -> List[Candidate]:
    """Process one PDF (or image). Returns the manifest rows.

    Writes ``crops/``, ``rejected/``, optional ``overlays/`` and
    ``manifest.csv`` under ``out_dir`` unless ``write_files`` is False.
    """
    cfg = config or load_config()
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    paper_id = paper_id or pdf_path.stem
    words = load_text_layer(pdf_path, cfg.render.dpi)
    rows: List[Candidate] = []
    for page, page_rgb in render_pages(pdf_path, cfg.render.dpi,
                                       cfg.render.first_page, cfg.render.last_page):
        page_rows = process_page(page_rgb, page, words.get(page, []), cfg, paper_id,
                                 pdf_path.name, out_dir, save_rejected, overlay)
        log.info("%s page %d: %d regions, %d kept", pdf_path.name, page, len(page_rows),
                 sum(r.label in ACCEPTED_LABELS for r in page_rows))
        rows.extend(page_rows)
    if write_files:
        write_manifest(rows, out_dir / "manifest.csv")
    return rows


def run_many(pdf_paths: Iterable[str | Path], out_dir: str | Path,
             config: Optional[Config] = None, save_rejected: bool = True,
             overlay: bool = False) -> List[Candidate]:
    """Process several PDFs into one combined manifest and summary."""
    out_dir = Path(out_dir)
    rows: List[Candidate] = []
    for pdf in pdf_paths:
        rows.extend(run_stage1_2(pdf, out_dir, config, None, save_rejected, overlay,
                                 write_files=False))
    write_manifest(rows, out_dir / "manifest.csv")
    write_summary(rows, out_dir / "summary.json")
    return rows


def write_summary(rows: Sequence[Candidate], path: str | Path) -> Dict:
    """Counts per paper and label, plus rejection reasons (for the README/report)."""
    summary: Dict = {"total_regions": len(rows), "by_label": dict(Counter(r.label for r in rows)),
                     "reject_reasons": dict(Counter(r.reason.split("(")[0] for r in rows
                                                    if r.label not in ACCEPTED_LABELS)),
                     "by_paper": {}}
    for paper in sorted({r.paper_id for r in rows}):
        subset = [r for r in rows if r.paper_id == paper]
        summary["by_paper"][paper] = dict(Counter(r.label for r in subset))
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
