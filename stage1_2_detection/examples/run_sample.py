"""Reproduce the sample run: build a synthetic paper, process it, score the result.

Usage (from the repository root):
    python -m stage1_2_detection.examples.run_sample

Writes ``examples/sample_input/sample_paper.pdf`` and ``examples/sample_output/``
(manifest.csv, summary.json, metrics.json, crops/, rejected/, overlays/).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from stage1_2_detection import load_config, run_many
from stage1_2_detection.pipeline import write_summary
from stage1_2_detection.tests.synthetic import build_sample_pdf, evaluate, truth_to_pixels

HERE = Path(__file__).resolve().parent


def main() -> None:
    input_dir, output_dir = HERE / "sample_input", HERE / "sample_output"
    input_dir.mkdir(exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    pdf = input_dir / "sample_paper.pdf"
    truth = build_sample_pdf(pdf)

    cfg = load_config()
    rows = run_many([pdf], output_dir, cfg, save_rejected=True, overlay=True)
    metrics = evaluate(rows, truth_to_pixels(truth, cfg.render.dpi))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary(rows, output_dir / "summary.json")

    for row in rows:
        print(f"{row.crop_id}  {row.label:12s} score={row.gate_score:.2f}  {row.reason}"
              f"  {row.markush_evidence}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
