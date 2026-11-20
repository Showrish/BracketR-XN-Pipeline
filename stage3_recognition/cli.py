"""Command line entry point.

    python -m stage3_recognition.cli --manifest out/stage12/manifest.csv --out out/stage3

Run once per environment, e.g. ``--tools molscribe`` in the MolScribe environment and
``--tools decimer`` in the DECIMER environment, with the same ``--out`` folder.
"""
from __future__ import annotations

import argparse
import json
import logging

from .config import load_config
from .pipeline import run_stage3
from .recognizers import AVAILABLE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 3: crops from stage 1-2 -> SMILES.")
    p.add_argument("--manifest", required=True, help="manifest.csv written by stage 1-2")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--tools", nargs="+", choices=AVAILABLE, help="recognizers to run (default: config)")
    p.add_argument("--pdf-dir", help="folder with the source PDFs, used to read R-group tables")
    p.add_argument("--rgroup-table", help="CSV with compound_label, R1, R2, ... (overrides the text layer)")
    p.add_argument("--config", help="YAML file overriding the defaults")
    p.add_argument("--no-markush", action="store_true", help="skip the Markush branch (3b)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.no_markush:
        cfg.markush.enabled = False
    summary = run_stage3(args.manifest, args.out, cfg, tools=args.tools, pdf_dir=args.pdf_dir,
                         rgroup_csv=args.rgroup_table)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
