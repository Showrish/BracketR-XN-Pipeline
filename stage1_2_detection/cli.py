"""Command line entry point: ``python -m stage1_2_detection.cli --pdf-dir papers --out out``."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .pipeline import run_many


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1-2: find molecule drawings in PDFs.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", nargs="+", help="one or more PDF/image files")
    src.add_argument("--pdf-dir", help="folder containing PDFs")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--config", help="YAML file overriding the defaults")
    p.add_argument("--dpi", type=int, help="render resolution (default 300)")
    p.add_argument("--backend", choices=["classical", "decimer"])
    p.add_argument("--first-page", type=int)
    p.add_argument("--last-page", type=int)
    p.add_argument("--no-rejected", action="store_true", help="do not save rejected crops")
    p.add_argument("--overlay", action="store_true", help="save page overlays with boxes")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.dpi:
        cfg.render.dpi = args.dpi
    if args.backend:
        cfg.detect.backend = args.backend
    cfg.render.first_page = args.first_page or cfg.render.first_page
    cfg.render.last_page = args.last_page or cfg.render.last_page

    pdfs = args.pdf or sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))
    rows = run_many(pdfs, args.out, cfg, save_rejected=not args.no_rejected,
                    overlay=args.overlay)
    kept = [r for r in rows if r.label != "not_molecule"]
    print(f"{len(rows)} regions found, {len(kept)} passed the gate. "
          f"Manifest: {Path(args.out) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
