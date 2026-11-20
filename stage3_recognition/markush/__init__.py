"""Stage 3b: the Markush branch (scaffold + R-group table -> full compounds)."""
from .attachments import attachment_labels, normalize_attachment_points
from .enumerate import Compound, enumerate_compounds, match_columns
from .rgroup_table import RGroupTable, parse_table_lines, table_from_csv, table_from_pdf_text
from .substituents import SUBSTITUENTS, to_fragment

__all__ = ["attachment_labels", "normalize_attachment_points", "Compound", "enumerate_compounds",
           "match_columns", "RGroupTable", "parse_table_lines", "table_from_csv",
           "table_from_pdf_text", "SUBSTITUENTS", "to_fragment"]
