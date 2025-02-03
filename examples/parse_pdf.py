"""1. Script to parse a directory of PDF files."""

from __future__ import annotations

import argparse

from predict_llm.data import pdf_parser

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--pdf_dir',
    help='Directory containing PDF files to parse.',
)
argparser.add_argument(
    '--output_dir',
    help='Directory to save parsed PDFs to.',
)

args = argparser.parse_args()

pdf_parser()
