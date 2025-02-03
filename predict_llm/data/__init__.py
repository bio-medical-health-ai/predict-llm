"""Data module for predict_llm package."""

from __future__ import annotations

from .utils import pdf_parser
from .utils import synthetic_data_generator

__all__ = [
    'pdf_parser',
    'synthetic_data_generator',
]
