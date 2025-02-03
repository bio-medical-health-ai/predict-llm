"""Shows how to use different components of the `predict_llm` package."""

from __future__ import annotations

from predict_llm.data import pdf_parser
from predict_llm.data import synthetic_data_generator
from predict_llm.finetune import finetune
from predict_llm.model import generate
from predict_llm.retrieval import generate_context
from predict_llm.retrieval import generate_vector_db

generate()
finetune()
generate_context()
generate_vector_db()
synthetic_data_generator()
pdf_parser()
