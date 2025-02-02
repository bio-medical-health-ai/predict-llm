from predict_llm.model import generate
from predict_llm.finetune import finetune
from predict_llm.rag import generate_context, generate_vector_db
from predict_llm.data import synthetic_data_generator, pdf_parser

generate()
finetune()
generate_context()
generate_vector_db()
synthetic_data_generator()
pdf_parser()