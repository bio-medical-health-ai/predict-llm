"""Retrieval Augmented Generation (RAG) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tiktoken
from chonkie import WordChunker
from template import general_cot
from template import general_cot_system
from template import general_medrag
from template import general_medrag_system
from transformers import AutoTokenizer
from utils import RetrievalSystem
from utils import RetrievalSystemConfig


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    llm_name: str = 'meta-llama/Llama-3.1-8B-instruct'
    retriever_name: str = 'MedCPT'
    corpus_name: str = 'Marfan'
    db_dir: str = './corpus'
    cache_dir: str | None = None
    max_length: int = 2048
    context_length: int = 1024


@dataclass
class ContextGenerationConfig:
    """Configuration for context generation."""

    k: int = 32
    sub_k: int | None = None
    total_k: int | None = None
    rrf_k: int = 100
    split: bool = False


class RAGSystem:
    """RAG system implementation."""

    def __init__(self, config: RAGConfig):
        """Initialize the RAG system with given configuration."""
        self.config = config
        self.retrieval_system = None
        self.tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self) -> None:
        """Initialize the appropriate tokenizer based on model name."""
        if 'openai' in self.config.llm_name.lower():
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
            if (
                'gpt-3.5' in self.config.llm_name
                or 'gpt-35' in self.config.llm_name
            ):
                self.config.max_length = 16384
                self.config.context_length = 14500
            elif 'gpt-4' in self.config.llm_name:
                self.config.max_length = 32768
                self.config.context_length = 29500
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_name,
                cache_dir=self.config.cache_dir,
            )
            self._set_model_specific_lengths()

    def _set_model_specific_lengths(self) -> None:
        """Set model-specific length constraints."""
        if 'mistral' in self.config.llm_name.lower():
            self.config.max_length = 8192
            self.config.context_length = 6656
        elif 'llama-2' in self.config.llm_name.lower():
            self.config.max_length = 4096
            self.config.context_length = 2560
        elif 'llama-3.' in self.config.llm_name.lower():
            self.config.max_length = 128000
            self.config.context_length = 125000
        elif 'llama-3' in self.config.llm_name.lower():
            self.config.max_length = 8192
            self.config.context_length = 6656


def generate_index(config: RAGConfig) -> RetrievalSystem:
    """Generate vector database for document indexing.

    Args:
        config: RAG configuration object

    Returns:
        RetrievalSystem: Initialized retrieval system
    """
    retrieval_config = RetrievalSystemConfig()
    retrieval_system = RetrievalSystem(
        retriever_name=config.retriever_name,
        corpus_name=config.corpus_name,
        db_dir=config.db_dir,
        config=retrieval_config,
    )
    msg = (
        f'Generated index for {config.corpus_name} '
        f'using {config.retriever_name}'
    )
    print(msg)
    return retrieval_system


def process_retrieved_snippets(
    retrieved_snippets: list[dict],
    scores: list[float],
    seen_documents: dict,
) -> tuple[list[dict], list[float], list[str]]:
    """Process retrieved snippets and handle duplicates.

    Args:
        retrieved_snippets: List of retrieved document snippets
        scores: List of relevance scores
        seen_documents: Dictionary tracking seen documents

    Returns:
        Tuple of processed snippets, scores, and context strings
    """
    current_snippets = []
    current_scores = []
    current_contexts = []

    for snippet, score in zip(retrieved_snippets, scores):
        doc_content = snippet['contents']

        if doc_content in seen_documents:
            prev_idx, prev_score = seen_documents[doc_content]
            if score > prev_score:
                seen_documents[doc_content] = (len(seen_documents), score)
        else:
            seen_documents[doc_content] = (len(seen_documents), score)
            current_snippets.append(snippet)
            current_scores.append(score)
            current_contexts.append(
                f'Document [{len(seen_documents) - 1}]: {doc_content}',
            )

    if not current_contexts:
        current_contexts = ['']

    return current_snippets, current_scores, current_contexts


def adjust_contexts_for_length(
    sub_question_results: list[dict],
    excess_tokens: int,
) -> None:
    """Adjust context lengths to fit within token limit.

    Args:
        sub_question_results: List of results for each sub-question
        excess_tokens: Number of tokens to reduce
    """
    tokens_to_reduce = excess_tokens // len(sub_question_results)

    for result in sub_question_results:
        if result['tokens'] > 0:
            keep_ratio = max(
                0,
                (result['tokens'] - tokens_to_reduce) / result['tokens'],
            )
            keep_contexts = max(
                1,
                int(len(result['contexts']) * keep_ratio),
            )
            result['contexts'] = result['contexts'][:keep_contexts]
            result['retrieved_snippets'] = result['retrieved_snippets'][
                :keep_contexts
            ]
            result['scores'] = result['scores'][:keep_contexts]


def generate_context(
    retrieval_system: RetrievalSystem,
    question: str,
    tokenizer: Any,
    context_length: int,
    config: ContextGenerationConfig | None = None,
) -> tuple[str, list[dict], list[float]]:
    """Generate context from retrieved documents using two-pass approach.

    Args:
        retrieval_system: Initialized retrieval system
        question: Input question
        tokenizer: Tokenizer for length calculation
        context_length: Maximum context length
        config: Configuration for context generation

    Returns:
        Tuple containing combined context, retrieved snippets, and scores
    """
    if config is None:
        config = ContextGenerationConfig()

    sub_questions = (
        [
            chunk.text
            for chunk in WordChunker(
                tokenizer='gpt2',
                chunk_size=512,
                chunk_overlap=25,
            ).chunk(question)
        ]
        if config.split
        else [question]
    )

    if config.sub_k is None:
        config.sub_k = config.k

    adjusted_sub_k = (
        min(config.sub_k, (config.total_k * 2) // len(sub_questions))
        if config.total_k and len(sub_questions) > 0
        else config.sub_k
    )

    sub_question_results = []
    total_tokens = 0
    seen_documents = {}

    for sub_question in sub_questions:
        retrieved_snippets, scores = retrieval_system.retrieve(
            sub_question,
            k=adjusted_sub_k,
            rrf_k=config.rrf_k,
        )

        snippets, scores, contexts = process_retrieved_snippets(
            retrieved_snippets,
            scores,
            seen_documents,
        )

        if config.total_k and len(seen_documents) >= config.total_k:
            break

        context_text = '\n'.join(contexts)
        tokens = (
            len(tokenizer.encode(context_text))
            if isinstance(tokenizer, tiktoken.Encoding)
            else len(tokenizer.encode(context_text, add_special_tokens=False))
        )

        sub_question_results.append(
            {
                'contexts': contexts,
                'retrieved_snippets': snippets,
                'scores': scores,
                'tokens': tokens,
            },
        )
        total_tokens += tokens

    if total_tokens > context_length and sub_question_results:
        adjust_contexts_for_length(
            sub_question_results,
            total_tokens - context_length,
        )

    all_contexts = []
    all_retrieved_snippets = []
    all_scores = []

    for result in sub_question_results:
        all_contexts.extend(result['contexts'])
        all_retrieved_snippets.extend(result['retrieved_snippets'])
        all_scores.extend(result['scores'])

    combined_context = '\n'.join(all_contexts)
    if isinstance(tokenizer, tiktoken.Encoding):
        combined_context = tokenizer.decode(
            tokenizer.encode(combined_context)[:context_length],
        )
    else:
        combined_context = tokenizer.decode(
            tokenizer.encode(
                combined_context,
                add_special_tokens=False,
            )[:context_length],
        )

    return combined_context, all_retrieved_snippets, all_scores


def generate_prompt(
    question: str,
    contexts: list[str],
    rag: bool = True,
    templates: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate prompt for the language model.

    Args:
        question: Input question
        contexts: Retrieved contexts
        rag: Whether to use RAG
        templates: Custom templates for prompt generation

    Returns:
        List of message dictionaries for the language model
    """
    if templates is None:
        templates = {
            'cot_system': general_cot_system,
            'cot_prompt': general_cot,
            'medrag_system': general_medrag_system,
            'medrag_prompt': general_medrag,
        }

    if not rag:
        prompt = templates['cot_prompt'].render(question=question)
        messages = [
            {'role': 'system', 'content': templates['cot_system']},
            {'role': 'user', 'content': prompt},
        ]
    else:
        combined_context = '\n'.join(contexts)
        prompt = templates['medrag_prompt'].render(
            context=combined_context,
            question=question,
        )
        messages = [
            {'role': 'system', 'content': templates['medrag_system']},
            {'role': 'user', 'content': prompt},
        ]

    return messages
