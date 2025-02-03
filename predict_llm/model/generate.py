"""Models and generation functions."""

from __future__ import annotations


class BaseHFModel:
    """Base Hugging Face model."""

    def __init__(self) -> None:
        """Load model."""
        print('Loading model...')


class FineTunedHFModel:
    """Fine-tuned Hugging Face model."""

    def __init__(self) -> None:
        """Load fine-tuned model."""
        print('Loading fine-tuned model...')


class RAGModel:
    """Retrieval-Augmented Generation model."""

    def __init__(self) -> None:
        """Load RAG model."""
        print('Loading RAG model...')


def generate() -> None:
    """Generate model output."""
    print('Generating model output...')
