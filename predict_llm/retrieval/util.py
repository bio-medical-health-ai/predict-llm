"""Utility functions and classes for the retrieval system.

This module provides classes and functions

for document retrieval and embedding.
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from sentence_transformers.models import Transformer

# Default configurations
DEFAULT_CORPUS_NAMES = {
    'Marfan': ['marfan'],
}

DEFAULT_RETRIEVER_NAMES = {
    'BM25': ['bm25'],
    'Contriever': ['facebook/contriever'],
    'SPECTER': ['allenai/specter'],
    'MedCPT': ['ncbi/MedCPT-Query-Encoder'],
    'RRF-2': ['bm25', 'ncbi/MedCPT-Query-Encoder'],
    'RRF-4': [
        'bm25',
        'facebook/contriever',
        'allenai/specter',
        'ncbi/MedCPT-Query-Encoder',
    ],
}


class RetrievalSystemConfig:
    """Configuration class for storing and validating retrieval settings."""

    def __init__(
        self,
        corpus_names: dict[str, list[str]] | None = None,
        retriever_names: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the retrieval configuration.

        Args:
            corpus_names: Dictionary mapping corpus names to lists of
                corpus identifiers
            retriever_names: Dictionary mapping retriever names to lists
                of model identifiers
        """
        self.corpus_names = corpus_names or DEFAULT_CORPUS_NAMES
        self.retriever_names = retriever_names or DEFAULT_RETRIEVER_NAMES
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the configuration settings."""
        if not isinstance(self.corpus_names, dict) or not isinstance(
            self.retriever_names,
            dict,
        ):
            raise ValueError(
                'corpus_names and retriever_names must be dictionaries',
            )

        for name, corpora in self.corpus_names.items():
            if not isinstance(corpora, list):
                raise ValueError(
                    f'Value for {name} in corpus_names must be a list',
                )

        for name, retrievers in self.retriever_names.items():
            if not isinstance(retrievers, list):
                raise ValueError(
                    f'Value for {name} in retriever_names must be a list',
                )


class CustomizeSentenceTransformer(SentenceTransformer):
    """Custom SentenceTransformer using CLS pooling instead of MEAN pooling."""

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        """Create a Transformer + CLS Pooling model and return the modules.

        Args:
            model_name_or_path: Path to the transformer model
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List containing transformer and pooling models
        """
        print(
            'No sentence-transformers model found with name '
            f'{model_name_or_path}. Creating a new one with CLS pooling.',
        )

        token = kwargs.get('token')
        cache_folder = kwargs.get('cache_folder')
        revision = kwargs.get('revision')
        trust_remote_code = kwargs.get('trust_remote_code', False)

        if any(
            key in kwargs
            for key in (
                'token',
                'cache_folder',
                'revision',
                'trust_remote_code',
            )
        ):
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={
                    'token': token,
                    'trust_remote_code': trust_remote_code,
                    'revision': revision,
                },
                tokenizer_args={
                    'token': token,
                    'trust_remote_code': trust_remote_code,
                    'revision': revision,
                },
            )
        else:
            transformer_model = Transformer(model_name_or_path)

        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(),
            'cls',
        )
        return [transformer_model, pooling_model]


def embed(chunk_dir: str, index_dir: str, model_name: str, **kwargs) -> int:
    """Embed text chunks using the specified model.

    Args:
        chunk_dir: Directory containing text chunks
        index_dir: Directory to save embeddings
        model_name: Name of the embedding model
        **kwargs: Additional keyword arguments

    Returns:
        Dimension of the embeddings
    """
    save_dir = os.path.join(index_dir, 'embedding')

    if 'contriever' in model_name:
        model = SentenceTransformer(
            model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    else:
        model = CustomizeSentenceTransformer(
            model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    model.eval()

    fnames = sorted(
        [fname for fname in os.listdir(chunk_dir) if fname.endswith('.jsonl')],
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace('.jsonl', '.npy'))
            if os.path.exists(save_path):
                continue

            with open(fpath) as f:
                if f.read().strip() == '':
                    continue

            with open(fpath) as f:
                texts = [
                    json.loads(item) for item in f.read().strip().split('\n')
                ]
            texts = [item['contents'] for item in texts]
            embed_chunks = model.encode(texts, **kwargs)
            np.save(save_path, embed_chunks)
        embed_chunks = model.encode([''], **kwargs)
    return embed_chunks.shape[-1]


def construct_index(
    index_dir: str,
    model_name: str,
    h_dim: int = 768,
) -> faiss.Index:
    """Construct a FAISS index from embeddings.

    Args:
        index_dir: Directory containing embeddings
        model_name: Name of the model used for embeddings
        h_dim: Dimension of embeddings

    Returns:
        FAISS index
    """
    with open(os.path.join(index_dir, 'metadatas.jsonl'), 'w') as f:
        f.write('')

    if 'specter' in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    for fname in tqdm.tqdm(
        sorted(os.listdir(os.path.join(index_dir, 'embedding'))),
    ):
        curr_embed = np.load(os.path.join(index_dir, 'embedding', fname))
        index.add(curr_embed)
        with open(os.path.join(index_dir, 'metadatas.jsonl'), 'a+') as f:
            f.write(
                '\n'.join(
                    [
                        json.dumps(
                            {'index': i, 'source': fname.replace('.npy', '')},
                        )
                        for i in range(len(curr_embed))
                    ],
                )
                + '\n',
            )

    faiss.write_index(index, os.path.join(index_dir, 'faiss.index'))
    return index


class Retriever:
    """Retriever class for getting relevant documents from an index."""

    def __init__(
        self,
        retriever_name: str = 'ncbi/MedCPT-Query-Encoder',
        corpus_name: str = 'Wikipedia',
        db_dir: str = './corpus',
        **kwargs,
    ) -> None:
        """Initialize the retriever.

        Args:
            retriever_name: Name of the retriever model
            corpus_name: Name of the corpus
            db_dir: Base directory for data
            **kwargs: Additional keyword arguments
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, 'chunk')
        if not os.path.exists(self.chunk_dir):
            raise FileNotFoundError(
                f'Chunk directory not found: {self.chunk_dir}',
            )

        self.index_dir = os.path.join(
            self.db_dir,
            self.corpus_name,
            'index',
            self.retriever_name.replace('Query-Encoder', 'Article-Encoder'),
        )

        self._initialize_index(**kwargs)

    def _initialize_index(self, **kwargs) -> None:
        """Initialize the index and embedding function."""
        if 'bm25' in self.retriever_name.lower():
            self._initialize_bm25_index()
        else:
            self._initialize_dense_index(**kwargs)

    def _initialize_bm25_index(self) -> None:
        """Initialize BM25 index."""
        from pyserini.search.lucene import LuceneSearcher

        self.metadatas = None
        self.embedding_function = None

        if os.path.exists(self.index_dir):
            self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            os.system(
                'python -m pyserini.index.lucene '
                '--collection JsonCollection '
                f'--input {self.chunk_dir} --index {self.index_dir} '
                '--generator DefaultLuceneDocumentGenerator --threads 16',
            )
            self.index = LuceneSearcher(os.path.join(self.index_dir))

    def _initialize_dense_index(self, **kwargs) -> None:
        """Initialize dense retrieval index."""
        if os.path.exists(os.path.join(self.index_dir, 'faiss.index')):
            self._load_existing_index()
        else:
            self._create_new_index(**kwargs)

    def _load_existing_index(self) -> None:
        """Load existing FAISS index and metadata."""
        self.index = faiss.read_index(
            os.path.join(self.index_dir, 'faiss.index'),
        )
        with open(os.path.join(self.index_dir, 'metadatas.jsonl')) as f:
            self.metadatas = [
                json.loads(line) for line in f.read().strip().split('\n')
            ]

    def _create_new_index(self, **kwargs) -> None:
        """Create new index by embedding documents."""
        print(
            '[In progress] Embedding the '
            f'{self.corpus_name} corpus with the '
            f'{
                self.retriever_name.replace("Query-Encoder", "Article-Encoder")
            } '
            'retriever...',
        )

        h_dim = embed(
            chunk_dir=self.chunk_dir,
            index_dir=self.index_dir,
            model_name=self.retriever_name.replace(
                'Query-Encoder',
                'Article-Encoder',
            ),
            **kwargs,
        )

        print(
            '[In progress] Embedding finished! '
            f'The dimension of the embeddings is {h_dim}.',
        )

        self.index = construct_index(
            index_dir=self.index_dir,
            model_name=self.retriever_name.replace(
                'Query-Encoder',
                'Article-Encoder',
            ),
            h_dim=h_dim,
        )

        print('[Finished] Corpus indexing finished!')

        with open(os.path.join(self.index_dir, 'metadatas.jsonl')) as f:
            self.metadatas = [
                json.loads(line) for line in f.read().strip().split('\n')
            ]

        if 'contriever' in self.retriever_name.lower():
            self.embedding_function = SentenceTransformer(
                self.retriever_name,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        else:
            self.embedding_function = CustomizeSentenceTransformer(
                self.retriever_name,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
        self.embedding_function.eval()

    def get_relevant_documents(
        self,
        question: str,
        k: int = 32,
        id_only: bool = False,
        **kwargs,
    ) -> tuple[list, list]:
        """Get relevant documents for a question.

        Args:
            question: Input question
            k: Number of documents to retrieve
            id_only: Whether to return only document IDs
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (documents, scores)
        """
        if not isinstance(question, str):
            raise TypeError('question must be a string')

        question = [question]

        if 'bm25' in self.retriever_name.lower():
            documents, scores = self._get_bm25_documents(question, k)
        else:
            documents, scores = self._get_dense_documents(
                question,
                k,
                **kwargs,
            )

        if id_only:
            return [{'id': doc_id} for doc_id in documents], scores
        else:
            return self.idx2txt(documents), scores

    def idx2txt(self, indices: list[dict]) -> list[dict]:
        """Convert document indices to their text content.

        Args:
            indices: List of document index dictionaries

        Returns:
            List of document content dictionaries
        """

        def remove_extension(filename: str) -> str:
            if filename.endswith('tei'):
                return filename
            return os.path.splitext(filename)[0]

        documents = []
        for idx in indices:
            fpath = os.path.join(
                self.chunk_dir,
                f'{remove_extension(idx["source"])}.jsonl',
            )
            with open(fpath) as f:
                document = json.loads(
                    f.read().strip().split('\n')[idx['index']],
                )
                documents.append(document)

        return documents


class RetrievalSystem:
    """Enhanced retrieval system with configurable corpus and models."""

    def __init__(
        self,
        retriever_name: str = 'MedCPT',
        corpus_name: str = 'Wikipedia',
        db_dir: str = './corpus',
        config: RetrievalSystemConfig | None = None,
    ) -> None:
        """Initialize the retrieval system.

        Args:
            retriever_name: Name of the retriever to use
            corpus_name: Name of the corpus to search
            db_dir: Base directory for data
            config: Optional configuration object
        """
        self.config = config or RetrievalSystemConfig()
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        # Validate input parameters
        if self.corpus_name not in self.config.corpus_names:
            available = list(self.config.corpus_names.keys())
            raise ValueError(
                f'Unsupported corpus_name: {self.corpus_name}. '
                f'Available options: {available}',
            )
        if self.retriever_name not in self.config.retriever_names:
            available = list(self.config.retriever_names.keys())
            raise ValueError(
                f'Unsupported retriever_name: {self.retriever_name}. '
                f'Available options: {available}',
            )

        # Initialize retrievers
        self.retrievers = []
        for retriever in self.config.retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in self.config.corpus_names[self.corpus_name]:
                self.retrievers[-1].append(
                    Retriever(retriever, corpus, db_dir),
                )

    def retrieve(
        self,
        question: str,
        k: int = 32,
        rrf_k: int = 100,
    ) -> tuple[list, list]:
        """Retrieve relevant documents based on the input question.

        Args:
            question: Input question string
            k: Number of documents to retrieve
            rrf_k: Parameter for reciprocal rank fusion

        Returns:
            Tuple of (texts, scores) where texts are the retrieved documents
            and scores are their relevance scores
        """
        if not isinstance(question, str):
            raise TypeError('question must be a string')

        texts = []
        scores = []

        k_ = max(k * 2, 100) if 'RRF' in self.retriever_name else k

        for i in range(len(self.config.retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(self.config.corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(
                    question,
                    k=k_,
                )
                texts[-1].append(t)
                scores[-1].append(s)

        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        return texts, scores

    def merge(
        self,
        texts: list,
        scores: list,
        k: int = 32,
        rrf_k: int = 100,
    ) -> tuple[list, list]:
        """Merge retriever results using reciprocal rank fusion.

        Args:
            texts: List of retrieved texts from different retrievers
            scores: List of corresponding scores
            k: Number of documents to return
            rrf_k: Parameter for reciprocal rank fusion

        Returns:
            Tuple of (merged_texts, merged_scores)
        """
        rrf_dict = {}
        for i in range(len(self.config.retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            for j in range(len(self.config.corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]

            if (
                'specter'
                in self.config.retriever_names[self.retriever_name][i].lower()
            ):
                sorted_index = np.array(scores_all).argsort()
            else:
                sorted_index = np.array(scores_all).argsort()[::-1]

            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]

            for j, item in enumerate(texts[i]):
                if item['id'] in rrf_dict:
                    rrf_dict[item['id']]['score'] += 1 / (rrf_k + j + 1)
                    rrf_dict[item['id']]['count'] += 1
                else:
                    rrf_dict[item['id']] = {
                        'id': item['id'],
                        'contents': item['contents'],
                        'score': 1 / (rrf_k + j + 1),
                        'count': 1,
                    }

        rrf_list = sorted(
            rrf_dict.items(),
            key=lambda x: x[1]['score'],
            reverse=True,
        )

        if len(texts) == 1:
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            texts = [
                {key: item[1][key] for key in ('id', 'contents')}
                for item in rrf_list[:k]
            ]
            scores = [item[1]['score'] for item in rrf_list[:k]]

        return texts, scores
