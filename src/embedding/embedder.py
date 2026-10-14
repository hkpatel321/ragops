"""
Local Sentence-Transformers embedding wrapper for RAGOps.
Wraps open-source sentence-transformers (e.g. all-MiniLM-L6-v2 or bge-small-en-v1.5)
providing thread-safe, batch-friendly embedding generation and ChromaDB integration.
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import cfg
from src.logger import logger


class Embedder:
    """
    Config-driven sentence-transformers wrapper for dense vector representations.
    """

    _instance: Optional["Embedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or cfg.embedding.model_name
        self.device = device or cfg.embedding.device
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None else cfg.embedding.normalize_embeddings
        )
        self.batch_size = batch_size or cfg.embedding.batch_size

        if Embedder._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
            Embedder._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer model loaded successfully.")

        self.model = Embedder._model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single string.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        """
        Dimension of the generated embedding vectors.
        """
        dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384


from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class ChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    ChromaDB compatible embedding function wrapper for Embedder.
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or Embedder()

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB expects a callable that takes a list of strings
        and returns a list of float vectors.
        """
        if isinstance(input, str):
            input = [input]
        return self.embedder.embed_batch(list(input))

    def name(self) -> str:
        return f"sentence_transformer_{self.embedder.model_name.replace('/', '_')}"


# Default shared embedder instance helper
_default_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """
    Return singleton instance of Embedder.
    """
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
