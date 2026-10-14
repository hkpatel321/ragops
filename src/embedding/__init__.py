"""
Embedding module for RAGOps.
"""

from src.embedding.embedder import (
    ChromaEmbeddingFunction,
    Embedder,
    get_embedder,
)

__all__ = ["Embedder", "ChromaEmbeddingFunction", "get_embedder"]
