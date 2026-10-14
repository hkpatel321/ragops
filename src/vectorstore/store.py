"""
ChromaDB Vector Store management for RAGOps.
Supports versioned collection indexing, metadata-filtered semantic search,
and version tracking (data_version + config_hash).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src.chunking.markdown_chunker import TextChunk
from src.config import AppConfig, cfg
from src.embedding.embedder import ChromaEmbeddingFunction, Embedder, get_embedder
from src.logger import logger


def compute_config_hash(config: Optional[AppConfig] = None) -> str:
    """
    Generate an 8-character deterministic hash of all indexing parameters.

    Args:
        config: Optional AppConfig instance (defaults to global cfg).

    Returns:
        Hexadecimal hash string.
    """
    c = config or cfg
    hash_payload = {
        "chunk_size": c.chunking.chunk_size,
        "overlap": c.chunking.overlap,
        "embedding_model": c.embedding.model_name,
        "distance_metric": c.vectorstore.distance_metric,
    }
    encoded = json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def compute_version_id(data_version: str, config: Optional[AppConfig] = None) -> str:
    """
    Build standardized version identifier combining data version and configuration hash.
    Format: {data_version}_{config_hash} (e.g. 'v1_a3f8c9b1').

    Args:
        data_version: Data version tag (e.g., 'v1', 'v2').
        config: Optional AppConfig instance.

    Returns:
        Composite version identifier.
    """
    cfg_hash = compute_config_hash(config)
    return f"{data_version}_{cfg_hash}"


class ChromaStore:
    """
    Production ChromaDB client managing index lifecycle, versioning, and retrieval.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory or cfg.vectorstore.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.base_collection_name = collection_name or cfg.vectorstore.collection_name

        self.embedder = embedder or get_embedder()
        self.embedding_function = ChromaEmbeddingFunction(self.embedder)

        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    def get_or_create_collection(
        self,
        version_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> chromadb.Collection:
        """
        Retrieve or create a version-tagged ChromaDB collection.

        Args:
            version_id: Composite version tag (data_version + config_hash).
            collection_name: Explicit collection name override.
            metadata: Additional collection-level metadata.

        Returns:
            ChromaDB Collection instance.
        """
        # If version_id given, embed it into collection name for isolated versioning
        if collection_name:
            c_name = collection_name
        elif version_id:
            c_name = f"{self.base_collection_name}_{version_id}"
        else:
            c_name = self.base_collection_name

        # Distance metric configuration
        metric = cfg.vectorstore.distance_metric
        col_metadata = {"hnsw:space": metric}
        if version_id:
            col_metadata["version_id"] = version_id
        if metadata:
            col_metadata.update(metadata)

        collection = self.client.get_or_create_collection(
            name=c_name,
            embedding_function=self.embedding_function,
            metadata=col_metadata,
        )
        return collection

    def add_chunks(
        self,
        chunks: List[TextChunk],
        version_id: str,
        collection_name: Optional[str] = None,
        batch_size: int = 128,
    ) -> int:
        """
        Embed and insert chunks into the versioned Chroma collection.

        Args:
            chunks: List of TextChunk objects.
            version_id: Version identifier tag.
            collection_name: Optional collection name override.
            batch_size: Insertion batch size.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        collection = self.get_or_create_collection(
            version_id=version_id,
            collection_name=collection_name,
            metadata={"version_id": version_id, "total_chunks": len(chunks)},
        )

        total = len(chunks)
        logger.info(f"Indexing {total} chunks into collection '{collection.name}' (version_id='{version_id}')...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]

            # Flatten and sanitize metadata (Chroma only accepts primitive scalar types)
            metadatas = []
            for c in batch:
                meta = {}
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                meta["version_id"] = version_id
                metadatas.append(meta)

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(f"Successfully indexed {total} chunks into Chroma collection '{collection.name}'.")
        return total

    def query(
        self,
        query_text: str,
        version_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity retrieval against the index.

        Args:
            query_text: The user query string.
            version_id: Version ID to query against.
            collection_name: Explicit collection name override.
            top_k: Number of nearest neighbors to return.
            where_filter: Optional Chroma metadata filter dictionary.

        Returns:
            List of result dictionaries containing chunk_id, document, metadata, and distance.
        """
        k = top_k or cfg.vectorstore.top_k
        collection = self.get_or_create_collection(
            version_id=version_id,
            collection_name=collection_name,
        )

        query_results = collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        if not query_results["ids"] or not query_results["ids"][0]:
            return results

        ids = query_results["ids"][0]
        docs = query_results["documents"][0] if query_results["documents"] else [""] * len(ids)
        metas = query_results["metadatas"][0] if query_results["metadatas"] else [{}] * len(ids)
        distances = query_results["distances"][0] if query_results["distances"] else [0.0] * len(ids)

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # For cosine distance: similarity = 1 - distance
            similarity = 1.0 - dist if cfg.vectorstore.distance_metric == "cosine" else dist
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                    "score": round(similarity, 4),
                }
            )

        return results

    def count(self, version_id: Optional[str] = None, collection_name: Optional[str] = None) -> int:
        """
        Get total number of vectors in collection.
        """
        collection = self.get_or_create_collection(version_id=version_id, collection_name=collection_name)
        return collection.count()

    def reset_collection(self, version_id: Optional[str] = None, collection_name: Optional[str] = None) -> None:
        """
        Delete and clear collection for rebuilds.
        """
        if collection_name:
            c_name = collection_name
        elif version_id:
            c_name = f"{self.base_collection_name}_{version_id}"
        else:
            c_name = self.base_collection_name

        try:
            self.client.delete_collection(name=c_name)
            logger.info(f"Deleted collection '{c_name}'.")
        except Exception as exc:
            logger.debug(f"Collection '{c_name}' not found for deletion: {exc}")
