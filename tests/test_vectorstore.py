"""
Unit tests for ChromaDB vector store, version tagging, and retrieval.
"""

import unittest
from src.chunking.markdown_chunker import TextChunk
from src.vectorstore.store import ChromaStore, compute_config_hash, compute_version_id


class TestVectorStore(unittest.TestCase):
    def test_compute_config_hash_deterministic(self):
        hash1 = compute_config_hash()
        hash2 = compute_config_hash()
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 8)

    def test_compute_version_id(self):
        v1_id = compute_version_id("v1")
        v2_id = compute_version_id("v2")
        self.assertTrue(v1_id.startswith("v1_"))
        self.assertTrue(v2_id.startswith("v2_"))
        self.assertNotEqual(v1_id, v2_id)

    def test_chroma_store_add_and_query(self):
        store = ChromaStore(
            persist_directory="data/test_chroma_db",
            collection_name="test_collection",
        )
        version_id = "test_v1"

        # Create dummy chunks
        chunks = [
            TextChunk(
                chunk_id="test_chunk_001",
                text="FastAPI supports dependency injection and async endpoint definitions.",
                metadata={"source": "fastapi", "version": "v1", "topic": "dependency"},
            ),
            TextChunk(
                chunk_id="test_chunk_002",
                text="MLflow tracking provides logging for parameters, metrics, and models.",
                metadata={"source": "mlflow", "version": "v1", "topic": "tracking"},
            ),
        ]

        indexed = store.add_chunks(chunks=chunks, version_id=version_id)
        self.assertEqual(indexed, 2)

        # Test querying
        results = store.query(
            query_text="How does MLflow tracking log parameters?",
            version_id=version_id,
            top_k=2,
        )
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("mlflow", top_match["text"].lower())
        self.assertIn("score", top_match)
        self.assertIn("chunk_id", top_match)

        # Cleanup
        store.reset_collection(version_id=version_id)


if __name__ == "__main__":
    unittest.main()
