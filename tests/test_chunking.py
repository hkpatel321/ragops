"""
Unit tests for markdown-aware chunking and metadata preservation.
"""

import unittest

from src.chunking.markdown_chunker import MarkdownChunker, TextChunk
from src.ingestion.loader import Document


class TestMarkdownChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = MarkdownChunker(chunk_size=200, overlap=40, min_chunk_length=30)
        self.sample_doc = Document(
            content=(
                "# Overview\n\n"
                "FastAPI is a modern, fast web framework for building APIs with Python.\n\n"
                "## Key Features\n\n"
                "Fast to code: Increase the speed to develop features by about 200% to 300%.\n\n"
                "Fewer bugs: Reduce about 40% of human (developer) induced errors.\n\n"
                "### Performance\n\n"
                "Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)."
            ),
            metadata={
                "source": "fastapi",
                "version": "v1",
                "path": "data/raw/v1/fastapi/index.md",
                "filename": "index.md",
                "title": "FastAPI Framework",
            },
        )

    def test_chunk_document_preserves_metadata(self):
        chunks = self.chunker.chunk_document(self.sample_doc)
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertEqual(chunk.source, "fastapi")
            self.assertEqual(chunk.version, "v1")
            self.assertIn("section", chunk.metadata)
            self.assertIn("chunk_id", chunk.metadata)
            self.assertTrue(chunk.chunk_id.startswith("v1_fastapi_index_"))

    def test_chunk_size_and_overlap_validation(self):
        with self.assertRaises(ValueError):
            # Overlap cannot be >= chunk_size
            MarkdownChunker(chunk_size=100, overlap=100)

    def test_chunk_document_sections_detected(self):
        chunks = self.chunker.chunk_document(self.sample_doc)
        sections = {c.metadata["section"] for c in chunks}
        # Verify headers were recognized as sections
        self.assertTrue(any("Overview" in s or "Features" in s or "Performance" in s for s in sections))


if __name__ == "__main__":
    unittest.main()
