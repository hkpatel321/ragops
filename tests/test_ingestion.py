"""
Unit tests for data ingestion loader and metadata extraction.
"""

import unittest
from pathlib import Path
from src.ingestion.loader import (
    Document,
    extract_title,
    load_documents_from_directory,
    load_markdown_file,
    load_versioned_docs,
)


class TestIngestionLoader(unittest.TestCase):
    def test_extract_title_h1(self):
        text = "# FastAPI Framework\nFastAPI is modern and fast."
        title = extract_title(text, "fallback.md")
        self.assertEqual(title, "FastAPI Framework")

    def test_extract_title_h2_and_fallback(self):
        text = "## Secondary Header\nSome details."
        title = extract_title(text, "fallback.md")
        self.assertEqual(title, "Secondary Header")

        fallback_title = extract_title("No headers here", "tutorial_first_steps.md")
        self.assertEqual(fallback_title, "Tutorial First Steps")

    def test_load_markdown_file_metadata(self):
        sample_path = Path("data/raw/v1/fastapi/index.md")
        if sample_path.exists():
            doc = load_markdown_file(sample_path)
            self.assertIsInstance(doc, Document)
            self.assertEqual(doc.source, "fastapi")
            self.assertEqual(doc.version, "v1")
            self.assertEqual(doc.metadata["filename"], "index.md")
            self.assertGreater(len(doc.content), 1000)
            self.assertTrue(doc.path.endswith("data/raw/v1/fastapi/index.md"))

    def test_load_documents_from_directory(self):
        dir_path = Path("data/raw/v1/mlflow")
        if dir_path.exists():
            docs = load_documents_from_directory(dir_path)
            self.assertGreater(len(docs), 0)
            for d in docs:
                self.assertEqual(d.source, "mlflow")
                self.assertEqual(d.version, "v1")
                self.assertTrue(len(d.content) > 0)

    def test_load_versioned_docs(self):
        versioned = load_versioned_docs("data/raw")
        self.assertIn("v1", versioned)
        self.assertIn("v2", versioned)
        self.assertGreater(len(versioned["v1"]), 0)
        self.assertGreater(len(versioned["v2"]), 0)

        # Check sources in v1 and v2
        v1_sources = {d.source for d in versioned["v1"]}
        v2_sources = {d.source for d in versioned["v2"]}
        self.assertIn("fastapi", v1_sources)
        self.assertIn("mlflow", v1_sources)
        self.assertIn("fastapi", v2_sources)
        self.assertIn("mlflow", v2_sources)


if __name__ == "__main__":
    unittest.main()
