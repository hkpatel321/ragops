"""
Data Ingestion module for RAGOps.
"""

from src.ingestion.loader import (
    Document,
    extract_title,
    load_documents_from_directory,
    load_markdown_file,
    load_versioned_docs,
)

__all__ = [
    "Document",
    "extract_title",
    "load_markdown_file",
    "load_documents_from_directory",
    "load_versioned_docs",
]
