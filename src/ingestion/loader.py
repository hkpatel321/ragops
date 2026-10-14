"""
Document Loader module for RAGOps.
Recursively parses raw markdown documentation files and extracts structured Document objects
retaining complete provenance metadata (source, version, file path, title, char count).
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.logger import logger


class Document(BaseModel):
    """
    Representation of an ingested document with full metadata.
    """
    content: str = Field(description="Raw text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary including source, version, path, and filename",
    )

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    @property
    def path(self) -> str:
        return self.metadata.get("path", "")

    def __repr__(self) -> str:
        return f"<Document source='{self.source}' version='{self.version}' path='{self.path}' len={len(self.content)}>"


def extract_title(content: str, fallback_filename: str) -> str:
    """
    Extract the first H1/H2 header from markdown text, or fall back to the filename.
    """
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()
    return Path(fallback_filename).stem.replace("_", " ").title()


def load_markdown_file(
    file_path: Union[str, Path],
    source: Optional[str] = None,
    version: Optional[str] = None,
) -> Document:
    """
    Load a single markdown file into a Document object with inferred or explicit metadata.

    Args:
        file_path: Path to the markdown file.
        source: Explicit documentation source ('fastapi', 'mlflow'). If None, inferred from path.
        version: Explicit version ('v1', 'v2'). If None, inferred from path.

    Returns:
        Loaded Document object.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"File not found: {path_obj}")

    content = path_obj.read_text(encoding="utf-8", errors="replace")

    # Infer metadata from path hierarchy if not provided
    # Expected structure: .../data/raw/{version}/{source}/{filename}
    parts = path_obj.parts
    inferred_version = version
    inferred_source = source

    if not inferred_version:
        for p in parts:
            if p.lower() in ("v1", "v2"):
                inferred_version = p.lower()
                break
        if not inferred_version:
            inferred_version = "v1"

    if not inferred_source:
        for p in parts:
            if p.lower() in ("fastapi", "mlflow"):
                inferred_source = p.lower()
                break
        if not inferred_source:
            inferred_source = "unknown"

    title = extract_title(content, path_obj.name)

    # Use forward slashes for cross-platform consistency
    relative_path = str(path_obj).replace("\\", "/")

    metadata: Dict[str, Any] = {
        "source": inferred_source,
        "version": inferred_version,
        "path": relative_path,
        "filename": path_obj.name,
        "title": title,
        "char_count": len(content),
    }

    return Document(content=content, metadata=metadata)


def load_documents_from_directory(
    dir_path: Union[str, Path],
    source: Optional[str] = None,
    version: Optional[str] = None,
) -> List[Document]:
    """
    Recursively read all markdown (*.md) files in a directory and return Document list.

    Args:
        dir_path: Root directory to search.
        source: Optional source override.
        version: Optional version override.

    Returns:
        List of Document instances.
    """
    path_obj = Path(dir_path)
    if not path_obj.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    documents: List[Document] = []
    # Collect all .md and .markdown files
    md_files = sorted(list(path_obj.rglob("*.md")) + list(path_obj.rglob("*.markdown")))

    for md_file in md_files:
        try:
            doc = load_markdown_file(md_file, source=source, version=version)
            if doc.content.strip():
                documents.append(doc)
        except Exception as exc:
            logger.error(f"Error loading {md_file}: {exc}")

    logger.info(f"Loaded {len(documents)} documents from {dir_path}")
    return documents


def load_versioned_docs(base_raw_dir: Union[str, Path] = "data/raw") -> Dict[str, List[Document]]:
    """
    Load all documents organized by version key ('v1', 'v2').

    Args:
        base_raw_dir: Path to data/raw.

    Returns:
        Dictionary mapping version ('v1', 'v2') to list of Document instances.
    """
    raw_path = Path(base_raw_dir)
    result: Dict[str, List[Document]] = {"v1": [], "v2": []}

    for ver in ["v1", "v2"]:
        ver_dir = raw_path / ver
        if ver_dir.exists():
            docs = load_documents_from_directory(ver_dir, version=ver)
            result[ver] = docs
            logger.info(f"Version [{ver}] loaded with {len(docs)} documents.")

    return result
