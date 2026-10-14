"""
Markdown-aware text chunking for RAGOps.
Splits markdown documents along structural boundaries (headers, code blocks, paragraphs)
while enforcing config-driven chunk_size and overlap constraints and preserving rich provenance metadata.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import cfg
from src.ingestion.loader import Document
from src.logger import logger


class TextChunk(BaseModel):
    """
    Representation of a single chunk derived from a Document.
    """
    chunk_id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="Textual content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata inherited from parent document plus chunk coordinates and section context",
    )

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    def __repr__(self) -> str:
        return f"<TextChunk id='{self.chunk_id}' len={len(self.text)} section='{self.metadata.get('section', '')}'>"


class MarkdownChunker:
    """
    Config-driven, markdown-aware document chunker.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_chunk_length: int = 50,
    ) -> None:
        self.chunk_size = chunk_size or cfg.chunking.chunk_size
        self.overlap = overlap if overlap is not None else cfg.chunking.overlap
        self.min_chunk_length = min_chunk_length

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def _split_into_sections(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Split markdown text by header boundaries (# , ## , ### ), keeping section titles.
        """
        lines = markdown_text.splitlines()
        sections: List[Dict[str, str]] = []
        current_title = "Introduction"
        current_lines: List[str] = []

        header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                # Flush previous section
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        sections.append({"title": current_title, "content": text_block})
                    current_lines = []
                current_title = match.group(2).strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                sections.append({"title": current_title, "content": text_block})

        return sections if sections else [{"title": "General", "content": markdown_text}]

    def _split_text_with_overlap(self, text: str) -> List[str]:
        """
        Break text exceeding chunk_size into overlapping chunks, respecting paragraphs and sentences.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        # Attempt to split on paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk: List[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len + 2 <= self.chunk_size:
                current_chunk.append(p)
                current_len += p_len + 2
            else:
                if current_chunk:
                    full_chunk_text = "\n\n".join(current_chunk).strip()
                    if full_chunk_text:
                        chunks.append(full_chunk_text)

                # If single paragraph is larger than chunk_size, split by sentences or characters
                if p_len > self.chunk_size:
                    step = self.chunk_size - self.overlap
                    for start in range(0, p_len, step):
                        end = min(start + self.chunk_size, p_len)
                        sub_text = p[start:end].strip()
                        if len(sub_text) >= self.min_chunk_length:
                            chunks.append(sub_text)
                    current_chunk = []
                    current_len = 0
                else:
                    # Start new chunk with overlap from previous if feasible
                    current_chunk = [p]
                    current_len = p_len

        if current_chunk:
            final_text = "\n\n".join(current_chunk).strip()
            if final_text and (not chunks or final_text != chunks[-1]):
                chunks.append(final_text)

        return chunks

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Chunk a single Document into a list of TextChunk objects.

        Args:
            document: Ingested Document object.

        Returns:
            List of TextChunk instances with preserved metadata.
        """
        sections = self._split_into_sections(document.content)
        all_chunks: List[TextChunk] = []
        chunk_idx = 0

        doc_base_id = Path(document.metadata.get("filename", "doc")).stem

        for sec in sections:
            sec_title = sec["title"]
            sec_content = sec["content"]

            sub_chunks = self._split_text_with_overlap(sec_content)

            for sub_text in sub_chunks:
                if len(sub_text.strip()) < self.min_chunk_length:
                    continue

                chunk_id = f"{document.version}_{document.source}_{doc_base_id}_{chunk_idx:04d}"

                chunk_metadata = dict(document.metadata)
                chunk_metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "section": sec_title,
                        "chunk_char_count": len(sub_text),
                    }
                )

                all_chunks.append(TextChunk(chunk_id=chunk_id, text=sub_text, metadata=chunk_metadata))
                chunk_idx += 1

        return all_chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Batch chunk a list of documents.

        Args:
            documents: List of Document instances.

        Returns:
            Flat list of TextChunk instances.
        """
        results: List[TextChunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            results.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(results)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )
        return results
