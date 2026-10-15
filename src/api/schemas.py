"""
Pydantic Request & Response Schemas for RAGOps FastAPI App.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The technical question to ask the documentation assistant",
        example="How do path parameters work in FastAPI?",
    )
    data_version: str = Field(
        default="v1",
        description="Target documentation version (e.g., 'v1' or 'v2')",
        example="v1",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve (defaults to config: 4)",
        example=4,
    )
    mock: bool = Field(
        default=False,
        description="If True, runs generation in deterministic mock mode without invoking Gemini API",
    )


class ChunkInfo(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    contexts: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    latency: float
    token_counts: Dict[str, int]
    model_name: str
    version_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    vectorstore: str
    storage: str
    total_logged_queries: int
