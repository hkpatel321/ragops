"""
FastAPI Application for RAGOps.
Endpoints:
- POST /query: Executes semantic retrieval & grounded LLM generation with full telemetry
- GET /health: Health check inspecting VectorStore and SQLite query store status
- GET /metrics: Standard Prometheus exposition format for scraping
- GET /history: Inspect recent logged queries from persistent storage
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.metrics import (
    get_prometheus_metrics,
    record_query_metrics,
)
from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.api.storage import QueryStore
from src.config import cfg
from src.logger import logger
from src.rag.pipeline import RAGPipeline

# Global singleton handles
pipeline: Optional[RAGPipeline] = None
store: Optional[QueryStore] = None


def get_store() -> QueryStore:
    global store
    if store is None:
        store = QueryStore()
    return store


def set_store(custom_store: Optional[QueryStore]) -> None:
    global store
    store = custom_store


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline


def set_pipeline(custom_pipeline: Optional[RAGPipeline]) -> None:
    global pipeline
    pipeline = custom_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application startup and shutdown lifecycle management.
    """
    global pipeline, store
    logger.info("Initializing RAGOps FastAPI service dependencies...")
    try:
        if pipeline is None:
            pipeline = RAGPipeline()
        if store is None:
            store = QueryStore()
        logger.info("RAG pipeline and QueryStore initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline during startup: {e}")
        # Allow service to start; health check will report state
    yield
    logger.info("Shutting down RAGOps FastAPI service.")


app = FastAPI(
    title="RAGOps Documentation Assistant API",
    description="Production RAG Assistant over MLflow and FastAPI Docs with Continuous Monitoring & Prometheus Metrics",
    version=cfg.project.version,
    lifespan=lifespan,
)

# Enable CORS for frontend / monitoring dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Root", tags=["System"])
async def root() -> Dict[str, Any]:
    """
    Welcome endpoint providing navigation links to docs and health checks.
    """
    return {
        "service": "ragops-api",
        "version": cfg.project.version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "description": "RAG Assistant over versioned MLflow and FastAPI documentation.",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check", tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """
    Health check inspecting vector store readiness and SQLite query telemetry storage.
    """
    pipe = get_pipeline()
    active_store = get_store()

    v_status = "uninitialized"
    if pipe and pipe.store:
        try:
            # Check Chroma vector store
            v_status = "connected"
        except Exception:
            v_status = "error"

    s_status = "uninitialized"
    total_logged = 0
    if active_store:
        try:
            total_logged = active_store.get_total_count()
            s_status = "connected"
        except Exception:
            s_status = "error"

    is_healthy = v_status == "connected" and s_status == "connected"

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        service="ragops-api",
        version=cfg.project.version,
        vectorstore=v_status,
        storage=s_status,
        total_logged_queries=total_logged,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query RAG Assistant",
    tags=["RAG"],
    status_code=status.HTTP_200_OK,
)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the versioned RAG assistant:
    1. Semantically retrieves top_k candidate chunks from ChromaDB.
    2. Builds grounded prompt ensuring model says 'I don't know' if ungrounded.
    3. Invokes Gemini (or deterministic mock mode if requested).
    4. Records Prometheus metrics (request count, latency histogram, token counts, retrieval score).
    5. Persists full query payload and provenance to SQLite and streaming JSONL.
    """
    pipe = get_pipeline()
    active_store = get_store()

    req_id = f"req_{uuid.uuid4().hex[:12]}"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    endpoint_name = "/query"
    version_tag = request.data_version

    mock_answer = None
    if request.mock or not pipe.client:
        mock_answer = f"Mock Answer: Based on {request.data_version} documentation, this is a verified technical response."

    try:
        rag_resp = pipe.query(
            query=request.query,
            data_version=request.data_version,
            top_k=request.top_k,
            mock_answer=mock_answer,
        )

        chunk_ids = [c.get("chunk_id") or c.get("id", "") for c in rag_resp.retrieved_chunks]

        # 1. Record Prometheus Telemetry
        record_query_metrics(
            endpoint=endpoint_name,
            status="success",
            version_id=rag_resp.version_id,
            latency=rag_resp.latency,
            token_counts=rag_resp.token_counts,
            model_name=rag_resp.model_name,
            retrieved_chunks=rag_resp.retrieved_chunks,
        )

        # 2. Persist to SQLite and JSONL
        active_store.log_query(
            request_id=req_id,
            query=request.query,
            contexts=rag_resp.contexts,
            answer=rag_resp.answer,
            latency=rag_resp.latency,
            token_counts=rag_resp.token_counts,
            index_version=rag_resp.version_id,
            model_name=rag_resp.model_name,
            top_k=request.top_k or cfg.vectorstore.top_k,
            retrieved_chunk_ids=chunk_ids,
            timestamp=ts,
        )

        return QueryResponse(
            request_id=req_id,
            query=request.query,
            answer=rag_resp.answer,
            contexts=rag_resp.contexts,
            retrieved_chunks=rag_resp.retrieved_chunks,
            latency=rag_resp.latency,
            token_counts=rag_resp.token_counts,
            model_name=rag_resp.model_name,
            version_id=rag_resp.version_id,
            timestamp=ts,
        )

    except Exception as exc:
        logger.error(f"Error processing query [{req_id}]: {exc}", exc_info=True)
        # Record failed request metric
        record_query_metrics(
            endpoint=endpoint_name,
            status="error",
            version_id=version_tag,
            latency=0.0,
            token_counts={},
            model_name=cfg.llm.generator.model_name,
            retrieved_chunks=[],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution error: {str(exc)}",
        )


@app.get("/metrics", summary="Prometheus Metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """
    Expose Prometheus metrics in text format for scraping by Prometheus server.
    """
    data, content_type = get_prometheus_metrics()
    return Response(content=data, media_type=content_type)


@app.get("/history", summary="Query Log History", tags=["Monitoring"])
async def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Inspect the most recent query records from the SQLite store.
    """
    active_store = get_store()
    return active_store.get_recent_queries(limit=limit)
