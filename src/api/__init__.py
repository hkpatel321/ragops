"""
API package for RAGOps.
Exposes FastAPI application, schemas, query store, and Prometheus metrics.
"""

from src.api.app import app
from src.api.metrics import get_prometheus_metrics, record_query_metrics
from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.api.storage import QueryStore

__all__ = [
    "app",
    "QueryStore",
    "record_query_metrics",
    "get_prometheus_metrics",
    "QueryRequest",
    "QueryResponse",
    "HealthResponse",
]
