"""
Prometheus Metrics Instrumentation for RAGOps API.
Exposes:
- Request count (ragops_requests_total)
- Request latency histogram (ragops_request_latency_seconds)
- Token usage counters (ragops_tokens_total)
- Retrieval similarity score distribution (ragops_retrieval_similarity_score)
"""

from typing import Any, Dict, List

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# 1. Request Counter
REQUEST_COUNT = Counter(
    name="ragops_requests_total",
    documentation="Total count of RAG API requests",
    labelnames=["endpoint", "status", "version_id"],
)

# 2. Latency Histogram
REQUEST_LATENCY = Histogram(
    name="ragops_request_latency_seconds",
    documentation="Latency of RAG API queries in seconds",
    labelnames=["endpoint", "version_id"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)

# 3. Token Usage Counter
TOKEN_USAGE = Counter(
    name="ragops_tokens_total",
    documentation="Total tokens consumed across RAG operations",
    labelnames=["token_type", "model_name"],
)

# 4. Retrieval Score Distribution Histogram
RETRIEVAL_SCORE = Histogram(
    name="ragops_retrieval_similarity_score",
    documentation="Distribution of cosine similarity scores of retrieved chunks",
    labelnames=["version_id"],
    buckets=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)


def record_query_metrics(
    endpoint: str,
    status: str,
    version_id: str,
    latency: float,
    token_counts: Dict[str, int],
    model_name: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> None:
    """
    Record full telemetry for a processed RAG query.
    """
    # Increment request counter
    REQUEST_COUNT.labels(endpoint=endpoint, status=status, version_id=version_id).inc()

    # Record latency
    REQUEST_LATENCY.labels(endpoint=endpoint, version_id=version_id).observe(latency)

    # Record token usage
    prompt_tokens = token_counts.get("prompt_tokens", 0)
    completion_tokens = token_counts.get("completion_tokens", 0)
    total_tokens = token_counts.get("total_tokens", prompt_tokens + completion_tokens)

    if prompt_tokens > 0:
        TOKEN_USAGE.labels(token_type="prompt", model_name=model_name).inc(prompt_tokens)
    if completion_tokens > 0:
        TOKEN_USAGE.labels(token_type="completion", model_name=model_name).inc(completion_tokens)
    if total_tokens > 0:
        TOKEN_USAGE.labels(token_type="total", model_name=model_name).inc(total_tokens)

    # Record retrieval similarity scores
    for chunk in retrieved_chunks:
        score = chunk.get("score")
        if score is not None:
            RETRIEVAL_SCORE.labels(version_id=version_id).observe(float(score))


def get_prometheus_metrics() -> tuple[bytes, str]:
    """
    Export current Prometheus registry in standard exposition format.
    """
    return generate_latest(), CONTENT_TYPE_LATEST
