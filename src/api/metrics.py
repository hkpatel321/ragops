"""
Prometheus Metrics Instrumentation for RAGOps API.
Exposes:
- Request count (ragops_requests_total)
- Request latency histogram (ragops_request_latency_seconds)
- Token usage counters (ragops_tokens_total)
- Retrieval similarity score distribution (ragops_retrieval_similarity_score)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.logger import logger

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

# 4. Token Cost Counter (in USD)
TOKEN_COST_USD = Counter(
    name="ragops_token_cost_usd_total",
    documentation="Estimated token cost in USD across RAG operations",
    labelnames=["model_name", "token_type"],
)

# 5. Retrieval Score Distribution Histogram
RETRIEVAL_SCORE = Histogram(
    name="ragops_retrieval_similarity_score",
    documentation="Distribution of cosine similarity scores of retrieved chunks",
    labelnames=["version_id"],
    buckets=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# 6. Retrieval Offline Evaluation Metrics
RETRIEVAL_HIT_RATE = Gauge(
    name="ragops_retrieval_hit_rate",
    documentation="Offline evaluation retrieval hit rate at k",
    labelnames=["version_id", "k"],
)

RETRIEVAL_MRR = Gauge(
    name="ragops_retrieval_mrr",
    documentation="Offline evaluation Mean Reciprocal Rank (MRR) at k",
    labelnames=["version_id", "k"],
)

# 7. Active Index Version Info Gauge
ACTIVE_INDEX_INFO = Gauge(
    name="ragops_active_index_info",
    documentation="Information indicator of active ChromaDB index version (value 1.0 when active)",
    labelnames=["version_id", "data_version", "status"],
)

# Gemini pricing estimates (USD per token)
PROMPT_COST_PER_TOKEN = 0.075 / 1_000_000
COMPLETION_COST_PER_TOKEN = 0.30 / 1_000_000


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

    # Record token usage and cost
    prompt_tokens = token_counts.get("prompt_tokens", 0)
    completion_tokens = token_counts.get("completion_tokens", 0)
    total_tokens = token_counts.get("total_tokens", prompt_tokens + completion_tokens)

    if prompt_tokens > 0:
        TOKEN_USAGE.labels(token_type="prompt", model_name=model_name).inc(prompt_tokens)
        TOKEN_COST_USD.labels(model_name=model_name, token_type="prompt").inc(prompt_tokens * PROMPT_COST_PER_TOKEN)
    if completion_tokens > 0:
        TOKEN_USAGE.labels(token_type="completion", model_name=model_name).inc(completion_tokens)
        TOKEN_COST_USD.labels(model_name=model_name, token_type="completion").inc(completion_tokens * COMPLETION_COST_PER_TOKEN)
    if total_tokens > 0:
        TOKEN_USAGE.labels(token_type="total", model_name=model_name).inc(total_tokens)

    # Record retrieval similarity scores
    for chunk in retrieved_chunks:
        score = chunk.get("score")
        if score is not None:
            RETRIEVAL_SCORE.labels(version_id=version_id).observe(float(score))


def sync_eval_and_index_metrics() -> None:
    """
    Sync latest offline evaluation metrics and active index version to Prometheus registry.
    """
    # 1. Sync active index version info
    active_path = Path("data/active_index_version.json")
    if active_path.is_file():
        try:
            with open(active_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            ACTIVE_INDEX_INFO.labels(
                version_id=info.get("version_id", "v1_promoted"),
                data_version=info.get("active_data_version", "v1"),
                status=info.get("status", "PROMOTED"),
            ).set(1.0)
        except Exception as e:
            logger.debug(f"Failed to read active index pointer: {e}")
    else:
        ACTIVE_INDEX_INFO.labels(
            version_id="v1_9ac1e3a5",
            data_version="v1",
            status="active",
        ).set(1.0)

    # 2. Sync latest offline evaluation report if available
    eval_path = Path("eval/eval_report_latest.json")
    if eval_path.is_file():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            metrics = eval_data.get("metrics", {})
            version_id = eval_data.get("version_id", "v1_9ac1e3a5")

            # Hit rate & MRR across common k values
            for k in [2, 4, 5, 10]:
                if f"hit_rate@{k}" in metrics:
                    RETRIEVAL_HIT_RATE.labels(version_id=version_id, k=str(k)).set(float(metrics[f"hit_rate@{k}"]))
                if f"mrr@{k}" in metrics:
                    RETRIEVAL_MRR.labels(version_id=version_id, k=str(k)).set(float(metrics[f"mrr@{k}"]))
        except Exception as e:
            logger.debug(f"Failed to sync eval report metrics: {e}")


# Initialize defaults on module import
sync_eval_and_index_metrics()


def get_prometheus_metrics() -> tuple[bytes, str]:
    """
    Export current Prometheus registry in standard exposition format.
    """
    sync_eval_and_index_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST
