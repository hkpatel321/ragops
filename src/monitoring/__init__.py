"""
Monitoring package for RAGOps:
- Knowledge base drift & staleness detection
- Production query sampling evaluator (faithfulness & answer relevancy)
- Embedding drift detection with Evidently and Prometheus metrics
- Telegram bot alerting and GitHub Actions automated re-indexing
"""

from src.monitoring.alerting import (
    AlertManager,
    TelegramAlerter,
    trigger_github_reindex,
)
from src.monitoring.embedding_drift import (
    EMBEDDING_DISTANCE_SHIFT,
    EMBEDDING_DRIFT_DETECTED,
    EMBEDDING_DRIFT_SCORE,
    EmbeddingDriftDetector,
    EmbeddingDriftResult,
)
from src.monitoring.kb_drift import (
    KB_FILES_CHANGED,
    KB_INDEX_STALE,
    KBDriftDetector,
    KBDriftReport,
)
from src.monitoring.sampling_evaluator import (
    ROLLING_ANSWER_RELEVANCY,
    ROLLING_FAITHFULNESS,
    SAMPLED_QUERIES_COUNT,
    SamplingEvaluator,
)

__all__ = [
    "KBDriftDetector",
    "KBDriftReport",
    "KB_INDEX_STALE",
    "KB_FILES_CHANGED",
    "SamplingEvaluator",
    "ROLLING_FAITHFULNESS",
    "ROLLING_ANSWER_RELEVANCY",
    "SAMPLED_QUERIES_COUNT",
    "EmbeddingDriftDetector",
    "EmbeddingDriftResult",
    "EMBEDDING_DRIFT_SCORE",
    "EMBEDDING_DRIFT_DETECTED",
    "EMBEDDING_DISTANCE_SHIFT",
    "TelegramAlerter",
    "AlertManager",
    "trigger_github_reindex",
]
