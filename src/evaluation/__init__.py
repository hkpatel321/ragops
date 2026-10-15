"""
Evaluation package for RAGOps.
Contains:
- Retrieval metrics (hit_rate, reciprocal_rank, recall, compute_retrieval_metrics)
- GeminiJudge (LangChain Google GenAI wrapper with caching and retry backoff)
- JudgeCache (Disk-backed persistent cache for judge evaluation scores)
- RAGEvaluator (End-to-end evaluator with rate-limit-safe batching)
"""

from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    reciprocal_rank_at_k,
    recall_at_k,
    compute_retrieval_metrics,
)
from src.evaluation.judge_llm import GeminiJudge, JudgeCache
from src.evaluation.evaluator import RAGEvaluator

__all__ = [
    "hit_rate_at_k",
    "reciprocal_rank_at_k",
    "recall_at_k",
    "compute_retrieval_metrics",
    "GeminiJudge",
    "JudgeCache",
    "RAGEvaluator",
]
