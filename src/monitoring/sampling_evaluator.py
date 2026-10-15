"""
Async Background Sampling Evaluator for RAGOps.
Samples 10% of production queries and evaluates them using the Gemini Judge:
- Faithfulness
- Answer Relevancy
Exposes rolling scores as Prometheus Gauges:
- ragops_rolling_faithfulness
- ragops_rolling_answer_relevancy
- ragops_sampled_queries_total
"""

import asyncio
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

import numpy as np
from prometheus_client import Counter, Gauge

from src.api.storage import QueryStore
from src.evaluation.judge_llm import GeminiJudge
from src.logger import logger

# Prometheus Metrics for Production Sampling Evaluation
ROLLING_FAITHFULNESS = Gauge(
    name="ragops_rolling_faithfulness",
    documentation="Rolling average faithfulness of sampled production queries",
    labelnames=["window_size"],
)

ROLLING_ANSWER_RELEVANCY = Gauge(
    name="ragops_rolling_answer_relevancy",
    documentation="Rolling average answer relevancy of sampled production queries",
    labelnames=["window_size"],
)

SAMPLED_QUERIES_COUNT = Counter(
    name="ragops_sampled_queries_total",
    documentation="Total count of production queries sampled and evaluated",
    labelnames=["status"],
)


class SamplingEvaluator:
    """
    Monitors logged production queries and asynchronously samples a configurable percentage (default 10%)
    to score with the Judge LLM and update rolling Prometheus metrics.
    """

    def __init__(
        self,
        store: Optional[QueryStore] = None,
        judge: Optional[GeminiJudge] = None,
        sample_rate: float = 0.10,
        window_size: int = 20,
    ) -> None:
        self.store = store or QueryStore()
        self.judge = judge or GeminiJudge()
        self.sample_rate = sample_rate
        self.window_size = window_size

        self.evaluated_ids: Set[str] = set()
        self.faithfulness_window: Deque[float] = deque(maxlen=window_size)
        self.relevancy_window: Deque[float] = deque(maxlen=window_size)

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    def process_new_queries(self, max_fetch: int = 100) -> Dict[str, Any]:
        """
        Poll the QueryStore for newly logged queries, sample 10%, and score them.
        """
        all_recent = self.store.get_recent_queries(limit=max_fetch)
        new_queries = [q for q in all_recent if q["id"] not in self.evaluated_ids]

        if not new_queries:
            return {
                "new_queries_found": 0,
                "sampled_count": 0,
                "rolling_faithfulness": self.get_rolling_faithfulness(),
                "rolling_relevancy": self.get_rolling_relevancy(),
            }

        # Calculate sample size (at least 1 if new queries arrived)
        sample_size = max(1, int(round(len(new_queries) * self.sample_rate)))
        sample_size = min(sample_size, len(new_queries))

        # Sample queries
        sampled = random.sample(new_queries, sample_size)
        logger.info(
            f"Sampling evaluator found {len(new_queries)} new queries. "
            f"Sampling {sample_size} ({self.sample_rate*100:.0f}%) for LLM scoring..."
        )

        scored_records: List[Dict[str, Any]] = []

        for q in sampled:
            q_id = q["id"]
            question = q["query"]
            answer = q.get("answer", "")
            contexts = q.get("contexts", [])

            try:
                # 1. Score Faithfulness
                faith_score, faith_reason = self.judge.evaluate_sample_with_cache(
                    metric_name="faithfulness",
                    question=question,
                    answer=answer,
                    contexts=contexts,
                )
                self.faithfulness_window.append(faith_score)

                # 2. Score Answer Relevancy
                rel_score, rel_reason = self.judge.evaluate_sample_with_cache(
                    metric_name="answer_relevancy",
                    question=question,
                    answer=answer,
                    contexts=contexts,
                )
                self.relevancy_window.append(rel_score)

                SAMPLED_QUERIES_COUNT.labels(status="success").inc()

                scored_records.append({
                    "id": q_id,
                    "faithfulness": faith_score,
                    "faithfulness_reason": faith_reason,
                    "answer_relevancy": rel_score,
                    "answer_relevancy_reason": rel_reason,
                })
            except Exception as e:
                logger.error(f"Error evaluating query sample [{q_id}]: {e}")
                SAMPLED_QUERIES_COUNT.labels(status="error").inc()

        # Mark all fetched new queries as seen so they are not evaluated again
        for q in new_queries:
            self.evaluated_ids.add(q["id"])

        # Update Prometheus rolling gauges
        rolling_f = self.get_rolling_faithfulness()
        rolling_r = self.get_rolling_relevancy()

        ROLLING_FAITHFULNESS.labels(window_size=str(self.window_size)).set(rolling_f)
        ROLLING_ANSWER_RELEVANCY.labels(window_size=str(self.window_size)).set(rolling_r)

        logger.info(
            f"Updated rolling metrics (window={self.window_size}): "
            f"Faithfulness={rolling_f:.4f} | Answer Relevancy={rolling_r:.4f}"
        )

        return {
            "new_queries_found": len(new_queries),
            "sampled_count": len(sampled),
            "rolling_faithfulness": rolling_f,
            "rolling_relevancy": rolling_r,
            "scored_records": scored_records,
        }

    def get_rolling_faithfulness(self) -> float:
        if not self.faithfulness_window:
            return 0.0
        return round(float(np.mean(self.faithfulness_window)), 4)

    def get_rolling_relevancy(self) -> float:
        if not self.relevancy_window:
            return 0.0
        return round(float(np.mean(self.relevancy_window)), 4)

    async def run_loop(self, interval_seconds: float = 10.0) -> None:
        """
        Asynchronous periodic loop to continuously monitor and sample queries.
        """
        self._running = True
        logger.info(f"Starting SamplingEvaluator background loop (interval={interval_seconds}s)...")
        while self._running:
            try:
                self.process_new_queries()
            except Exception as e:
                logger.error(f"Unexpected error in sampling loop: {e}")
            await asyncio.sleep(interval_seconds)

    def start_worker(self, interval_seconds: float = 10.0) -> asyncio.Task:
        """
        Start the background worker task inside an active asyncio event loop.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_loop(interval_seconds=interval_seconds))
            logger.info("Launched SamplingEvaluator background worker task.")
        return self._task

    def stop_worker(self) -> None:
        """
        Gracefully stop the background worker task.
        """
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Stopped SamplingEvaluator background worker task.")
