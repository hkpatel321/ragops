"""
End-to-End Evaluation Module for RAGOps.
Coordinates retrieval and generation evaluation:
- Retrieval metrics (Hit Rate@k, MRR@k, Recall@k)
- Generation metrics (Faithfulness, Answer Relevancy, Context Precision)
  using Gemini Judge wrapped with LangChain Google GenAI integration.
- Rate-limit safe batching and caching.
- Quality gate validation against configurable thresholds.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import cfg
from src.evaluation.judge_llm import GeminiJudge
from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.logger import logger


class RAGEvaluator:
    """
    Evaluates RAG pipeline outputs on both retrieval quality and generation quality.
    """

    def __init__(
        self,
        judge: Optional[GeminiJudge] = None,
        k: Optional[int] = None,
        batch_size: int = 5,
        batch_delay: float = 0.5,
    ) -> None:
        self.judge = judge or GeminiJudge()
        self.k = k if k is not None else cfg.vectorstore.top_k
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    def evaluate_sample(
        self,
        question: str,
        generated_answer: str,
        retrieved_chunk_ids: List[str],
        retrieved_contexts: List[str],
        ground_truth_chunk_id: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG query result.
        """
        # 1. Retrieval metrics for this query
        hit = hit_rate_at_k(retrieved_chunk_ids, ground_truth_chunk_id or "", k=self.k)
        rr = reciprocal_rank_at_k(retrieved_chunk_ids, ground_truth_chunk_id or "", k=self.k)
        recall = recall_at_k(retrieved_chunk_ids, ground_truth_chunk_id or "", k=self.k)

        # 2. Generation metrics via Gemini Judge (cached & rate-limit safe)
        faithfulness, faith_reason = self.judge.evaluate_sample_with_cache(
            metric_name="faithfulness",
            question=question,
            answer=generated_answer,
            contexts=retrieved_contexts,
            ground_truth=ground_truth_answer,
        )

        relevancy, rel_reason = self.judge.evaluate_sample_with_cache(
            metric_name="answer_relevancy",
            question=question,
            answer=generated_answer,
            contexts=retrieved_contexts,
            ground_truth=ground_truth_answer,
        )

        ctx_prec, ctx_reason = self.judge.evaluate_sample_with_cache(
            metric_name="context_precision",
            question=question,
            answer=generated_answer,
            contexts=retrieved_contexts,
            ground_truth=ground_truth_answer,
        )

        return {
            "question": question,
            "generated_answer": generated_answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "ground_truth_chunk_id": ground_truth_chunk_id,
            "ground_truth_answer": ground_truth_answer,
            f"hit_rate@{self.k}": hit,
            f"mrr@{self.k}": rr,
            f"recall@{self.k}": recall,
            "faithfulness": round(faithfulness, 4),
            "faithfulness_reason": faith_reason,
            "answer_relevancy": round(relevancy, 4),
            "answer_relevancy_reason": rel_reason,
            "context_precision": round(ctx_prec, 4),
            "context_precision_reason": ctx_reason,
        }

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a list of sample records with rate-limit-safe batching.

        Each sample must include:
        - 'question': str
        - 'generated_answer': str
        - 'retrieved_chunk_ids': List[str]
        - 'retrieved_contexts': List[str]
        Optional:
        - 'ground_truth_chunk_id' or 'chunk_id'
        - 'ground_truth_answer' or 'ground_truth'
        """
        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]

        total_samples = len(samples)
        logger.info(f"Starting evaluation of {total_samples} samples (batch_size={self.batch_size}, top_k={self.k})...")

        detailed_results: List[Dict[str, Any]] = []

        for i in range(0, total_samples, self.batch_size):
            batch = samples[i : i + self.batch_size]
            logger.info(f"Evaluating batch {i // self.batch_size + 1}/{(total_samples + self.batch_size - 1) // self.batch_size} (samples {i+1} to {min(i + len(batch), total_samples)})...")

            for item in batch:
                gt_chunk = item.get("ground_truth_chunk_id") or item.get("chunk_id") or ""
                gt_ans = item.get("ground_truth_answer") or item.get("ground_truth") or ""

                res = self.evaluate_sample(
                    question=item["question"],
                    generated_answer=item.get("generated_answer", ""),
                    retrieved_chunk_ids=item.get("retrieved_chunk_ids", []),
                    retrieved_contexts=item.get("retrieved_contexts", []),
                    ground_truth_chunk_id=gt_chunk,
                    ground_truth_answer=gt_ans,
                )
                detailed_results.append(res)

            # Pacing delay between batches to stay within rate limits
            if i + self.batch_size < total_samples and self.batch_delay > 0:
                time.sleep(self.batch_delay)

        # Compute aggregate metrics
        retrieval_agg = compute_retrieval_metrics(
            eval_records=[
                {
                    "retrieved_chunk_ids": r.get("retrieved_chunk_ids", []),
                    "ground_truth_chunk_id": r.get("ground_truth_chunk_id", ""),
                }
                for r in detailed_results
            ],
            k=self.k,
        )

        faithfulness_vals = [r["faithfulness"] for r in detailed_results]
        relevancy_vals = [r["answer_relevancy"] for r in detailed_results]
        precision_vals = [r["context_precision"] for r in detailed_results]

        metrics = {
            f"hit_rate@{self.k}": retrieval_agg.get(f"hit_rate@{self.k}", 0.0),
            f"mrr@{self.k}": retrieval_agg.get(f"mrr@{self.k}", 0.0),
            f"recall@{self.k}": retrieval_agg.get(f"recall@{self.k}", 0.0),
            "faithfulness": round(float(np.mean(faithfulness_vals)), 4) if faithfulness_vals else 0.0,
            "answer_relevancy": round(float(np.mean(relevancy_vals)), 4) if relevancy_vals else 0.0,
            "context_precision": round(float(np.mean(precision_vals)), 4) if precision_vals else 0.0,
            "total_evaluated": len(detailed_results),
            "judge_cache_hits": len(self.judge.cache),
        }

        # Quality Gate Verification against configs/config.yaml thresholds
        thresholds = cfg.thresholds
        target_relevancy_threshold = getattr(thresholds, "answer_relevancy", getattr(thresholds, "answer_relevance", 0.80))
        passed_gates = {
            "faithfulness": metrics["faithfulness"] >= thresholds.faithfulness,
            "answer_relevancy": metrics["answer_relevancy"] >= target_relevancy_threshold,
            "context_precision": metrics["context_precision"] >= thresholds.context_precision,
            f"hit_rate@{self.k}": metrics[f"hit_rate@{self.k}"] >= thresholds.hit_rate_at_k,
            f"mrr@{self.k}": metrics[f"mrr@{self.k}"] >= thresholds.mrr_at_k,
        }
        all_passed = all(passed_gates.values())

        report = {
            "metrics": metrics,
            "thresholds": {
                "faithfulness": thresholds.faithfulness,
                "answer_relevancy": target_relevancy_threshold,
                "context_precision": thresholds.context_precision,
                "hit_rate_at_k": thresholds.hit_rate_at_k,
                "mrr_at_k": thresholds.mrr_at_k,
            },
            "quality_gates": passed_gates,
            "all_passed": all_passed,
            "detailed_results": detailed_results,
        }

        logger.info(
            f"Evaluation finished: Faithfulness={metrics['faithfulness']} | "
            f"Relevancy={metrics['answer_relevancy']} | Precision={metrics['context_precision']} | "
            f"HitRate@{self.k}={metrics[f'hit_rate@{self.k}']} | MRR@{self.k}={metrics[f'mrr@{self.k}']} | "
            f"Overall Gate Passed: {all_passed}"
        )
        return report
