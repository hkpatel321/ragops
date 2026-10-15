"""
Unit tests for RAGOps evaluation module:
- Non-LLM retrieval metrics (hit_rate, mrr, recall)
- JudgeCache
- GeminiJudge
- RAGEvaluator and quality gates
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    reciprocal_rank_at_k,
    recall_at_k,
    compute_retrieval_metrics,
)
from src.evaluation.judge_llm import JudgeCache, GeminiJudge
from src.evaluation.evaluator import RAGEvaluator


class TestRetrievalMetrics(unittest.TestCase):
    def test_hit_rate_at_k(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        # Hit in top 2
        self.assertEqual(hit_rate_at_k(retrieved, "chunk_2", k=2), 1.0)
        # Miss in top 1
        self.assertEqual(hit_rate_at_k(retrieved, "chunk_2", k=1), 0.0)
        # Miss completely
        self.assertEqual(hit_rate_at_k(retrieved, "chunk_99", k=4), 0.0)

    def test_reciprocal_rank_at_k(self):
        retrieved = ["chunk_a", "chunk_b", "chunk_c"]
        # Rank 1 -> 1.0
        self.assertEqual(reciprocal_rank_at_k(retrieved, "chunk_a", k=3), 1.0)
        # Rank 2 -> 0.5
        self.assertEqual(reciprocal_rank_at_k(retrieved, "chunk_b", k=3), 0.5)
        # Rank 3 -> 0.3333
        self.assertAlmostEqual(reciprocal_rank_at_k(retrieved, "chunk_c", k=3), 1.0 / 3.0, places=3)
        # Outside k
        self.assertEqual(reciprocal_rank_at_k(retrieved, "chunk_c", k=2), 0.0)

    def test_recall_at_k(self):
        retrieved = ["c1", "c2", "c3"]
        gt = ["c1", "c3", "c5"]  # 2 out of 3 present in top 3
        self.assertAlmostEqual(recall_at_k(retrieved, gt, k=3), 2.0 / 3.0, places=3)
        # Top 1 has only c1
        self.assertAlmostEqual(recall_at_k(retrieved, gt, k=1), 1.0 / 3.0, places=3)

    def test_compute_retrieval_metrics(self):
        records = [
            {"retrieved_chunk_ids": ["c1", "c2"], "ground_truth_chunk_id": "c1"},  # hit=1, rr=1.0, rec=1.0
            {"retrieved_chunk_ids": ["c1", "c2"], "ground_truth_chunk_id": "c2"},  # hit=1, rr=0.5, rec=1.0
            {"retrieved_chunk_ids": ["c1", "c2"], "ground_truth_chunk_id": "c3"},  # hit=0, rr=0.0, rec=0.0
        ]
        metrics = compute_retrieval_metrics(records, k=2)
        self.assertAlmostEqual(metrics["hit_rate@2"], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(metrics["mrr@2"], (1.0 + 0.5 + 0.0) / 3.0, places=3)
        self.assertAlmostEqual(metrics["recall@2"], 2.0 / 3.0, places=3)
        self.assertEqual(metrics["total_queries"], 3.0)


class TestJudgeCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.test_dir) / "test_cache.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_set_and_get(self):
        cache = JudgeCache(cache_file=str(self.cache_path))
        key = cache.compute_key("model-x", "faithfulness", "q?", "a", ["ctx"])
        self.assertIsNone(cache.get(key))

        cache.set(key, {"score": 0.95, "reason": "test"})
        self.assertEqual(len(cache), 1)

        # Reload cache from disk
        cache2 = JudgeCache(cache_file=str(self.cache_path))
        entry = cache2.get(key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["score"], 0.95)


class TestGeminiJudgeAndEvaluator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.test_dir) / "eval_cache.json"
        self.judge = GeminiJudge(cache_path=str(self.cache_path))
        # Ensure offline deterministic mode for testing
        self.judge.langchain_llm = None

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_deterministic_scoring(self):
        score, reason = self.judge.evaluate_sample_with_cache(
            metric_name="faithfulness",
            question="What is FastAPI?",
            answer="FastAPI is a modern web framework using standard Python type annotations.",
            contexts=["FastAPI is a modern web framework using standard Python type annotations for high performance."],
        )
        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_rag_evaluator(self):
        evaluator = RAGEvaluator(judge=self.judge, k=2, batch_size=2, batch_delay=0.0)
        samples = [
            {
                "question": "How to create MLflow experiment?",
                "generated_answer": "Use mlflow.create_experiment() with a name.",
                "retrieved_chunk_ids": ["c1", "c2"],
                "retrieved_contexts": ["You can create an experiment using mlflow.create_experiment() function."],
                "ground_truth_chunk_id": "c1",
                "ground_truth_answer": "Use mlflow.create_experiment()",
            },
            {
                "question": "What is FastAPI?",
                "generated_answer": "FastAPI is a Python web framework.",
                "retrieved_chunk_ids": ["c3", "c4"],
                "retrieved_contexts": ["FastAPI is a fast web framework."],
                "ground_truth_chunk_id": "c5",
                "ground_truth_answer": "A fast web framework.",
            },
        ]
        report = evaluator.evaluate_dataset(samples)
        self.assertIn("metrics", report)
        self.assertIn("quality_gates", report)
        self.assertIn("thresholds", report)
        self.assertEqual(report["metrics"]["total_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
