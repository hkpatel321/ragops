"""
Unit tests for RAGOps Monitoring:
- Knowledge base drift and staleness detection
- Production query sampling evaluator and rolling metrics
- Embedding drift detector (statistical shift and report generation)
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.monitoring.embedding_drift import EmbeddingDriftDetector
from src.monitoring.kb_drift import KBDriftDetector
from src.monitoring.sampling_evaluator import SamplingEvaluator


class TestKBDrift(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.v1_dir = self.root / "v1"
        self.v2_dir = self.root / "v2"
        self.manifests_dir = self.root / "manifests"

        self.v1_dir.mkdir()
        self.v2_dir.mkdir()
        self.manifests_dir.mkdir()

        # Create baseline files in v1
        (self.v1_dir / "doc1.md").write_text("# Doc 1\nIntroduction to service.", encoding="utf-8")
        (self.v1_dir / "doc2.md").write_text("# Doc 2\nOriginal API endpoints.", encoding="utf-8")

        # Create v2 with 1 unchanged, 1 modified, and 1 added file
        (self.v2_dir / "doc1.md").write_text("# Doc 1\nIntroduction to service.", encoding="utf-8")
        (self.v2_dir / "doc2.md").write_text("# Doc 2\nUpdated API with breaking changes.", encoding="utf-8")
        (self.v2_dir / "doc3.md").write_text("# Doc 3\nBrand new v2 feature guide.", encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_kb_drift_baseline_creation(self):
        detector = KBDriftDetector(manifests_dir=str(self.manifests_dir))
        report = detector.detect_drift(self.v1_dir, data_version="v1")
        self.assertFalse(report.is_stale)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.total_current_files, 2)
        self.assertEqual(report.total_reference_files, 2)
        self.assertEqual(len(report.new_files), 0)
        self.assertEqual(len(report.changed_files), 0)

    def test_kb_drift_detection_when_files_change(self):
        detector = KBDriftDetector(manifests_dir=str(self.manifests_dir))
        v1_manifest = detector.build_manifest(self.v1_dir)

        report = detector.detect_drift(
            current_dir=self.v2_dir,
            data_version="v2",
            reference_manifest=v1_manifest,
        )

        self.assertTrue(report.is_stale)
        self.assertTrue(report.drift_detected)
        self.assertEqual(len(report.new_files), 1)
        self.assertIn("doc3.md", report.new_files)
        self.assertEqual(len(report.changed_files), 1)
        self.assertIn("doc2.md", report.changed_files)
        self.assertEqual(len(report.unchanged_files), 1)
        self.assertIn("doc1.md", report.unchanged_files)


class TestSamplingEvaluator(unittest.TestCase):
    def test_sampling_evaluator_scoring_and_rolling_averages(self):
        mock_store = MagicMock()
        mock_judge = MagicMock()

        mock_queries = [
            {
                "id": f"q_{i}",
                "query": f"Sample question {i}",
                "answer": f"Sample answer {i}",
                "contexts": [f"Sample context {i}"],
            }
            for i in range(10)
        ]
        mock_store.get_recent_queries.return_value = mock_queries
        mock_judge.evaluate_sample_with_cache.side_effect = [
            (0.85, "Good ground truth"),  # faithfulness
            (0.90, "Highly relevant"),   # answer relevancy
        ]

        evaluator = SamplingEvaluator(
            store=mock_store,
            judge=mock_judge,
            sample_rate=0.10,
            window_size=5,
        )

        res = evaluator.process_new_queries()

        self.assertEqual(res["new_queries_found"], 10)
        self.assertGreaterEqual(res["sampled_count"], 1)
        self.assertGreater(res["rolling_faithfulness"], 0.0)
        self.assertGreater(res["rolling_relevancy"], 0.0)
        self.assertEqual(len(evaluator.evaluated_ids), 10)


class TestEmbeddingDrift(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.reports_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_embedding_drift_detector(self):
        detector = EmbeddingDriftDetector(reports_dir=str(self.reports_dir), drift_threshold=0.10)

        np.random.seed(42)
        ref_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(20, 64))
        curr_embeddings = np.random.normal(loc=2.0, scale=1.0, size=(20, 64))

        result = detector.detect_drift(
            reference_embeddings=ref_embeddings,
            current_embeddings=curr_embeddings,
            embedding_type="test_query",
        )

        self.assertEqual(result.embedding_type, "test_query")
        self.assertEqual(result.reference_count, 20)
        self.assertEqual(result.current_count, 20)
        self.assertGreaterEqual(result.drift_score, 0.0)
        self.assertGreaterEqual(result.mean_cosine_shift, 0.0)
        self.assertTrue(Path(result.metrics_json_path).exists())


if __name__ == "__main__":
    unittest.main()
