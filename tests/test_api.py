"""
Unit & Integration Tests for RAGOps FastAPI application:
- GET /
- GET /health
- POST /query (mock & grounded retrieval)
- GET /metrics (Prometheus exposition format)
- GET /history
- SQLite & JSONL telemetry verification
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app, set_store
from src.api.storage import QueryStore


class TestRAGOpsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Use temporary storage directory for isolated testing
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_queries.db"
        self.jsonl_path = Path(self.test_dir) / "test_queries.jsonl"
        self.test_store = QueryStore(
            db_path=str(self.db_path),
            jsonl_path=str(self.jsonl_path),
        )
        set_store(self.test_store)

    def tearDown(self):
        set_store(None)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "ragops-api")
        self.assertIn("docs", data)
        self.assertIn("health", data)
        self.assertIn("metrics", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("vectorstore", data)
        self.assertIn("storage", data)

    def test_query_endpoint_mock(self):
        payload = {
            "query": "How do path parameters work in FastAPI?",
            "data_version": "v1",
            "top_k": 2,
            "mock": True,
        }
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("contexts", data)
        self.assertIn("retrieved_chunks", data)
        self.assertIn("latency", data)
        self.assertIn("token_counts", data)
        self.assertIn("request_id", data)
        self.assertTrue(data["request_id"].startswith("req_"))

        # Verify record was stored in SQLite
        recent = self.test_store.get_recent_queries(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["id"], data["request_id"])
        self.assertEqual(recent[0]["query"], payload["query"])

        # Verify JSONL file exists and contains record
        self.assertTrue(self.jsonl_path.exists())
        lines = self.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

    def test_metrics_endpoint(self):
        # Trigger a query to populate metrics
        self.client.post("/query", json={"query": "Metrics test query", "mock": True})

        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        text = response.text
        # Assert Prometheus metrics are properly exposed
        self.assertIn("ragops_requests_total", text)
        self.assertIn("ragops_request_latency_seconds", text)
        self.assertIn("ragops_tokens_total", text)
        self.assertIn("ragops_retrieval_similarity_score", text)

    def test_history_endpoint(self):
        # Insert a sample query
        self.client.post("/query", json={"query": "History test query", "mock": True})

        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        history = response.json()
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["query"], "History test query")


if __name__ == "__main__":
    unittest.main()
