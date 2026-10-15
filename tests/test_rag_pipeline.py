"""
Unit tests for the RAG pipeline, prompt construction, and response structure.
"""

import unittest
from unittest.mock import MagicMock

from src.rag.pipeline import GROUNDED_PROMPT_TEMPLATE, RAGPipeline, RAGResponse


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()
        self.sample_chunks = [
            {
                "chunk_id": "v1_fastapi_001",
                "text": "FastAPI allows path parameters to be declared with Python format strings.",
                "metadata": {
                    "source": "fastapi",
                    "version": "v1",
                    "section": "Path Parameters",
                    "title": "FastAPI Tutorial",
                },
                "score": 0.88,
            },
            {
                "chunk_id": "v1_mlflow_001",
                "text": "MLflow tracking records metrics across runs.",
                "metadata": {
                    "source": "mlflow",
                    "version": "v1",
                    "section": "Tracking API",
                    "title": "MLflow Tracking",
                },
                "score": 0.75,
            },
        ]

    def test_grounded_prompt_instructions(self):
        # Must instruct model to say "I don't know" when context is insufficient
        self.assertIn("I don't know based on the provided documentation.", GROUNDED_PROMPT_TEMPLATE)
        self.assertIn("Answer the user's question strictly and solely based on the provided context below.", GROUNDED_PROMPT_TEMPLATE)

    def test_format_context_provenance(self):
        formatted = self.pipeline.format_context(self.sample_chunks)
        self.assertIn("[Document 1]", formatted)
        self.assertIn("Source: fastapi (v1)", formatted)
        self.assertIn("Section: Path Parameters", formatted)
        self.assertIn("[Document 2]", formatted)
        self.assertIn("Source: mlflow (v1)", formatted)

    def test_query_mock_execution(self):
        query = "How do path parameters work in FastAPI?"
        res = self.pipeline.query(
            query=query,
            data_version="v1",
            mock_answer="Path parameters are declared using Python format string syntax.",
        )
        self.assertIsInstance(res, RAGResponse)
        self.assertEqual(res.query, query)
        self.assertIn("Path parameters", res.answer)
        self.assertGreater(len(res.contexts), 0)
        self.assertGreater(res.latency, 0.0)
        self.assertIn("prompt_tokens", res.token_counts)
        self.assertIn("completion_tokens", res.token_counts)
        self.assertIn("total_tokens", res.token_counts)
        self.assertTrue(res.version_id.startswith("v1_"))

    def test_retry_on_rate_limit(self):
        # Test retry handler with mock client throwing 429 then succeeding
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response after rate limit"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=10,
            total_token_count=60,
        )

        # Fail twice with 429 then succeed
        mock_client.models.generate_content.side_effect = [
            Exception("429 Resource Exhausted: Rate limit exceeded"),
            Exception("429 Too Many Requests"),
            mock_response,
        ]

        test_pipeline = RAGPipeline(initial_backoff=0.01)
        test_pipeline.client = mock_client

        answer, tokens = test_pipeline._call_gemini_with_retry("test prompt")
        self.assertEqual(answer, "Grounded response after rate limit")
        self.assertEqual(tokens["total_tokens"], 60)
        self.assertEqual(mock_client.models.generate_content.call_count, 3)


if __name__ == "__main__":
    unittest.main()
