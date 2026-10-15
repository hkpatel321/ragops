"""
Unit tests for configuration loading and validation using standard unittest.
"""

import unittest

from src.config import AppConfig, load_config


class TestConfig(unittest.TestCase):
    def test_load_config_defaults(self):
        config = load_config("configs/config.yaml")
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.project.name, "ragops")
        self.assertEqual(config.chunking.chunk_size, 512)
        self.assertEqual(config.chunking.overlap, 64)
        self.assertEqual(config.embedding.model_name, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.vectorstore.top_k, 4)
        self.assertEqual(config.llm.generator.model_name, "gemini-2.5-flash")
        self.assertEqual(config.llm.judge.model_name, "gemini-2.5-pro")
        self.assertGreaterEqual(config.thresholds.faithfulness, 0.80)
        self.assertGreaterEqual(config.thresholds.answer_relevance, 0.85)


if __name__ == "__main__":
    unittest.main()
