"""
Unit tests for the golden evaluation dataset schema and integrity.
"""

import json
import unittest
from pathlib import Path


class TestGoldenSet(unittest.TestCase):
    def setUp(self):
        self.golden_path = Path("eval/golden_v1.jsonl")

    def test_golden_file_exists_and_count(self):
        self.assertTrue(self.golden_path.is_file(), "eval/golden_v1.jsonl does not exist")
        with open(self.golden_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        count = len(lines)
        self.assertGreaterEqual(count, 60, f"Expected at least 60 items, got {count}")
        self.assertLessEqual(count, 100, f"Expected at most 100 items, got {count}")

    def test_golden_records_schema(self):
        required_fields = {
            "id",
            "question",
            "ground_truth_answer",
            "ground_truth_chunk_id",
            "ground_truth_context",
            "question_type",
            "source",
            "version",
        }

        types_found = set()
        sources_found = set()

        with open(self.golden_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for field in required_fields:
                    self.assertIn(field, record, f"Missing field '{field}' in record {record.get('id')}")

                self.assertTrue(len(record["question"]) > 10)
                self.assertTrue(len(record["ground_truth_answer"]) > 10)
                self.assertTrue(len(record["ground_truth_chunk_id"]) > 0)
                self.assertEqual(record["version"], "v1")

                types_found.add(record["question_type"])
                sources_found.add(record["source"])

        # Check coverage of question types
        self.assertIn("factual", types_found)
        self.assertIn("how_to", types_found)
        self.assertIn("version_specific", types_found)

        # Check coverage of sources
        self.assertIn("fastapi", sources_found)
        self.assertIn("mlflow", sources_found)


if __name__ == "__main__":
    unittest.main()
