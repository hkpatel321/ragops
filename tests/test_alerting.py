"""
Unit tests for RAGOps Telegram Alerting, Automated Incident Response,
Failed Queries Logging, and Production Promotion/Rollback Controller.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.promote_or_rollback import promote_index, rollback_index
from src.monitoring.alerting import (
    AlertManager,
    TelegramAlerter,
    trigger_github_reindex,
)
from src.monitoring.kb_drift import KBDriftReport
from src.monitoring.sampling_evaluator import SamplingEvaluator


class TestTelegramAlerter(unittest.TestCase):
    def test_unconfigured_alerter(self):
        alerter = TelegramAlerter(bot_token="", chat_id="")
        self.assertFalse(alerter.is_configured)
        # Should gracefully return False without exceptions
        self.assertFalse(alerter.send_message("Test alert"))

    @patch("urllib.request.urlopen")
    def test_configured_alerter_success(self, mock_urlopen):
        # Mock successful 200 response
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        alerter = TelegramAlerter(bot_token="test_token_123", chat_id="test_chat_456")
        self.assertTrue(alerter.is_configured)

        res = alerter.send_message("🚨 Test Message")
        self.assertTrue(res)
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertIn("api.telegram.org/bottest_token_123/sendMessage", req.full_url)
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["chat_id"], "test_chat_456")
        self.assertEqual(body["text"], "🚨 Test Message")

    @patch("urllib.request.urlopen")
    def test_alert_helper_methods(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        alerter = TelegramAlerter(bot_token="token", chat_id="chat")

        # 1. Low faithfulness
        self.assertTrue(alerter.alert_low_faithfulness(rolling_faithfulness=0.62, threshold=0.75))
        body1 = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertIn("Low Rolling Faithfulness", body1["text"])
        self.assertIn("0.6200", body1["text"])

        # 2. Embedding drift
        self.assertTrue(alerter.alert_drift_detected("query", drift_score=0.25, cosine_shift=0.35))
        body2 = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertIn("Embedding Drift Detected", body2["text"])

        # 3. Index stale
        self.assertTrue(alerter.alert_index_stale("v2", new_files=3, changed_files=2, removed_files=0))
        body3 = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertIn("Knowledge Base Stale", body3["text"])

        # 4. Deployment status
        self.assertTrue(alerter.alert_deployment_status("v2", status="PROMOTED", gate_passed=True))
        body4 = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertIn("PROMOTED TO PRODUCTION", body4["text"])


class TestGitHubWorkflowTrigger(unittest.TestCase):
    def test_trigger_without_token_returns_false(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.monitoring.alerting.cfg.alerting.github_workflow.github_token", ""):
                res = trigger_github_reindex(data_version="v2")
                self.assertFalse(res)

    @patch("urllib.request.urlopen")
    def test_trigger_with_credentials_sends_dispatch(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with patch("src.monitoring.alerting.cfg.alerting.github_workflow.github_token", "gh_secret_token"):
            with patch("src.monitoring.alerting.cfg.alerting.github_workflow.repository", "user/repo"):
                res = trigger_github_reindex(data_version="v2", reason="drift_detected")
                self.assertTrue(res)
                mock_urlopen.assert_called_once()
                req = mock_urlopen.call_args[0][0]
                self.assertIn("actions/workflows/reindex_and_eval.yml/dispatches", req.full_url)
                body = json.loads(req.data.decode("utf-8"))
                self.assertEqual(body["inputs"]["data_version"], "v2")
                self.assertEqual(body["inputs"]["trigger_reason"], "drift_detected")


class TestAlertManager(unittest.TestCase):
    def setUp(self):
        self.mock_alerter = MagicMock()
        self.manager = AlertManager(alerter=self.mock_alerter)

    @patch("src.monitoring.alerting.trigger_github_reindex")
    def test_process_kb_drift_when_stale(self, mock_reindex):
        mock_reindex.return_value = True
        report = KBDriftReport(
            data_version="v2",
            is_stale=True,
            drift_detected=True,
            total_reference_files=4,
            total_current_files=5,
            new_files=["doc_new.md"],
            changed_files=["doc_mod.md"],
            removed_files=[],
        )

        res = self.manager.process_kb_drift(report, auto_reindex=True)
        self.assertTrue(res["reindex_dispatched"])
        mock_reindex.assert_called_once_with(data_version="v2", reason="kb_drift_staleness")
        self.mock_alerter.alert_index_stale.assert_called_once()

    def test_process_kb_drift_when_fresh(self):
        report = KBDriftReport(
            data_version="v1",
            is_stale=False,
            drift_detected=False,
            total_reference_files=5,
            total_current_files=5,
            new_files=[],
            changed_files=[],
            removed_files=[],
        )

        res = self.manager.process_kb_drift(report, auto_reindex=True)
        self.assertFalse(res["alert_sent"])
        self.assertFalse(res["reindex_dispatched"])
        self.mock_alerter.alert_index_stale.assert_not_called()

    def test_process_faithfulness_drop(self):
        # Above threshold: no breach
        res_ok = self.manager.process_faithfulness_drop(rolling_faithfulness=0.85, threshold=0.70)
        self.assertFalse(res_ok["breached"])
        self.mock_alerter.alert_low_faithfulness.assert_not_called()

        # Below threshold: triggers breach alert
        self.mock_alerter.alert_low_faithfulness.return_value = True
        res_fail = self.manager.process_faithfulness_drop(rolling_faithfulness=0.55, threshold=0.70)
        self.assertTrue(res_fail["breached"])
        self.mock_alerter.alert_low_faithfulness.assert_called_once_with(
            rolling_faithfulness=0.55,
            threshold=0.70,
            window_size=20,
        )


class TestFailedQueriesLogging(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.failed_queries_path = self.root / "failed_queries.jsonl"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_log_failed_query_direct(self):
        evaluator = SamplingEvaluator(
            sample_rate=1.0,
            faithfulness_threshold=0.70,
            relevancy_threshold=0.70,
            failed_queries_path=str(self.failed_queries_path),
        )

        evaluator.log_failed_query(
            query_id="manual_test_q_1",
            query="How do I log parameters in MLflow?",
            answer="MLflow does not support parameters.",
            contexts=["MLflow supports mlflow.log_param()"],
            faithfulness=0.2,
            faithfulness_reason="Contradicts provided context.",
            relevancy=0.8,
            relevancy_reason="Answers the query topic.",
            index_version="v1",
        )

        self.assertTrue(self.failed_queries_path.is_file())
        lines = self.failed_queries_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

        saved = json.loads(lines[0])
        self.assertEqual(saved["id"], "manual_test_q_1")
        self.assertEqual(saved["faithfulness"], 0.2)
        self.assertEqual(saved["failure_reason"], "low_faithfulness")
        self.assertEqual(saved["index_version"], "v1")

    def test_process_new_queries_appends_failed(self):
        evaluator = SamplingEvaluator(
            sample_rate=1.0,
            faithfulness_threshold=0.70,
            relevancy_threshold=0.70,
            failed_queries_path=str(self.failed_queries_path),
        )

        # Mock store with 1 logged query
        mock_store = MagicMock()
        mock_store.get_recent_queries.return_value = [
            {
                "id": "q_low_score_100",
                "query": "How to deploy FastAPI?",
                "answer": "Use Flask.",
                "contexts": ["FastAPI is deployed using Uvicorn or Gunicorn."],
                "index_version": "v1",
            }
        ]
        evaluator.store = mock_store

        # Mock judge to score low faithfulness
        mock_judge = MagicMock()
        mock_judge.evaluate_sample_with_cache.side_effect = [
            (0.35, "Answer contradicts context recommendation."),  # faithfulness
            (0.80, "Relevant query addressed."),                   # relevancy
        ]
        evaluator.judge = mock_judge

        summary = evaluator.process_new_queries()
        self.assertEqual(summary["sampled_count"], 1)
        self.assertEqual(summary["failed_queries_logged"], 1)

        # Verify failed query file contains the record
        self.assertTrue(self.failed_queries_path.is_file())
        lines = self.failed_queries_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        item = json.loads(lines[0])
        self.assertEqual(item["id"], "q_low_score_100")
        self.assertEqual(item["faithfulness"], 0.35)
        self.assertEqual(item["failure_reason"], "low_faithfulness")


class TestPromotionAndRollback(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.active_path = self.root / "active_index_version.json"

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("scripts.promote_or_rollback.TelegramAlerter")
    @patch("scripts.promote_or_rollback.MlflowClient")
    def test_promote_index(self, mock_client_cls, mock_alerter_cls):
        mock_alerter = MagicMock()
        mock_alerter_cls.return_value = mock_alerter

        eval_report = {
            "version_id": "v2_hash123",
            "overall_gate_passed": True,
            "metrics": {"faithfulness": 0.88, "mrr_at_k": 0.85},
        }

        summary = promote_index(
            data_version="v2",
            eval_report=eval_report,
            active_version_path=self.active_path,
            experiment_name="RAGOps_Eval",
            tracking_uri="file:///tmp/mlruns",
        )

        self.assertEqual(summary["status"], "PROMOTED")
        self.assertEqual(summary["stage"], "Production")
        self.assertEqual(summary["active_data_version"], "v2")

        # Verify active index pointer file written
        self.assertTrue(self.active_path.is_file())
        saved_pointer = json.loads(self.active_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_pointer["status"], "PROMOTED")
        self.assertEqual(saved_pointer["active_data_version"], "v2")

        # Verify Telegram alert called
        mock_alerter.alert_deployment_status.assert_called_once()

    @patch("scripts.promote_or_rollback.TelegramAlerter")
    @patch("scripts.promote_or_rollback.MlflowClient")
    def test_rollback_index(self, mock_client_cls, mock_alerter_cls):
        mock_alerter = MagicMock()
        mock_alerter_cls.return_value = mock_alerter

        eval_report = {
            "version_id": "v2_badhash",
            "overall_gate_passed": False,
            "metrics": {"faithfulness": 0.50, "mrr_at_k": 0.40},
        }

        summary = rollback_index(
            data_version="v2",
            baseline_version="v1",
            eval_report=eval_report,
            active_version_path=self.active_path,
            experiment_name="RAGOps_Eval",
            tracking_uri="file:///tmp/mlruns",
        )

        self.assertEqual(summary["status"], "ROLLED_BACK")
        self.assertEqual(summary["stage"], "Archived")
        self.assertEqual(summary["active_data_version"], "v1")

        # Verify active pointer retains baseline
        self.assertTrue(self.active_path.is_file())
        saved_pointer = json.loads(self.active_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_pointer["status"], "ROLLED_BACK")
        self.assertEqual(saved_pointer["active_data_version"], "v1")

        # Verify Telegram alert called with failure status
        mock_alerter.alert_deployment_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
