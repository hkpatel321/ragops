"""
Alerting and Incident Response Module for RAGOps.
Supports:
1. Real-time Telegram Bot alerting for:
   - Rolling faithfulness dropping below threshold
   - Semantic embedding drift detected
   - Knowledge base index staleness
   - Index promotion and rollback events
2. Automated GitHub Actions workflow dispatch triggering index rebuilding and offline re-evaluation.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from src.config import cfg
from src.logger import logger


class TelegramAlerter:
    """
    Sends automated incident alerts and notifications to a Telegram chat via Bot API.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else (cfg.alerting.telegram.bot_token or os.getenv("TELEGRAM_BOT_TOKEN"))
        self.chat_id = chat_id if chat_id is not None else (cfg.alerting.telegram.chat_id or os.getenv("TELEGRAM_CHAT_ID"))
        self.parse_mode = parse_mode or cfg.alerting.telegram.parse_mode

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str) -> bool:
        """
        Send text message to configured Telegram chat.
        Returns True on success, False if unconfigured or on failure.
        """
        if not self.is_configured:
            logger.info("Telegram alerting skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.info("Telegram alert sent successfully.")
                    return True
                logger.warning(f"Telegram API responded with HTTP {resp.status}")
                return False
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            logger.warning(f"Telegram API HTTP error {e.code}: {err_msg}")
            return False
        except Exception as e:
            logger.warning(f"Failed to send Telegram alert: {e}")
            return False

    def alert_low_faithfulness(
        self,
        rolling_faithfulness: float,
        threshold: float,
        window_size: int = 20,
    ) -> bool:
        """
        Send incident alert when rolling faithfulness drops below threshold.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        msg = (
            f"🚨 *RAGOps Alert: Low Rolling Faithfulness*\n\n"
            f"• *Rolling Faithfulness*: `{rolling_faithfulness:.4f}`\n"
            f"• *Configured Threshold*: `{threshold:.4f}`\n"
            f"• *Evaluation Window*: `{window_size}` queries\n"
            f"• *Timestamp*: `{ts}`\n"
            f"• *Action*: Check `eval/failed_queries.jsonl` for sampled hallucinations and query failures."
        )
        return self.send_message(msg)

    def alert_drift_detected(
        self,
        embedding_type: str,
        drift_score: float,
        cosine_shift: float,
        report_path: Optional[str] = None,
    ) -> bool:
        """
        Send alert when semantic embedding drift is detected.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        msg = (
            f"⚠️ *RAGOps Alert: Embedding Drift Detected*\n\n"
            f"• *Space*: `{embedding_type.upper()}` embeddings\n"
            f"• *Drift Score*: `{drift_score:.4f}`\n"
            f"• *Centroid Cosine Shift*: `{cosine_shift:.4f}`\n"
            f"• *Report Path*: `{report_path or 'data/monitoring/'}`\n"
            f"• *Timestamp*: `{ts}`\n"
            f"• *Status*: Distribution shift exceeds configured tolerance."
        )
        return self.send_message(msg)

    def alert_index_stale(
        self,
        data_version: str,
        new_files: int,
        changed_files: int,
        removed_files: int,
        reindex_triggered: bool = False,
    ) -> bool:
        """
        Send alert when knowledge base files change and index becomes stale.
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        reindex_status = "⚡ *Automated Re-index Workflow Dispatched*" if reindex_triggered else "⚠️ *Re-index Required*"
        msg = (
            f"📢 *RAGOps Alert: Knowledge Base Stale*\n\n"
            f"• *Data Version*: `{data_version}`\n"
            f"• *Modified Files*: `{changed_files}`\n"
            f"• *New Files Added*: `{new_files}`\n"
            f"• *Files Removed*: `{removed_files}`\n"
            f"• *Timestamp*: `{ts}`\n"
            f"• *Action*: {reindex_status}"
        )
        return self.send_message(msg)

    def alert_deployment_status(
        self,
        data_version: str,
        status: str,
        gate_passed: bool,
        details: str = "",
    ) -> bool:
        """
        Notify on index promotion or rollback.
        """
        emoji = "✅" if gate_passed else "❌"
        action = "PROMOTED TO PRODUCTION" if gate_passed else "ROLLED BACK TO BASELINE"
        msg = (
            f"{emoji} *RAGOps Index Deployment: {action}*\n\n"
            f"• *Version*: `{data_version}`\n"
            f"• *Status*: `{status}`\n"
            f"• *Quality Gate*: `{'PASSED' if gate_passed else 'FAILED'}`\n"
            f"• *Details*: {details}"
        )
        return self.send_message(msg)


def trigger_github_reindex(
    data_version: str = "v2",
    reason: str = "alert_triggered",
    workflow_id: Optional[str] = None,
    ref: Optional[str] = None,
) -> bool:
    """
    Dispatch GitHub Actions workflow via REST API to rebuild index and run quality gate.
    Requires GITHUB_TOKEN and GITHUB_REPOSITORY in environment.
    """
    token = cfg.alerting.github_workflow.github_token or os.getenv("GITHUB_TOKEN")
    repo = cfg.alerting.github_workflow.repository or os.getenv("GITHUB_REPOSITORY")
    workflow = workflow_id or cfg.alerting.github_workflow.workflow_id
    target_ref = ref or cfg.alerting.github_workflow.ref

    if not token or not repo:
        logger.info(
            f"GitHub workflow dispatch skipped: GITHUB_TOKEN or GITHUB_REPOSITORY not set "
            f"(simulated trigger for {workflow} on {data_version})."
        )
        return False

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    payload = {
        "ref": target_ref,
        "inputs": {
            "data_version": data_version,
            "trigger_reason": reason,
        },
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
                "User-Agent": "RAGOps-Alert-Manager",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 204:
                logger.info(f"Successfully dispatched GitHub workflow '{workflow}' for repo '{repo}'.")
                return True
            logger.warning(f"GitHub API returned unexpected status {resp.status}")
            return False
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        logger.warning(f"GitHub API HTTP error {e.code}: {err_msg}")
        return False
    except Exception as e:
        logger.warning(f"Failed to trigger GitHub workflow: {e}")
        return False


class AlertManager:
    """
    Centralized monitoring event listener that evaluates alert conditions and triggers
    both Telegram notifications and GitHub Actions reindexing workflows.
    """

    def __init__(self, alerter: Optional[TelegramAlerter] = None) -> None:
        self.alerter = alerter or TelegramAlerter()

    def process_kb_drift(self, kb_report: Any, auto_reindex: bool = True) -> Dict[str, Any]:
        """
        Evaluate KB drift report; alert and trigger workflow if stale.
        """
        if not getattr(kb_report, "is_stale", False):
            return {"alert_sent": False, "reindex_dispatched": False}

        reindex_dispatched = False
        if auto_reindex:
            reindex_dispatched = trigger_github_reindex(
                data_version=getattr(kb_report, "data_version", "v2"),
                reason="kb_drift_staleness",
            )

        alert_sent = self.alerter.alert_index_stale(
            data_version=getattr(kb_report, "data_version", "v2"),
            new_files=len(getattr(kb_report, "new_files", [])),
            changed_files=len(getattr(kb_report, "changed_files", [])),
            removed_files=len(getattr(kb_report, "removed_files", [])),
            reindex_triggered=reindex_dispatched,
        )

        return {"alert_sent": alert_sent, "reindex_dispatched": reindex_dispatched}

    def process_embedding_drift(self, drift_result: Any, auto_reindex: bool = False) -> Dict[str, Any]:
        """
        Evaluate embedding drift result; alert if drift detected.
        """
        if not getattr(drift_result, "drift_detected", False):
            return {"alert_sent": False, "reindex_dispatched": False}

        reindex_dispatched = False
        if auto_reindex:
            reindex_dispatched = trigger_github_reindex(
                data_version="v2",
                reason=f"embedding_drift_{getattr(drift_result, 'embedding_type', 'query')}",
            )

        alert_sent = self.alerter.alert_drift_detected(
            embedding_type=getattr(drift_result, "embedding_type", "query"),
            drift_score=getattr(drift_result, "drift_score", 0.0),
            cosine_shift=getattr(drift_result, "mean_cosine_shift", 0.0),
            report_path=getattr(drift_result, "report_html_path", None),
        )

        return {"alert_sent": alert_sent, "reindex_dispatched": reindex_dispatched}

    def process_faithfulness_drop(
        self,
        rolling_faithfulness: float,
        threshold: Optional[float] = None,
        window_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Check if rolling faithfulness dropped below threshold and alert.
        """
        target_thresh = threshold if threshold is not None else cfg.thresholds.faithfulness
        if rolling_faithfulness >= target_thresh:
            return {"alert_sent": False, "breached": False}

        alert_sent = self.alerter.alert_low_faithfulness(
            rolling_faithfulness=rolling_faithfulness,
            threshold=target_thresh,
            window_size=window_size,
        )
        return {"alert_sent": alert_sent, "breached": True}
