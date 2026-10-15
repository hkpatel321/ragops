"""
Production Index Promotion & Rollback Controller for RAGOps.
Inspects offline evaluation results:
- If quality gate passed: Promotes the new index version to 'Production' in MLflow and updates active pointer.
- If quality gate failed: Reverts/rolls back to the stable baseline version in MLflow and logs the rollback.
Sends deployment status notifications to Telegram.

Usage:
    python scripts/promote_or_rollback.py --eval-report eval/eval_report_latest.json --data-version v2 --baseline-version v1
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.config import cfg
from src.logger import logger
from src.monitoring.alerting import TelegramAlerter


def load_eval_report(report_path: Path) -> Dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Evaluation report not found at: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def promote_index(
    data_version: str,
    eval_report: Dict[str, Any],
    active_version_path: Path,
    experiment_name: str,
    tracking_uri: str,
) -> Dict[str, Any]:
    """
    Promote newly built index to Production in MLflow and update local active index pointer.
    """
    logger.info(f"[PROMOTION] Quality Gate PASSED! Promoting index [{data_version}] to PRODUCTION...")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    version_id = eval_report.get("version_id", f"{data_version}_promoted")

    summary = {
        "status": "PROMOTED",
        "stage": "Production",
        "active_data_version": data_version,
        "version_id": version_id,
        "promoted_at": timestamp,
        "metrics": eval_report.get("metrics", {}),
    }

    # 1. Update MLflow run tags
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=3,
            )
            if runs:
                latest_run_id = runs[0].info.run_id
                client.set_tag(latest_run_id, "deployment_status", "PROMOTED")
                client.set_tag(latest_run_id, "stage", "Production")
                client.set_tag(latest_run_id, "promoted_at", timestamp)
                logger.info(f"Updated MLflow run [{latest_run_id}] with stage='Production'.")
    except Exception as e:
        logger.warning(f"Notice updating MLflow tags during promotion: {e}")

    # 2. Update active version pointer file
    active_version_path.parent.mkdir(parents=True, exist_ok=True)
    with open(active_version_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Updated active index pointer at '{active_version_path}': {summary}")

    # 3. Notify Telegram
    alerter = TelegramAlerter()
    alerter.alert_deployment_status(
        data_version=data_version,
        status="PROMOTED",
        gate_passed=True,
        details=f"Faithfulness={eval_report.get('metrics', {}).get('faithfulness', 'N/A')}, MRR={eval_report.get('metrics', {}).get('mrr_at_k', 'N/A')}",
    )

    return summary


def rollback_index(
    data_version: str,
    baseline_version: str,
    eval_report: Dict[str, Any],
    active_version_path: Path,
    experiment_name: str,
    tracking_uri: str,
) -> Dict[str, Any]:
    """
    Roll back index to baseline version in MLflow and maintain baseline active pointer.
    """
    logger.warning(
        f"[ROLLBACK] Quality Gate FAILED for index [{data_version}]! "
        f"Rolling back deployment to baseline [{baseline_version}]..."
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    summary = {
        "status": "ROLLED_BACK",
        "stage": "Archived",
        "failed_data_version": data_version,
        "active_data_version": baseline_version,
        "rolled_back_at": timestamp,
        "rollback_reason": "Failed quality gate thresholds or regression check",
        "metrics": eval_report.get("metrics", {}),
    }

    # 1. Update MLflow run tags
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=3,
            )
            if runs:
                latest_run_id = runs[0].info.run_id
                client.set_tag(latest_run_id, "deployment_status", "ROLLED_BACK")
                client.set_tag(latest_run_id, "stage", "Archived")
                client.set_tag(latest_run_id, "rollback_target", baseline_version)
                logger.info(f"Updated MLflow run [{latest_run_id}] with stage='Archived' (ROLLED_BACK).")
    except Exception as e:
        logger.warning(f"Notice updating MLflow tags during rollback: {e}")

    # 2. Update active version pointer file to retain baseline
    active_version_path.parent.mkdir(parents=True, exist_ok=True)
    with open(active_version_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Active index remains locked to baseline '{baseline_version}'.")

    # 3. Notify Telegram
    alerter = TelegramAlerter()
    alerter.alert_deployment_status(
        data_version=data_version,
        status="ROLLED_BACK",
        gate_passed=False,
        details=f"Gate failed. Active index preserved at '{baseline_version}'.",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote or roll back RAG index based on quality gate evaluation.")
    parser.add_argument("--eval-report", type=str, default="eval/eval_report_latest.json", help="Path to eval report JSON")
    parser.add_argument("--data-version", type=str, default="v2", help="Candidate data version evaluated")
    parser.add_argument("--baseline-version", type=str, default="v1", help="Stable baseline data version")
    parser.add_argument("--active-version-path", type=str, default="data/active_index_version.json", help="Active index pointer file")
    parser.add_argument("--experiment-name", type=str, default=cfg.mlflow.experiment_name, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=cfg.mlflow.tracking_uri, help="MLflow tracking URI")
    parser.add_argument("--force-promote", action="store_true", help="Force promotion regardless of gate status")

    args = parser.parse_args()

    report_path = Path(args.eval_report)
    active_path = Path(args.active_version_path)

    report_data = load_eval_report(report_path)
    gate_passed = report_data.get("overall_gate_passed", False) or args.force_promote

    print("\n" + "=" * 55)
    print("      RAGOps DEPLOYMENT & MLflow PROMOTION CONTROLLER     ")
    print("=" * 55)
    print(f"Candidate Version    : {args.data_version}")
    print(f"Baseline Version     : {args.baseline_version}")
    print(f"Quality Gate Result  : {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 55 + "\n")

    if gate_passed:
        promote_index(
            data_version=args.data_version,
            eval_report=report_data,
            active_version_path=active_path,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
        )
        print(">>> RESULT: Promotion successful. Version is active in Production. <<<\n")
        sys.exit(0)
    else:
        rollback_index(
            data_version=args.data_version,
            baseline_version=args.baseline_version,
            eval_report=report_data,
            active_version_path=active_path,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
        )
        print(">>> RESULT: Deployment rejected. Rolled back to baseline. <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
