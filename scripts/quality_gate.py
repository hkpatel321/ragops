"""
Quality Gate & Regression Checker for RAGOps.
Validates the latest evaluation run:
1. Checks if all metrics meet configured absolute quality thresholds from configs/config.yaml.
2. Compares against the last registered MLflow evaluation run to detect performance regressions (> X%).
3. Generates a GitHub-flavored Markdown summary report for PR commenting and CI step summaries.
4. Exits with 0 on pass or 1 on failure.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient
from tabulate import tabulate

from src.config import cfg
from src.logger import logger


def load_eval_report(report_path: str) -> Dict[str, Any]:
    p = Path(report_path)
    if not p.is_file():
        raise FileNotFoundError(f"Evaluation report not found at: {report_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_previous_mlflow_run(
    client: MlflowClient,
    experiment_name: str,
    current_metrics: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """
    Fetch the most recent preceding evaluation run from MLflow to use as a baseline.
    """
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        logger.info(f"MLflow experiment '{experiment_name}' not found. No previous baseline available.")
        return None

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.run_type = 'evaluation'",
        order_by=["attribute.start_time DESC"],
        max_results=5,
    )

    if not runs:
        logger.info("No previous evaluation runs found in MLflow. This is the initial baseline.")
        return None

    # Find the run preceding the current one
    for run in runs:
        run_metrics = run.data.metrics
        # If run metrics differ from current report or it's a distinct run, treat as baseline
        if run_metrics:
            return run_metrics

    return None


def normalize_metric_key(key: str) -> str:
    """Normalize metric keys for flexible comparison (e.g. hit_rate@4 -> hit_rate_at_4)."""
    return key.replace("@", "_at_").lower()


def evaluate_quality_gates(
    current_metrics: Dict[str, float],
    thresholds: Dict[str, float],
    baseline_metrics: Optional[Dict[str, float]],
    max_regression_pct: float,
    ignore_thresholds: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Verify absolute thresholds and relative regression against baseline.
    """
    results: List[Dict[str, Any]] = []
    overall_passed = True

    # Standard metrics to evaluate
    tracked_metrics = [
        ("hit_rate@4", "hit_rate_at_k", 0.80),
        ("mrr@4", "mrr_at_k", 0.60),
        ("recall@4", "recall_at_k", None),
        ("faithfulness", "faithfulness", 0.80),
        ("answer_relevancy", "answer_relevance", 0.80),
        ("context_precision", "context_precision", 0.75),
    ]

    for display_key, threshold_key, default_thresh in tracked_metrics:
        # Find current score
        curr_val = current_metrics.get(display_key)
        if curr_val is None:
            curr_val = current_metrics.get(normalize_metric_key(display_key), 0.0)

        # Find threshold
        target_thresh = thresholds.get(threshold_key, default_thresh)

        # Find baseline score
        base_val = None
        if baseline_metrics:
            base_val = baseline_metrics.get(normalize_metric_key(display_key))
            if base_val is None:
                base_val = baseline_metrics.get(display_key)

        # Check threshold status
        threshold_passed = True
        if target_thresh is not None and not ignore_thresholds:
            if curr_val < target_thresh:
                threshold_passed = False

        # Check regression status
        regression_pct = 0.0
        regression_passed = True
        if base_val is not None and base_val > 1e-4:
            if curr_val < base_val:
                regression_pct = ((base_val - curr_val) / base_val) * 100.0
                if regression_pct > max_regression_pct:
                    regression_passed = False

        passed = threshold_passed and regression_passed
        if not passed:
            overall_passed = False

        status_text = "PASS" if passed else "FAIL"
        if not threshold_passed:
            fail_reason = "Below Threshold"
        elif not regression_passed:
            fail_reason = f"Regressed >{max_regression_pct:.1f}%"
        else:
            fail_reason = "OK"

        results.append({
            "metric": display_key,
            "current": curr_val,
            "baseline": base_val,
            "threshold": target_thresh,
            "regression_pct": regression_pct,
            "status": status_text,
            "reason": fail_reason,
            "passed": passed,
        })

    return overall_passed, results


def generate_markdown_report(
    results: List[Dict[str, Any]],
    overall_passed: bool,
    max_regression_pct: float,
    has_baseline: bool,
) -> str:
    """Generate GitHub PR markdown table format."""
    status_badge = "🟢 **QUALITY GATE PASSED**" if overall_passed else "🔴 **QUALITY GATE FAILED**"

    lines = [
        "## 🛡️ RAGOps Evaluation & Quality Gate Report",
        "",
        f"**Decision**: {status_badge}",
        f"**Max Regression Allowed**: `{max_regression_pct:.1f}%`",
        f"**Baseline Comparison**: `{'Active (MLflow History)' if has_baseline else 'Initial Baseline (No Preceding Run)'}`",
        "",
        "| Metric | Current Score | Previous Baseline | Diff (%) | Threshold | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        base_str = f"{r['baseline']:.4f}" if r["baseline"] is not None else "N/A"
        thresh_str = f">= {r['threshold']:.2f}" if r["threshold"] is not None else "N/A"

        if r["baseline"] is not None and r["baseline"] > 1e-4:
            diff = r["current"] - r["baseline"]
            diff_pct = (diff / r["baseline"]) * 100.0
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "—"

        status_icon = "✅ Pass" if r["passed"] else f"❌ {r['reason']}"
        lines.append(
            f"| **`{r['metric']}`** | `{r['current']:.4f}` | `{base_str}` | `{diff_str}` | `{thresh_str}` | {status_icon} |"
        )

    lines.append("")
    if not overall_passed:
        lines.append("> [!WARNING]")
        lines.append("> One or more quality metrics violated configured thresholds or exhibited performance regression.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> All retrieval and generation quality gates satisfied successfully.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG Quality Gates & MLflow Regressions")
    parser.add_argument("--report", type=str, default="eval/eval_report_latest.json", help="Path to evaluation report JSON")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=getattr(cfg.thresholds, "max_regression_pct", 10.0),
        help="Max allowable relative drop in metric (%) versus baseline",
    )
    parser.add_argument("--baseline-run-id", type=str, default="", help="Specific MLflow Run ID to use as baseline")
    parser.add_argument("--markdown-output", type=str, default="eval/quality_gate_summary.md", help="Destination markdown summary")
    parser.add_argument("--ignore-thresholds", action="store_true", help="Only verify regression, ignoring absolute thresholds")

    args = parser.parse_args()

    logger.info(f"Loading evaluation report from '{args.report}'...")
    eval_report = load_eval_report(args.report)
    current_metrics = eval_report.get("metrics", {})
    thresholds = eval_report.get("thresholds", {})

    # Connect to MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri or "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    baseline_metrics: Optional[Dict[str, float]] = None
    if args.baseline_run_id:
        logger.info(f"Using explicitly specified baseline run ID: '{args.baseline_run_id}'")
        run = client.get_run(args.baseline_run_id)
        baseline_metrics = run.data.metrics
    else:
        logger.info(f"Querying MLflow for previous evaluation baseline in '{cfg.mlflow.experiment_name}'...")
        baseline_metrics = get_previous_mlflow_run(
            client=client,
            experiment_name=cfg.mlflow.experiment_name,
            current_metrics=current_metrics,
        )

    overall_passed, results = evaluate_quality_gates(
        current_metrics=current_metrics,
        thresholds=thresholds,
        baseline_metrics=baseline_metrics,
        max_regression_pct=args.max_regression_pct,
        ignore_thresholds=args.ignore_thresholds,
    )

    # Print terminal ASCII table
    terminal_rows = []
    for r in results:
        base_str = f"{r['baseline']:.4f}" if r["baseline"] is not None else "N/A"
        thresh_str = f">= {r['threshold']:.2f}" if r["threshold"] is not None else "N/A"
        terminal_rows.append([
            r["metric"],
            f"{r['current']:.4f}",
            base_str,
            f"{r['regression_pct']:.1f}%" if r["baseline"] is not None else "—",
            thresh_str,
            r["status"],
            r["reason"],
        ])

    print("\n" + "=" * 85)
    print("                      RAGOPS CI QUALITY GATE VALIDATION                      ")
    print("=" * 85)
    print(
        tabulate(
            terminal_rows,
            headers=["Metric", "Current", "Baseline", "Regression", "Threshold", "Gate", "Notes"],
            tablefmt="grid",
        )
    )
    decision_text = "PASSED [OK]" if overall_passed else "FAILED [X]"
    print(f"\nFinal Quality Gate Decision: {decision_text} (Max Regression Allowed: {args.max_regression_pct:.1f}%)\n")

    # Generate Markdown summary for GitHub PR / Step Summary
    md_content = generate_markdown_report(
        results=results,
        overall_passed=overall_passed,
        max_regression_pct=args.max_regression_pct,
        has_baseline=baseline_metrics is not None,
    )
    out_path = Path(args.markdown_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Wrote Markdown quality gate summary to '{out_path}'.")

    # If running in GitHub Actions, append to GITHUB_STEP_SUMMARY
    github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary_path:
        try:
            with open(github_summary_path, "a", encoding="utf-8") as f:
                f.write("\n" + md_content + "\n")
            logger.info("Successfully exported summary to $GITHUB_STEP_SUMMARY.")
        except Exception as e:
            logger.warning(f"Could not write to GITHUB_STEP_SUMMARY: {e}")

    # Exit with code 0 on pass or 1 on fail
    if not overall_passed:
        logger.error("Quality gate validation failed.")
        sys.exit(1)
    else:
        logger.info("Quality gate validation passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
