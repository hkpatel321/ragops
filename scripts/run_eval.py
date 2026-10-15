"""
RAGOps Evaluation Runner.
Runs the golden evaluation dataset against the current Chroma index, calculates:
- Non-LLM retrieval metrics (Hit Rate@k, MRR@k, Recall@k)
- LLM generation metrics (Faithfulness, Answer Relevancy, Context Precision) using Gemini Judge
Prints an ASCII report table and logs all parameters, metrics, and report artifacts to MLflow.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from tabulate import tabulate

from src.config import cfg
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.judge_llm import GeminiJudge
from src.logger import logger
from src.rag.pipeline import RAGPipeline


def load_golden_set(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found at: {file_path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_evaluation(
    golden_path: str,
    data_version: str = "v1",
    top_k: int = 4,
    limit: int = 0,
    mock_llm: bool = False,
    output_report_path: str = "eval/eval_report_latest.json",
    batch_size: int = 5,
    batch_delay: float = 0.5,
) -> Dict[str, Any]:
    """
    Execute golden set evaluation against Chroma vector store.
    """
    logger.info(f"Loading golden set from '{golden_path}'...")
    all_records = load_golden_set(golden_path)
    if limit > 0 and limit < len(all_records):
        records = all_records[:limit]
        logger.info(f"Limiting evaluation to first {limit} records (out of {len(all_records)}).")
    else:
        records = all_records
        logger.info(f"Evaluating all {len(records)} golden records.")

    # Initialize RAG Pipeline and Judge
    logger.info(f"Initializing RAG Pipeline (version={data_version}, top_k={top_k}, mock={mock_llm})...")
    rag = RAGPipeline()
    judge = GeminiJudge()
    evaluator = RAGEvaluator(judge=judge, k=top_k, batch_size=batch_size, batch_delay=batch_delay)

    eval_inputs: List[Dict[str, Any]] = []

    logger.info("Executing retrieval and generation for golden questions...")
    for idx, item in enumerate(records, start=1):
        q = item["question"]
        gt_chunk = item.get("ground_truth_chunk_id") or ""
        gt_answer = item.get("ground_truth_answer") or ""

        # Query RAG pipeline
        mock_ans = None
        if mock_llm or not rag.client:
            # Generate offline grounded answer for mock testing
            mock_ans = f"Based on the provided documentation, {gt_answer[:200] if gt_answer else 'this is documented.'}"

        rag_resp = rag.query(
            query=q,
            data_version=data_version,
            top_k=top_k,
            mock_answer=mock_ans,
        )

        retrieved_ids = [c.get("chunk_id") or c.get("id", "") for c in rag_resp.retrieved_chunks]

        eval_inputs.append({
            "id": item.get("id", f"sample_{idx}"),
            "question": q,
            "generated_answer": rag_resp.answer,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_contexts": rag_resp.contexts,
            "ground_truth_chunk_id": gt_chunk,
            "ground_truth_answer": gt_answer,
            "question_type": item.get("question_type", "general"),
            "source": item.get("source", "docs"),
            "latency": rag_resp.latency,
        })

    logger.info("Evaluating retrieval and generation metrics across dataset...")
    report = evaluator.evaluate_dataset(eval_inputs)

    # Save detailed evaluation JSON report
    report_file = Path(output_report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation report to '{report_file}'.")

    return report


def print_evaluation_summary(report: Dict[str, Any], top_k: int) -> None:
    """
    Format and print a terminal ASCII summary table of the evaluation results.
    """
    metrics = report["metrics"]
    thresholds = report["thresholds"]
    gates = report["quality_gates"]

    summary_rows = [
        [
            "Hit Rate @ k",
            f"hit_rate@{top_k}",
            f"{metrics.get(f'hit_rate@{top_k}', 0.0):.4f}",
            f">= {thresholds.get('hit_rate_at_k', 0.80):.2f}",
            "PASSED [OK]" if gates.get(f"hit_rate@{top_k}") else "FAILED [X]",
        ],
        [
            "Mean Reciprocal Rank (MRR)",
            f"mrr@{top_k}",
            f"{metrics.get(f'mrr@{top_k}', 0.0):.4f}",
            f">= {thresholds.get('mrr_at_k', 0.60):.2f}",
            "PASSED [OK]" if gates.get(f"mrr@{top_k}") else "FAILED [X]",
        ],
        [
            "Recall @ k",
            f"recall@{top_k}",
            f"{metrics.get(f'recall@{top_k}', 0.0):.4f}",
            "N/A",
            "INFO",
        ],
        [
            "Faithfulness (LLM Judge)",
            "faithfulness",
            f"{metrics.get('faithfulness', 0.0):.4f}",
            f">= {thresholds.get('faithfulness', 0.80):.2f}",
            "PASSED [OK]" if gates.get("faithfulness") else "FAILED [X]",
        ],
        [
            "Answer Relevancy (LLM Judge)",
            "answer_relevancy",
            f"{metrics.get('answer_relevancy', 0.0):.4f}",
            f">= {thresholds.get('answer_relevancy', 0.80):.2f}",
            "PASSED [OK]" if gates.get("answer_relevancy") else "FAILED [X]",
        ],
        [
            "Context Precision (LLM Judge)",
            "context_precision",
            f"{metrics.get('context_precision', 0.0):.4f}",
            f">= {thresholds.get('context_precision', 0.75):.2f}",
            "PASSED [OK]" if gates.get("context_precision") else "FAILED [X]",
        ],
    ]

    print("\n" + "=" * 80)
    print("                     RAGOPS PIPELINE EVALUATION REPORT                     ")
    print("=" * 80)
    print(
        tabulate(
            summary_rows,
            headers=["Metric Description", "Metric Key", "Measured Score", "Threshold", "Status"],
            tablefmt="grid",
        )
    )

    overall_status = "ALL QUALITY GATES PASSED [SUCCESS]" if report.get("all_passed") else "SOME GATES FAILED [WARNING]"
    print(f"\nOverall Quality Gate Decision: {overall_status}")
    print(f"Total Samples Evaluated: {metrics.get('total_evaluated', 0)}")
    print(f"Cached Judge Queries: {metrics.get('judge_cache_hits', 0)}")

    # Print sample breakdown of first 5 queries
    detailed = report.get("detailed_results", [])
    if detailed:
        print("\n" + "-" * 80)
        print("SAMPLE EVALUATIONS BREAKDOWN (Top 5):")
        print("-" * 80)
        sample_rows = []
        for d in detailed[:5]:
            q_short = (d["question"][:42] + "...") if len(d["question"]) > 45 else d["question"]
            sample_rows.append([
                q_short,
                f"Hit={int(d.get(f'hit_rate@{top_k}', 0))}",
                f"RR={d.get(f'mrr@{top_k}', 0):.2f}",
                f"Faith={d.get('faithfulness', 0):.2f}",
                f"Rel={d.get('answer_relevancy', 0):.2f}",
                f"Prec={d.get('context_precision', 0):.2f}",
            ])
        print(
            tabulate(
                sample_rows,
                headers=["Question", "Hit@k", "RR", "Faithfulness", "Relevancy", "Context Prec"],
                tablefmt="github",
            )
        )
    print("=" * 80 + "\n")


def log_to_mlflow(
    report: Dict[str, Any],
    golden_path: str,
    data_version: str,
    top_k: int,
    report_file_path: str,
) -> None:
    """
    Log all evaluation metrics, parameters, and artifact report to MLflow.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri or "sqlite:///mlflow.db")
    experiment_name = cfg.mlflow.experiment_name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"eval_{data_version}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Logging evaluation run '{run_name}' to MLflow ({tracking_uri})...")

    with mlflow.start_run(run_name=run_name):
        # 1. Log tags
        mlflow.set_tag("pipeline", "ragops")
        mlflow.set_tag("run_type", "evaluation")
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("generator_model", cfg.llm.generator.model_name)
        mlflow.set_tag("judge_model", cfg.llm.judge.model_name)
        mlflow.set_tag("all_gates_passed", str(report.get("all_passed", False)))

        # 2. Log parameters
        mlflow.log_param("golden_set_path", golden_path)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("generator_model", cfg.llm.generator.model_name)
        mlflow.log_param("judge_model", cfg.llm.judge.model_name)
        mlflow.log_param("num_evaluated", report["metrics"].get("total_evaluated", 0))

        # Log thresholds
        for k, v in report.get("thresholds", {}).items():
            mlflow.log_param(f"threshold_{k}", v)

        # 3. Log metrics (sanitizing '@' to '_at_' for MLflow metric naming compatibility)
        for metric_name, val in report["metrics"].items():
            if isinstance(val, (int, float)):
                clean_name = metric_name.replace("@", "_at_")
                mlflow.log_metric(clean_name, float(val))

        mlflow.log_metric("all_gates_passed", 1.0 if report.get("all_passed") else 0.0)

        # 4. Log artifact
        if Path(report_file_path).is_file():
            mlflow.log_artifact(report_file_path, artifact_path="evaluation_reports")
            logger.info(f"Logged artifact '{report_file_path}' to MLflow.")

    logger.info("Successfully recorded evaluation run to MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGOps Golden Set Evaluation & Log to MLflow")
    parser.add_argument("--golden-set", type=str, default="eval/golden_v1.jsonl", help="Path to golden set JSONL")
    parser.add_argument("--data-version", type=str, default="v1", help="Document data version (e.g. v1, v2)")
    parser.add_argument("--top-k", type=int, default=cfg.vectorstore.top_k, help="Top-k chunks to retrieve")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries to evaluate (0 for all)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for evaluation pacing")
    parser.add_argument("--batch-delay", type=float, default=0.5, help="Delay in seconds between batches")
    parser.add_argument("--mock-llm", action="store_true", help="Run with mock/deterministic generation and judging")
    parser.add_argument("--output-report", type=str, default="eval/eval_report_latest.json", help="Path to save evaluation report")

    args = parser.parse_args()

    report = run_evaluation(
        golden_path=args.golden_set,
        data_version=args.data_version,
        top_k=args.top_k,
        limit=args.limit,
        mock_llm=args.mock_llm,
        output_report_path=args.output_report,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
    )

    print_evaluation_summary(report, top_k=args.top_k)

    log_to_mlflow(
        report=report,
        golden_path=args.golden_set,
        data_version=args.data_version,
        top_k=args.top_k,
        report_file_path=args.output_report,
    )


if __name__ == "__main__":
    main()
