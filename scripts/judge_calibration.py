"""
Judge Calibration Script for RAGOps.
Compares Gemini Judge scores against 30 manually labeled ground-truth samples.
Calculates calibration and alignment metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation (r)
- Agreement Rate within tolerance (+/- 0.20)
- Classification Accuracy (High vs Low quality)
Prints a formatted comparison table and logs calibration metrics to MLflow.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from tabulate import tabulate

from src.config import cfg
from src.evaluation.judge_llm import GeminiJudge
from src.logger import logger


def load_calibration_dataset(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found at: {file_path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_calibration_stats(
    human_scores: List[float],
    judge_scores: List[float],
    tolerance: float = 0.20,
) -> Dict[str, float]:
    """
    Calculate statistical alignment between human and judge annotations.
    """
    h = np.array(human_scores, dtype=float)
    j = np.array(judge_scores, dtype=float)

    mae = float(np.mean(np.abs(j - h)))
    rmse = float(np.sqrt(np.mean((j - h) ** 2)))

    # Pearson correlation (guard against constant arrays)
    if np.std(h) > 1e-6 and np.std(j) > 1e-6:
        pearson_r, _ = stats.pearsonr(h, j)
        pearson_r = float(pearson_r)
    else:
        pearson_r = 1.0 if np.allclose(h, j) else 0.0

    # Agreement within tolerance (e.g. within +/- 0.20)
    agreements = np.abs(j - h) <= tolerance
    agreement_rate = float(np.mean(agreements))

    # Binary classification agreement (>= 0.70 is High, < 0.70 is Low)
    h_binary = (h >= 0.70).astype(int)
    j_binary = (j >= 0.70).astype(int)
    class_accuracy = float(np.mean(h_binary == j_binary))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson_r, 4),
        "agreement_rate": round(agreement_rate, 4),
        "classification_accuracy": round(class_accuracy, 4),
    }


def run_calibration(
    dataset_path: str = "eval/calibration_samples_30.jsonl",
    tolerance: float = 0.20,
    mock_judge: bool = False,
    output_report_path: str = "eval/calibration_report.json",
) -> Dict[str, Any]:
    """
    Run calibration against manually labeled samples.
    """
    logger.info(f"Loading calibration dataset from '{dataset_path}'...")
    samples = load_calibration_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} manually annotated samples.")

    judge = GeminiJudge()
    if mock_judge:
        logger.info("Running judge calibration in mock/deterministic mode.")
        judge.langchain_llm = None

    per_sample_results: List[Dict[str, Any]] = []

    human_faith: List[float] = []
    judge_faith: List[float] = []

    human_rel: List[float] = []
    judge_rel: List[float] = []

    human_prec: List[float] = []
    judge_prec: List[float] = []

    for i, s in enumerate(samples, start=1):
        q = s["question"]
        ans = s["generated_answer"]
        ctxs = s.get("contexts", [])

        # Human scores
        h_f = float(s["human_faithfulness"])
        h_r = float(s["human_answer_relevancy"])
        h_p = float(s["human_context_precision"])

        human_faith.append(h_f)
        human_rel.append(h_r)
        human_prec.append(h_p)

        # Judge scores
        j_f, f_reason = judge.evaluate_sample_with_cache(
            metric_name="faithfulness",
            question=q,
            answer=ans,
            contexts=ctxs,
        )
        j_r, r_reason = judge.evaluate_sample_with_cache(
            metric_name="answer_relevancy",
            question=q,
            answer=ans,
            contexts=ctxs,
        )
        j_p, p_reason = judge.evaluate_sample_with_cache(
            metric_name="context_precision",
            question=q,
            answer=ans,
            contexts=ctxs,
        )

        judge_faith.append(j_f)
        judge_rel.append(j_r)
        judge_prec.append(j_p)

        per_sample_results.append({
            "id": s.get("id", f"sample_{i}"),
            "question": q,
            "generated_answer": ans,
            "human_scores": {"faithfulness": h_f, "answer_relevancy": h_r, "context_precision": h_p},
            "judge_scores": {"faithfulness": round(j_f, 4), "answer_relevancy": round(j_r, 4), "context_precision": round(j_p, 4)},
            "errors": {
                "faithfulness_diff": round(abs(j_f - h_f), 4),
                "answer_relevancy_diff": round(abs(j_r - h_r), 4),
                "context_precision_diff": round(abs(j_p - h_p), 4),
            },
            "judge_reasons": {
                "faithfulness": f_reason,
                "answer_relevancy": r_reason,
                "context_precision": p_reason,
            },
            "notes": s.get("notes", ""),
        })

    # Compute stats for each metric
    faith_stats = compute_calibration_stats(human_faith, judge_faith, tolerance=tolerance)
    rel_stats = compute_calibration_stats(human_rel, judge_rel, tolerance=tolerance)
    prec_stats = compute_calibration_stats(human_prec, judge_prec, tolerance=tolerance)

    overall_mae = float(np.mean([faith_stats["mae"], rel_stats["mae"], prec_stats["mae"]]))
    overall_agreement = float(np.mean([faith_stats["agreement_rate"], rel_stats["agreement_rate"], prec_stats["agreement_rate"]]))

    calibration_report = {
        "num_samples": len(samples),
        "judge_model": judge.model_name,
        "tolerance": tolerance,
        "metrics": {
            "faithfulness": faith_stats,
            "answer_relevancy": rel_stats,
            "context_precision": prec_stats,
            "overall_mae": round(overall_mae, 4),
            "overall_agreement_rate": round(overall_agreement, 4),
        },
        "detailed_samples": per_sample_results,
    }

    # Save artifact
    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved judge calibration report to '{out_file}'.")

    return calibration_report


def print_calibration_summary(report: Dict[str, Any]) -> None:
    """
    Print an ASCII calibration table to stdout.
    """
    m = report["metrics"]
    tol = report["tolerance"]

    summary_rows = [
        [
            "Faithfulness",
            f"{m['faithfulness']['mae']:.4f}",
            f"{m['faithfulness']['rmse']:.4f}",
            f"{m['faithfulness']['pearson_r']:.4f}",
            f"{m['faithfulness']['agreement_rate'] * 100:.1f}%",
            f"{m['faithfulness']['classification_accuracy'] * 100:.1f}%",
        ],
        [
            "Answer Relevancy",
            f"{m['answer_relevancy']['mae']:.4f}",
            f"{m['answer_relevancy']['rmse']:.4f}",
            f"{m['answer_relevancy']['pearson_r']:.4f}",
            f"{m['answer_relevancy']['agreement_rate'] * 100:.1f}%",
            f"{m['answer_relevancy']['classification_accuracy'] * 100:.1f}%",
        ],
        [
            "Context Precision",
            f"{m['context_precision']['mae']:.4f}",
            f"{m['context_precision']['rmse']:.4f}",
            f"{m['context_precision']['pearson_r']:.4f}",
            f"{m['context_precision']['agreement_rate'] * 100:.1f}%",
            f"{m['context_precision']['classification_accuracy'] * 100:.1f}%",
        ],
    ]

    print("\n" + "=" * 90)
    print("                    LLM JUDGE CALIBRATION BENCHMARK REPORT                    ")
    print("=" * 90)
    print(f"Evaluator Model: {report['judge_model']} | Samples: {report['num_samples']} | Tolerance: +/- {tol}")
    print(
        tabulate(
            summary_rows,
            headers=["Evaluated Metric", "MAE", "RMSE", "Pearson r", f"Agreed (<={tol})", "Class Acc"],
            tablefmt="grid",
        )
    )
    print(f"\nOverall Judge MAE: {m['overall_mae']:.4f}")
    print(f"Overall Agreement Rate (+/- {tol}): {m['overall_agreement_rate'] * 100:.1f}%")

    # Sample comparison
    print("\n" + "-" * 90)
    print("SAMPLE COMPARISON BREAKDOWN (10 Representative Samples):")
    print("-" * 90)
    sample_table = []
    for s in report["detailed_samples"][:10]:
        sid = s["id"]
        h = s["human_scores"]
        j = s["judge_scores"]
        sample_table.append([
            sid,
            f"H:{h['faithfulness']:.2f} / J:{j['faithfulness']:.2f}",
            f"H:{h['answer_relevancy']:.2f} / J:{j['answer_relevancy']:.2f}",
            f"H:{h['context_precision']:.2f} / J:{j['context_precision']:.2f}",
            s.get("notes", "")[:35] + "...",
        ])

    print(
        tabulate(
            sample_table,
            headers=["Sample ID", "Faith (Human/Judge)", "Relevancy (Human/Judge)", "Precision (Human/Judge)", "Notes"],
            tablefmt="github",
        )
    )
    print("=" * 90 + "\n")


def log_calibration_to_mlflow(report: Dict[str, Any], report_path: str) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri or "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("RAGOps-Judge-Calibration")

    run_name = f"judge_calib_{time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Logging judge calibration to MLflow under 'RAGOps-Judge-Calibration' ({tracking_uri})...")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("pipeline", "ragops")
        mlflow.set_tag("run_type", "judge_calibration")
        mlflow.set_tag("judge_model", report["judge_model"])

        mlflow.log_param("num_samples", report["num_samples"])
        mlflow.log_param("judge_model", report["judge_model"])
        mlflow.log_param("tolerance", report["tolerance"])

        m = report["metrics"]
        mlflow.log_metric("overall_mae", m["overall_mae"])
        mlflow.log_metric("overall_agreement_rate", m["overall_agreement_rate"])

        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            for stat_k, stat_v in m[metric].items():
                mlflow.log_metric(f"{metric}_{stat_k}", float(stat_v))

        if Path(report_path).is_file():
            mlflow.log_artifact(report_path, artifact_path="calibration_reports")

    logger.info("Successfully recorded judge calibration to MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Calibration Script against 30 Manually Labeled Samples")
    parser.add_argument("--dataset", type=str, default="eval/calibration_samples_30.jsonl", help="Path to 30 labeled samples")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Tolerance threshold for score agreement")
    parser.add_argument("--mock-judge", action="store_true", help="Run judge in deterministic mock mode")
    parser.add_argument("--output-report", type=str, default="eval/calibration_report.json", help="Report destination")

    args = parser.parse_args()

    report = run_calibration(
        dataset_path=args.dataset,
        tolerance=args.tolerance,
        mock_judge=args.mock_judge,
        output_report_path=args.output_report,
    )

    print_calibration_summary(report)

    log_calibration_to_mlflow(report, report_path=args.output_report)


if __name__ == "__main__":
    main()
