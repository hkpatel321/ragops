"""
Drift Simulation and Demonstration Script for RAGOps.
Simulates production drift scenarios:
1. Switches raw document knowledge base from v1 to v2, detecting file additions/modifications and marking the index STALE.
2. Compares embedding distributions (query embeddings and doc embeddings) across reference and drifted windows using Evidently.
3. Replays queries through the RAG pipeline, logging requests, and triggers the async SamplingEvaluator to score 10% with LLM Judge.
4. Exports Evidently HTML/JSON drift reports and verifies Prometheus gauges.

Usage:
    python scripts/simulate_drift.py
    python scripts/simulate_drift.py --num-queries 5 --mock-judge
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.storage import QueryStore
from src.chunking import MarkdownChunker
from src.embedding import Embedder
from src.ingestion import load_documents_from_directory
from src.logger import logger
from src.monitoring.embedding_drift import EmbeddingDriftDetector
from src.monitoring.kb_drift import KBDriftDetector
from src.monitoring.sampling_evaluator import SamplingEvaluator
from src.rag.pipeline import RAGPipeline


def simulate_kb_drift(v1_dir: Path, v2_dir: Path, output_dir: Path):
    """
    Simulate Knowledge Base drift by switching baseline from v1 to v2.
    """
    logger.info("=" * 60)
    logger.info("[Step 1/3] Demonstrating Knowledge Base Drift & Staleness...")
    logger.info("=" * 60)

    kb_detector = KBDriftDetector(manifests_dir=str(output_dir / "manifests"))

    # 1. Establish baseline manifest for v1
    logger.info(f"Indexing reference knowledge base from: {v1_dir}")
    v1_manifest = kb_detector.build_manifest(v1_dir)
    kb_detector.save_manifest(v1_manifest, version_tag="v1")
    ref_report = kb_detector.detect_drift(v1_dir, data_version="v1")
    logger.info(f"Baseline v1 status: Stale={ref_report.is_stale}, Total Files={ref_report.total_current_files}")

    # 2. Switch data to v2 and detect drift against v1 baseline
    logger.info(f"\n>>> SWITCHING KNOWLEDGE BASE DATA: v1 -> v2 ({v2_dir}) <<<")
    drift_report = kb_detector.detect_drift(
        current_dir=v2_dir,
        data_version="v2",
        reference_manifest=v1_manifest,
    )

    print("\n" + "=" * 50)
    print("      KNOWLEDGE BASE DRIFT REPORT (v1 -> v2)      ")
    print("=" * 50)
    print(f"Index Marked Stale?     : {'YES [CRITICAL]' if drift_report.is_stale else 'NO [FRESH]'}")
    print(f"Total Reference Files   : {drift_report.total_reference_files}")
    print(f"Total Current Files     : {drift_report.total_current_files}")
    print(f"New Files Added         : {len(drift_report.new_files)}")
    print(f"Files Modified/Changed  : {len(drift_report.changed_files)}")
    print(f"Files Removed           : {len(drift_report.removed_files)}")

    if drift_report.changed_files:
        print("\nSample Modified Files:")
        for f in drift_report.changed_files[:5]:
            print(f"  * [MODIFIED] {f}")

    if drift_report.new_files:
        print("\nSample New Files:")
        for f in drift_report.new_files[:5]:
            print(f"  * [NEW] {f}")

    print("=" * 50 + "\n")
    return drift_report


def simulate_embedding_drift(v1_dir: Path, v2_dir: Path, output_dir: Path):
    """
    Simulate embedding drift on document chunks and query spaces using Evidently.
    """
    logger.info("=" * 60)
    logger.info("[Step 2/3] Demonstrating Embedding Drift Detection (Evidently)...")
    logger.info("=" * 60)

    chunker = MarkdownChunker()
    embedder = Embedder()
    drift_detector = EmbeddingDriftDetector(reports_dir=str(output_dir))

    # Ingest and chunk v1 and v2 docs
    logger.info("Ingesting and chunking v1 reference documents...")
    v1_docs = load_documents_from_directory(v1_dir, version="v1")
    v1_chunks = chunker.chunk_documents(v1_docs[:5])  # Sample 5 docs for fast demo

    logger.info("Ingesting and chunking v2 shifted documents...")
    v2_docs = load_documents_from_directory(v2_dir, version="v2")
    v2_chunks = chunker.chunk_documents(v2_docs[:5])

    v1_texts = [c.text for c in v1_chunks[:30]]
    v2_texts = [c.text for c in v2_chunks[:30]]

    logger.info(f"Computing embeddings for {len(v1_texts)} v1 chunks vs {len(v2_texts)} v2 chunks...")
    v1_doc_embeddings = embedder.embed_batch(v1_texts)
    v2_doc_embeddings = embedder.embed_batch(v2_texts)

    # 1. Document Embedding Drift
    doc_result = drift_detector.detect_drift(
        reference_embeddings=v1_doc_embeddings,
        current_embeddings=v2_doc_embeddings,
        embedding_type="doc",
        report_filename="embedding_drift_doc",
    )

    # 2. Query Embedding Drift: reference baseline queries vs shifted queries
    ref_queries = [
        "How do I log parameters in MLflow?",
        "How to define route handlers in FastAPI?",
        "What is an MLflow run and experiment?",
        "How to use Pydantic models for request bodies?",
        "How do I install FastAPI dependencies?",
        "What is MLflow tracking URI configuration?",
        "How to return JSON responses in FastAPI?",
        "How to log metrics with mlflow.log_metric?",
    ]

    shifted_queries = [
        "What are the breaking changes in MLflow 2.20 compared to 2.10?",
        "How does FastAPI 0.115 handle asynchronous background tasks and life-span events?",
        "Is MLflow autologging compatible with the latest PyTorch Lightning releases in v2?",
        "How to implement custom HTTP middleware in newer FastAPI releases?",
        "Describe MLflow Prompt Registry and GenAI evaluation integrations in 2.20",
        "How does dependency injection differ in FastAPI v0.115 with Annotated typing?",
        "How to migrate MLflow model artifacts across S3 bucket namespaces in 2.20?",
        "Explain FastAPI Websockets streaming security upgrades in v0.115",
    ]

    logger.info("Computing embeddings for reference queries vs shifted queries...")
    ref_query_embs = embedder.embed_batch(ref_queries)
    shifted_query_embs = embedder.embed_batch(shifted_queries)

    query_result = drift_detector.detect_drift(
        reference_embeddings=ref_query_embs,
        current_embeddings=shifted_query_embs,
        embedding_type="query",
        report_filename="embedding_drift_query",
    )

    print("\n" + "=" * 50)
    print("      EMBEDDING DRIFT DETECTION REPORT (Evidently)      ")
    print("=" * 50)
    print("DOCUMENT CHUNKS (v1 vs v2):")
    print(f"  * Drift Detected?     : {'YES [ALERT]' if doc_result.drift_detected else 'NO [NORMAL]'}")
    print(f"  * Drift Score         : {doc_result.drift_score:.4f}")
    print(f"  * Drift Share         : {doc_result.drift_share:.2%}")
    print(f"  * Cosine Dist Shift   : {doc_result.mean_cosine_shift:.4f}")
    print(f"  * Report HTML         : {doc_result.report_html_path}")
    print(f"  * Metrics JSON        : {doc_result.metrics_json_path}")

    print("\nQUERY SPACE (Reference vs Shifted Incoming Traffic):")
    print(f"  * Drift Detected?     : {'YES [ALERT]' if query_result.drift_detected else 'NO [NORMAL]'}")
    print(f"  * Drift Score         : {query_result.drift_score:.4f}")
    print(f"  * Drift Share         : {query_result.drift_share:.2%}")
    print(f"  * Cosine Dist Shift   : {query_result.mean_cosine_shift:.4f}")
    print(f"  * Report HTML         : {query_result.report_html_path}")
    print(f"  * Metrics JSON        : {query_result.metrics_json_path}")
    print("=" * 50 + "\n")

    return doc_result, query_result


def simulate_query_replay_and_sampling(
    golden_path: Path,
    num_queries: int,
    output_dir: Path,
    mock_judge: bool = False,
):
    """
    Replay production queries through RAG pipeline, store them, and run SamplingEvaluator.
    """
    logger.info("=" * 60)
    logger.info("[Step 3/3] Demonstrating Query Replay & 10% Sampling Evaluator...")
    logger.info("=" * 60)

    # Load golden questions or default queries
    queries_to_replay: List[str] = []
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    queries_to_replay.append(item.get("question", ""))
                    if len(queries_to_replay) >= num_queries:
                        break

    if not queries_to_replay:
        queries_to_replay = [
            "How do I log parameters in MLflow?",
            "How to define route handlers in FastAPI?",
            "What is an MLflow run?",
            "How to use Pydantic models for request bodies?",
            "How to log metrics with MLflow?",
        ]

    db_path = output_dir / "simulated_queries.db"
    store = QueryStore(db_path=str(db_path))

    pipeline = None
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        logger.warning(f"Could not initialize full RAGPipeline (using synthetic logs): {e}")

    logger.info(f"Replaying {len(queries_to_replay)} queries through pipeline and logging to QueryStore...")
    for idx, q_text in enumerate(queries_to_replay, 1):
        if pipeline:
            try:
                res = pipeline.run(q_text)
                answer = res.answer
                contexts = res.contexts
                latency = res.latency_seconds
                tokens = res.token_counts
            except Exception:
                answer = f"Synthesized answer for query {idx}"
                contexts = ["Sample context snippet 1", "Sample context snippet 2"]
                latency = 0.45
                tokens = {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}
        else:
            answer = f"Synthesized answer for query {idx}"
            contexts = ["Sample context snippet 1", "Sample context snippet 2"]
            latency = 0.45
            tokens = {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}

        import uuid
        store.log_query(
            request_id=f"sim_{uuid.uuid4().hex[:8]}",
            query=q_text,
            contexts=contexts,
            answer=answer,
            latency=latency,
            token_counts=tokens,
            index_version="v1",
            model_name="gemini-2.5-flash",
            top_k=len(contexts),
        )

    # Initialize sampling evaluator
    judge = None
    if mock_judge:
        from unittest.mock import MagicMock
        judge = MagicMock()
        judge.evaluate_sample_with_cache.return_value = (0.92, "High factual grounding verified.")

    evaluator = SamplingEvaluator(
        store=store,
        judge=judge,
        sample_rate=0.20,  # 20% for demo so at least 1-2 queries are scored immediately
        window_size=10,
    )

    eval_result = evaluator.process_new_queries()

    print("\n" + "=" * 50)
    print("      ASYNC SAMPLING EVALUATION REPORT      ")
    print("=" * 50)
    print(f"New Queries Logged      : {eval_result['new_queries_found']}")
    print(f"Sampled & LLM-Scored    : {eval_result['sampled_count']}")
    print(f"Rolling Faithfulness    : {eval_result['rolling_faithfulness']:.4f}")
    print(f"Rolling Answer Relevancy: {eval_result['rolling_relevancy']:.4f}")
    print("Prometheus Gauges Updated:")
    print(f"  * ragops_rolling_faithfulness    -> {eval_result['rolling_faithfulness']}")
    print(f"  * ragops_rolling_answer_relevancy-> {eval_result['rolling_relevancy']}")
    print("=" * 50 + "\n")

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Simulate production drift across KB, embeddings, and query traffic.")
    parser.add_argument("--v1-dir", type=str, default="data/raw/v1", help="Path to v1 reference raw docs")
    parser.add_argument("--v2-dir", type=str, default="data/raw/v2", help="Path to v2 shifted raw docs")
    parser.add_argument("--golden-path", type=str, default="eval/golden_v1.jsonl", help="Path to golden queries")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to replay")
    parser.add_argument("--output-dir", type=str, default="data/monitoring", help="Output directory for reports")
    parser.add_argument("--mock-judge", action="store_true", help="Use mock judge to preserve API quota")

    args = parser.parse_args()

    v1_dir = Path(args.v1_dir)
    v2_dir = Path(args.v2_dir)
    golden_path = Path(args.golden_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("   RAGOps PRODUCTION DRIFT & MONITORING SIMULATOR")
    print("#" * 60 + "\n")

    # 1. Knowledge Base Drift
    simulate_kb_drift(v1_dir, v2_dir, output_dir)

    # 2. Embedding Drift
    simulate_embedding_drift(v1_dir, v2_dir, output_dir)

    # 3. Query Replay & Sampling Evaluator
    simulate_query_replay_and_sampling(
        golden_path=golden_path,
        num_queries=args.num_queries,
        output_dir=output_dir,
        mock_judge=args.mock_judge,
    )

    print("\n" + "#" * 60)
    print("   DRIFT DEMO COMPLETE - ALL PROMETHEUS GAUGES UPDATED")
    print(f"   Evidently Reports Generated in: {output_dir.resolve()}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
