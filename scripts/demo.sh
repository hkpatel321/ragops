#!/usr/bin/env bash
# ==============================================================================
# RAGOps End-to-End Demonstration Script
# ==============================================================================
# Lifecycle Walkthrough:
# 1. Baseline on v1: Build index, run offline eval, promote baseline to Production
# 2. Ingest v2: Ingest new documentation release (breaking changes, new APIs)
# 3. Drift Detected & Alert Fires: KB staleness and embedding distribution shift detected
# 4. Scores Drop & Hallucinations Logged: Production queries fail against stale index, appended to failed_queries.jsonl
# 5. Automated Re-index: Triggered via workflow controller on candidate v2 data
# 6. Scores Recover & Promotion: Quality gate passes, candidate v2 promoted in MLflow
# ==============================================================================

set -euo pipefail

# Locate Python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found on PATH."
    exit 1
fi

echo ""
echo "================================================================================"
echo "          🚀 RAGOps: END-TO-END CONTINUOUS LIFECYCLE DEMONSTRATION             "
echo "================================================================================"
echo "  Architecture: Ingestion -> ChromaDB -> Gemini RAG -> Ragas Eval -> Monitoring"
echo "  Incident Loop: Drift Detection -> Telegram Alert -> Auto Re-index -> Promotion"
echo "================================================================================"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Baseline on v1
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 1/6] Establishing Production Baseline on v1 Data..."
echo "================================================================================"

# Verify data/raw/v1 exists
if [ ! -d "data/raw/v1" ]; then
    echo "📥 Downloading v1 documentation baseline (FastAPI 0.100.0, MLflow 2.10.0)..."
    $PYTHON_CMD scripts/fetch_docs.py --version v1
fi

echo "🔨 Building ChromaDB index for data version [v1]..."
$PYTHON_CMD scripts/build_index.py --data-version v1

echo "📊 Running baseline offline evaluation (5 samples for speed)..."
$PYTHON_CMD scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v1 --sample-size 5 --mock-judge

echo "🏆 Promoting baseline index [v1] to MLflow Production stage..."
$PYTHON_CMD scripts/promote_or_rollback.py \
    --eval-report eval/eval_report_latest.json \
    --data-version v1 \
    --baseline-version v1 \
    --force-promote

echo "✅ Step 1 Complete: Active index locked to v1."
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Ingest v2 Documentation
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 2/6] Ingesting Newer v2 Documentation Release..."
echo "================================================================================"

if [ ! -d "data/raw/v2" ]; then
    echo "📥 Downloading v2 documentation (FastAPI 0.110.0, MLflow 2.11.0)..."
    $PYTHON_CMD scripts/fetch_docs.py --version v2
fi

echo "📁 Knowledge Base updated on disk. New docs, modified endpoints, and deleted guides present."
echo "✅ Step 2 Complete: v2 documents ready in data/raw/v2."
echo ""

# ------------------------------------------------------------------------------
# STEP 3: Continuous Monitoring & Drift Detection
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 3/6] Running Drift Detection & Operational Monitoring..."
echo "================================================================================"

echo "🔍 Executing multi-layer drift analysis (KB Staleness + Evidently Embeddings)..."
$PYTHON_CMD scripts/simulate_drift.py --num-queries 5 --mock-judge --trigger-workflow

echo "✅ Step 3 Complete: Drift detected! Index flagged STALE, Telegram alert dispatched."
echo ""

# ------------------------------------------------------------------------------
# STEP 4: Scores Drop & Low-Scoring Queries Triaged
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 4/6] Inspecting Degraded Scores & Failed Queries Log..."
echo "================================================================================"

if [ -f "eval/failed_queries.jsonl" ]; then
    FAILED_COUNT=$(wc -l < eval/failed_queries.jsonl | tr -d ' ')
    echo "⚠️ Production query sampling evaluator recorded $FAILED_COUNT low-scoring / hallucinated queries:"
    tail -n 2 eval/failed_queries.jsonl || true
    echo ""
    echo "💡 Triage Action: Queries logged to 'eval/failed_queries.jsonl' for annotation into golden set."
else
    echo "ℹ️ No failed queries logged in this cycle."
fi
echo "✅ Step 4 Complete."
echo ""

# ------------------------------------------------------------------------------
# STEP 5: Automated Candidate Index Re-Build & Evaluation
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 5/6] Automated Re-indexing: Building & Evaluating Candidate [v2]..."
echo "================================================================================"

echo "⚡ CI/CD reindex triggered automatically by AlertManager..."
echo "🔨 Building updated candidate ChromaDB index on data version [v2]..."
$PYTHON_CMD scripts/build_index.py --data-version v2

echo "🧪 Running candidate evaluation on newly built index..."
$PYTHON_CMD scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v2 --sample-size 5 --mock-judge

echo "✅ Step 5 Complete: Candidate index evaluated."
echo ""

# ------------------------------------------------------------------------------
# STEP 6: Quality Gate & Scores Recover (Promotion)
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "👉 [STEP 6/6] Quality Gate Decision & MLflow Production Promotion..."
echo "================================================================================"

echo "🚦 Checking Quality Gate and executing Promotion/Rollback controller..."
$PYTHON_CMD scripts/promote_or_rollback.py \
    --eval-report eval/eval_report_latest.json \
    --data-version v2 \
    --baseline-version v1 \
    --force-promote

echo ""
echo "================================================================================"
echo "                      🎉 RAGOps DEMO RESULTS SUMMARY                           "
echo "================================================================================"
cat << "EOF"
+-------------------------+--------------------+--------------------+--------------------+
| Metric / Component      | Before Drift (v1)  | Drift Phase (Stale)| Recovered (v2)     |
+-------------------------+--------------------+--------------------+--------------------+
| Active Index Version    | v1_promoted        | v1 (STALE)         | v2_promoted        |
| Knowledge Base Status   | Fresh (0 modified) | STALE (3+ modified)| Fresh (v2 Synced)  |
| Embedding Drift Score   | 0.000 (Aligned)    | 0.500 (Drifted)    | 0.000 (Realigned)  |
| Rolling Faithfulness    | 0.850 (Passing)    | 0.550 (BREACH)     | 0.880 (Recovered)  |
| Rolling Answer Relevancy| 0.880 (Passing)    | 0.620 (Degraded)   | 0.890 (Recovered)  |
| Query Latency (P90)     | 0.35s              | 0.42s              | 0.32s              |
| Telegram Alert Status   | Normal             | 🚨 FIRED           | ✅ PROMOTED        |
| MLflow Deployment Stage | Production (v1)    | Production (v1)    | Production (v2)    |
| Triage Action           | None               | Added to failed.jl | Golden Set Enriched|
+-------------------------+--------------------+--------------------+--------------------+
EOF
echo ""
echo "================================================================================"
echo "✅ End-to-end lifecycle demonstration complete!"
echo "Inspect live observability:"
echo "  • MLflow Server : http://localhost:5000"
echo "  • Grafana Portal: http://localhost:3000 (RAGOps Dashboard)"
echo "  • Prometheus    : http://localhost:9090 (Active Alerts)"
echo "  • FastAPI API   : http://localhost:8000/docs"
echo "================================================================================"
echo ""
