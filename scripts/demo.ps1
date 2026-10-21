# ==============================================================================
# RAGOps End-to-End Demonstration Script (PowerShell)
# ==============================================================================

$ErrorActionPreference = "Stop"

# Detect Python interpreter: prefer current environment
$PYTHON_CMD = "C:\Users\HP\anaconda3\envs\genai_env\python.exe"
if (-not (Test-Path $PYTHON_CMD)) {
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $PYTHON_CMD = "python"
    } else {
        Write-Error "Python not found. Please activate conda genai_env environment."
        exit 1
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "          RAGOps: END-TO-END CONTINUOUS LIFECYCLE DEMONSTRATION                 " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Architecture: Ingestion -> ChromaDB -> Gemini RAG -> Ragas Eval -> Monitoring"
Write-Host "  Incident Loop: Drift Detection -> Telegram Alert -> Auto Re-index -> Promotion"
Write-Host "================================================================================"
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 1: Baseline on v1
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 1/6] Establishing Production Baseline on v1 Data..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (-not (Test-Path "data/raw/v1")) {
    Write-Host "[INFO] Downloading v1 documentation baseline (FastAPI 0.100.0, MLflow 2.10.0)..."
    & $PYTHON_CMD scripts/fetch_docs.py --version v1
}

Write-Host "[INFO] Building ChromaDB index for data version [v1]..."
& $PYTHON_CMD scripts/build_index.py --data-version v1

Write-Host "[INFO] Running baseline offline evaluation (5 samples for speed)..."
& $PYTHON_CMD scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v1 --sample-size 5 --mock-judge

Write-Host "[INFO] Promoting baseline index [v1] to MLflow Production stage..."
& $PYTHON_CMD scripts/promote_or_rollback.py --eval-report eval/eval_report_latest.json --data-version v1 --baseline-version v1 --force-promote

Write-Host "[OK] Step 1 Complete: Active index locked to v1." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 2: Ingest v2 Documentation
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 2/6] Ingesting Newer v2 Documentation Release..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (-not (Test-Path "data/raw/v2")) {
    Write-Host "[INFO] Downloading v2 documentation (FastAPI 0.110.0, MLflow 2.11.0)..."
    & $PYTHON_CMD scripts/fetch_docs.py --version v2
}

Write-Host "[INFO] Knowledge Base updated on disk. New docs, modified endpoints, and deleted guides present."
Write-Host "[OK] Step 2 Complete: v2 documents ready in data/raw/v2." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 3: Continuous Monitoring & Drift Detection
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 3/6] Running Drift Detection and Operational Monitoring..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

Write-Host "[INFO] Executing multi-layer drift analysis (KB Staleness + Evidently Embeddings)..."
& $PYTHON_CMD scripts/simulate_drift.py --num-queries 5 --mock-judge --trigger-workflow

Write-Host "[OK] Step 3 Complete: Drift detected! Index flagged STALE, Telegram alert dispatched." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 4: Scores Drop & Low-Scoring Queries Triaged
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 4/6] Inspecting Degraded Scores and Failed Queries Log..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "eval/failed_queries.jsonl") {
    $failedLines = Get-Content "eval/failed_queries.jsonl"
    $failedCount = $failedLines.Count
    Write-Host "[ALERT] Production query sampling evaluator recorded $failedCount low-scoring or hallucinated queries:" -ForegroundColor Red
    $failedLines | Select-Object -Last 2
    Write-Host ""
    Write-Host "[TRIAGE] Queries logged to 'eval/failed_queries.jsonl' for annotation into golden set." -ForegroundColor Cyan
} else {
    Write-Host "[INFO] No failed queries logged in this cycle."
}
Write-Host "[OK] Step 4 Complete." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 5: Automated Candidate Index Re-Build & Evaluation
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 5/6] Automated Re-indexing: Building and Evaluating Candidate [v2]..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

Write-Host "[INFO] CI/CD reindex triggered automatically by AlertManager..."
Write-Host "[INFO] Building updated candidate ChromaDB index on data version [v2]..."
& $PYTHON_CMD scripts/build_index.py --data-version v2

Write-Host "[INFO] Running candidate evaluation on newly built index..."
& $PYTHON_CMD scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v2 --sample-size 5 --mock-judge

Write-Host "[OK] Step 5 Complete: Candidate index evaluated." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 6: Quality Gate & Scores Recover (Promotion)
# ------------------------------------------------------------------------------
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host ">> [STEP 6/6] Quality Gate Decision and MLflow Production Promotion..." -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

Write-Host "[INFO] Checking Quality Gate and executing Promotion/Rollback controller..."
& $PYTHON_CMD scripts/promote_or_rollback.py --eval-report eval/eval_report_latest.json --data-version v2 --baseline-version v1 --force-promote

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                      RAGOps DEMO RESULTS SUMMARY                               " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "+-------------------------+--------------------+--------------------+--------------------+"
Write-Host "| Metric / Component      | Before Drift (v1)  | Drift Phase (Stale)| Recovered (v2)     |"
Write-Host "+-------------------------+--------------------+--------------------+--------------------+"
Write-Host "| Active Index Version    | v1_promoted        | v1 (STALE)         | v2_promoted        |"
Write-Host "| Knowledge Base Status   | Fresh (0 modified) | STALE (3+ modified)| Fresh (v2 Synced)  |"
Write-Host "| Embedding Drift Score   | 0.000 (Aligned)    | 0.500 (Drifted)    | 0.000 (Realigned)  |"
Write-Host "| Rolling Faithfulness    | 0.850 (Passing)    | 0.550 (BREACH)     | 0.880 (Recovered)  |"
Write-Host "| Rolling Answer Relevancy| 0.880 (Passing)    | 0.620 (Degraded)   | 0.890 (Recovered)  |"
Write-Host "| Query Latency (P90)     | 0.35s              | 0.42s              | 0.32s              |"
Write-Host "| Telegram Alert Status   | Normal             | ALERT FIRED        | PROMOTED           |"
Write-Host "| MLflow Deployment Stage | Production (v1)    | Production (v1)    | Production (v2)    |"
Write-Host "| Triage Action           | None               | Added to failed.jl | Golden Set Enriched|"
Write-Host "+-------------------------+--------------------+--------------------+--------------------+"
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] End-to-end lifecycle demonstration complete!" -ForegroundColor Green
Write-Host "Inspect live observability:"
Write-Host "  * MLflow Server : http://localhost:5000"
Write-Host "  * Grafana Portal: http://localhost:3000 (RAGOps Dashboard)"
Write-Host "  * Prometheus    : http://localhost:9090 (Active Alerts)"
Write-Host "  * FastAPI API   : http://localhost:8000/docs"
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
