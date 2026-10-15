# RAGOps: Continuous LLM-Based Evaluation and Drift Monitoring for RAG Pipelines

Production-grade MLOps platform for Retrieval-Augmented Generation (RAG) over versioned MLflow and FastAPI documentation with automated evaluation, drift monitoring, CI/CD quality gates, and observability.

---

## 1. Documentation Ingestion & Versioning (`src/ingestion`, `scripts/fetch_docs.py`)

### Architecture
* **Sources Tracked**:
  * **MLflow**: v1 (`v2.10.0`) vs v2 (`v2.20.0`)
  * **FastAPI**: v1 (`0.100.0`) vs v2 (`0.115.0`)
* **Storage Hierarchy**:
  * `data/raw/v1/fastapi/` & `data/raw/v1/mlflow/`
  * `data/raw/v2/fastapi/` & `data/raw/v2/mlflow/`
* **Data Version Control (DVC)**:
  * Both `data/raw/v1` and `data/raw/v2` are tracked with DVC (`data/raw/v1.dvc`, `data/raw/v2.dvc`).
  * Raw data files are ignored in Git, ensuring lightweight git commits while DVC handles dataset versioning and immutability.
* **Document Loader** (`src/ingestion/loader.py`):
  * Recursively loads all markdown documents.
  * Preserves full provenance metadata in `Document.metadata`:
    * `source`: `fastapi` | `mlflow`
    * `version`: `v1` | `v2`
    * `path`: file path
    * `filename`: name of file
    * `title`: auto-extracted H1/H2 header or title
    * `char_count`: length of content

### Running Ingestion
```bash
python scripts/fetch_docs.py
dvc add data/raw/v1 data/raw/v2
git add data/raw/v1.dvc data/raw/v2.dvc data/raw/.gitignore
git commit -m "update raw docs with DVC"
```

---

## 2. Markdown-Aware Chunking (`src/chunking`)

### Features
* Splits along markdown structural boundaries (`#`, `##`, `###` headers, code blocks, paragraphs) to avoid splitting code or logic midway.
* Config-driven sizing via `configs/config.yaml`:
  * `chunk_size`: 512
  * `overlap`: 64
* Generates `TextChunk` models with enriched chunk-level metadata:
  * `chunk_id`: `{version}_{source}_{doc_stem}_{chunk_idx}` (e.g. `v1_fastapi_index_0001`)
  * `section`: Section title (extracted from markdown headers)
  * `chunk_char_count`: Length of the chunk text

---

## 3. Dense Embeddings & Vector Store (`src/embedding`, `src/vectorstore`)

### Embeddings (`src/embedding/embedder.py`)
* Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* Configurable normalization (L2 cosine similarity ready) and device selection (CPU/CUDA).
* ChromaDB integration via custom `ChromaEmbeddingFunction`.

### ChromaDB Store (`src/vectorstore/store.py`)
* Persistent vector database at `data/chroma_db`.
* **Version Identification**:
  * Each build is tagged with a composite version identifier: `version_id = f"{data_version}_{config_hash}"` (e.g. `v1_9ac1e3a5`).
  * `config_hash` is an 8-character deterministic SHA-256 hash computed from `chunk_size`, `overlap`, `embedding_model`, and `distance_metric`.
* Isolated collection names per build (e.g. `ragops_docs_v1_9ac1e3a5`).

---

## 4. Index Build & MLflow Tracking (`scripts/build_index.py`)

### Usage
```bash
python scripts/build_index.py --data-version v1
python scripts/build_index.py --data-version v2
```

### MLflow Logging
* **Parameters**: `data_version`, `version_id`, `config_hash`, `chunk_size`, `overlap`, `embedding_model`, `distance_metric`, `collection_name`.
* **Metrics**: `doc_count`, `chunk_count`, `avg_chunk_chars`, `build_duration_sec`.
* **Artifacts**:
  * `metadata/index_summary.json`
  * `chroma_store/` (Full ChromaDB vector database files)
* **Tags**: `stage: index_build`, `version_id`, `data_version`.

---

## 5. Grounded RAG Pipeline (`src/rag/pipeline.py`)

### Features
* **Semantic Retrieval**: Queries the versioned ChromaDB index for the `top_k` most relevant chunks.
* **Grounded Prompting**: Implements strict grounding rules compelling the model to answer only using facts directly stated in the context, and explicitly output `"I don't know based on the provided documentation."` when the context is insufficient.
* **Google Gemini Integration**: Uses model configured in `configs/config.yaml` (`gemini-2.5-flash`), with token usage tracking (`prompt_tokens`, `completion_tokens`, `total_tokens`) and latency measurement.
* **Resilience**: Exponential backoff with jitter on HTTP 429 rate limits or transient 5xx errors (up to 4 retry attempts).
* **Structured Output (`RAGResponse`)**:
  * `query`: Input question
  * `answer`: Generated answer
  * `contexts`: Raw text of retrieved chunks
  * `retrieved_chunks`: Detailed metadata, scores, and chunk coordinates
  * `latency`: Response latency in seconds
  * `token_counts`: Prompt, completion, and total tokens
  * `version_id`: Target collection version ID

### Query CLI
```bash
# Query the live Gemini RAG assistant over version 1 docs
python src/rag/pipeline.py --query "How do path parameters work in FastAPI?" --data-version v1

# Query with explicit top_k and version_id
python src/rag/pipeline.py --query "How does MLflow tracking log metrics?" --version-id v1_9ac1e3a5 --top-k 5

# Offline mock test (verifies retrieval and prompt formatting without API credits)
python src/rag/pipeline.py --query "How to use Pydantic models in FastAPI?" --data-version v1 --mock
```

---

## 6. Golden Evaluation Dataset (`scripts/make_golden_set.py`, `eval/golden_v1.jsonl`)

### Overview
* Contains 75 carefully stratified question/ground-truth-answer pairs extracted and generated from `v1` documentation chunks.
* Balanced representation across sources:
  * **FastAPI**: 37 pairs
  * **MLflow**: 38 pairs
* Balanced distribution across query archetypes:
  * **Factual** (25 pairs): Inquiring about concepts, definitions, components, and return types.
  * **How-To** (25 pairs): Practical syntax, code examples, setup patterns, and configuration steps.
  * **Version-Specific** (25 pairs): Focusing on v1 conventions, release 0.100.0 / 2.10.0 specifics.
* **Retrieval Ground Truth**: Each item carries `ground_truth_chunk_id` and `ground_truth_context` linking directly to the ChromaDB chunk ID used as the ground truth target during retrieval evaluation.
* **DVC Tracking**: `eval/golden_v1.jsonl` is tracked and versioned using DVC (`eval/golden_v1.jsonl.dvc`).

### Commands
```bash
# Generate 75 Q&A pairs (uses Gemini if API key set, or deterministic grounded synthesis)
python scripts/make_golden_set.py --count 75

# Inspect 10 random samples for manual review
python scripts/inspect_golden_set.py --samples 10

# Track with DVC
dvc add eval/golden_v1.jsonl
git add eval/golden_v1.jsonl.dvc eval/.gitignore
git commit -m "update golden dataset with DVC"
```

---

## 7. Continuous Evaluation & Quality Gates

The `src/evaluation` package implements continuous automated evaluation of the RAG pipeline across two core dimensions:

### A. Non-LLM Retrieval Metrics
Deterministic ranking and retrieval quality computed directly without LLM overhead:
* **Hit Rate @ k**: Fraction of queries where at least one ground-truth document chunk is retrieved in the top-$k$ candidates.
* **Mean Reciprocal Rank (MRR @ k)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$, penalizing relevant chunks retrieved lower in the candidate list.
* **Recall @ k**: Proportion of all relevant ground-truth chunks successfully retrieved in the top-$k$.

### B. LLM Generation Metrics (Gemini Judge)
Grounded generation quality evaluated using Google Gemini (`gemini-2.5-pro`) via the LangChain Google GenAI integration (`ChatGoogleGenerativeAI`):
* **Faithfulness**: Verifies that every claim in the generated answer is strictly grounded in and supported by the retrieved contexts.
* **Answer Relevancy**: Assesses how directly and completely the response addresses the user's inquiry without extraneous digression.
* **Context Precision**: Determines whether the retrieved chunks are relevant and focused on the question.
* **Judge/Generator Separation**: Generation uses `gemini-2.5-flash` (`cfg.llm.generator`), whereas evaluation uses `gemini-2.5-pro` (`cfg.llm.judge`, temperature `0.0`), configured strictly through `configs/config.yaml`.

### C. Rate-Limit Safety & Disk Caching
* **Persistent Cache (`JudgeCache`)**: Caches LLM judge evaluations to disk (`data/artifacts/judge_cache.json`) indexed by `SHA256(model | metric | question | answer | contexts)` to avoid redundant API calls and prevent token waste.
* **Rate-Limit Safe Batching**: Evaluates samples in configurable batches with exponential backoff on HTTP 429/ResourceExhausted errors and pacing delays to stay within API tier limits.

### Commands

#### Run Golden Set Evaluation & Log to MLflow
```bash
# Run evaluation across the golden dataset and log to MLflow
python scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v1 --top-k 4

# Run with mock/deterministic scoring (for fast offline CI runs)
python scripts/run_eval.py --golden-set eval/golden_v1.jsonl --data-version v1 --limit 10 --mock-llm
```

#### Run Judge Calibration against 30 Manually Labeled Samples
#### Run Quality Gate & MLflow Regression Checker
```bash
# Verify absolute quality thresholds and check for performance regressions (>10%) vs MLflow baseline
python scripts/quality_gate.py --report eval/eval_report_latest.json --max-regression-pct 10.0
```

---

## 8. Automated CI/CD Quality Gates (`rag_ci.yml`)

The repository includes an end-to-end continuous integration workflow in [`.github/workflows/rag_ci.yml`](file:///.github/workflows/rag_ci.yml) triggered on every push and pull request touching `src/`, `configs/`, `data/`, `eval/`, or `scripts/`.

### CI Pipeline Workflow Steps:
1. **Linting & Code Quality**: Fast Python linting and formatting validation via `ruff check .`.
2. **Unit Test Suite**: Runs all unit tests covering ingestion, chunking, vector storage, pipeline execution, retrieval metrics, judge caching, and quality gates.
3. **Index Build**: Builds and logs the v1 vector index into ChromaDB and MLflow.
4. **Offline Evaluation**: Evaluates a reduced subset of golden queries with caching enabled to preserve API quotas.
5. **Quality Gate Verification**: Invokes [`scripts/quality_gate.py`](file:///scripts/quality_gate.py) to assert that:
   - All evaluation metrics meet the configured quality thresholds in [`configs/config.yaml`](file:///configs/config.yaml).
   - No metric regresses by more than `max_regression_pct` (default: 10%) compared to the previous evaluation run registered in MLflow.
   - Fails the build if thresholds or regression limits are violated.
6. **Artifact Archival**: Uploads the detailed JSON report and summary as workflow artifacts (`ragops-eval-artifacts`).
7. **PR Comment**: Automatically posts a formatted Markdown summary table with status badges and score deltas directly onto pull requests.

---

## 9. FastAPI Service & Docker Deployment

The `src/api` module exposes a high-performance REST service for interacting with the versioned RAG assistant, streaming telemetry to persistent storage, and serving metrics in standard Prometheus exposition format.

### Endpoints
* **`POST /query`**: Accepts a question and version tag, performs vector similarity retrieval, builds a grounded prompt, calls Gemini (with "I don't know" safeguard), records telemetry metrics, and logs the query.
* **`GET /health`**: Health check reporting the connectivity status of ChromaDB and SQLite telemetry storage.
* **`GET /metrics`**: Serves Prometheus metrics for scraper consumption:
  * `ragops_requests_total`: Total count of requests by endpoint, HTTP status, and version ID.
  * `ragops_request_latency_seconds`: Latency distribution histogram (0.05s to 10.0s buckets).
  * `ragops_tokens_total`: Total prompt, completion, and aggregate token counts.
  * `ragops_retrieval_similarity_score`: Cosine similarity score distribution across retrieved chunks.
* **`GET /history`**: Returns recent queries with latency, token usage, and provenance from the SQLite store.

### Dual-Persistence Storage
Every query payload is saved to:
1. **SQLite (`data/logs/query_log.db`)**: Indexed relational database for fast querying, filtering, and latency analytics.
2. **Streaming JSONL (`data/logs/query_log.jsonl`)**: Append-only log stream for drift monitoring and offline evaluation pipelines.

### Running Locally
```bash
# Run server using Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or via Python module
python src/api/main.py
```

### Querying the API
```bash
# Ask a question to the assistant
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do path parameters work in FastAPI?", "data_version": "v1", "top_k": 4}'

# Check service health
curl http://localhost:8000/health

# Scrape Prometheus metrics
curl http://localhost:8000/metrics
```

### Running with Docker & Docker Compose
The application stack includes the FastAPI assistant, MLflow tracking server, Prometheus, and Grafana:

```bash
# Build and run the entire stack
docker compose up --build -d

# Check service status
docker compose ps

# Stop the stack
docker compose down
```

Service URLs:
* **FastAPI Application**: `http://localhost:8000` (Interactive docs at `/docs`)
* **MLflow Tracking Server**: `http://localhost:5000`
* **Prometheus Server**: `http://localhost:9090`
* **Grafana Dashboard**: `http://localhost:3000` (user: `admin`, pass: `admin`)

---

## 10. Continuous Monitoring & Drift Detection (`src/monitoring`, `scripts/simulate_drift.py`)

Production RAG systems suffer from three primary forms of drift: knowledge staleness, embedding distribution shift, and generation degradation. RAGOps implements automated monitoring for all three:

### (a) Async Production Query Sampling Evaluator (`src/monitoring/sampling_evaluator.py`)
* Asynchronously polls logged production traffic in `QueryStore` (SQLite/JSONL).
* Configurable sampling rate (default 10%) selects unseen queries for LLM scoring without adding user-facing latency.
* Scores queries with the **Gemini Judge** on:
  * **Faithfulness**: Is the answer factually grounded in the retrieved contexts?
  * **Answer Relevancy**: Does the answer directly address the question?
* Updates rolling Prometheus gauges with configurable window sizes:
  * `ragops_rolling_faithfulness`: Rolling average faithfulness.
  * `ragops_rolling_answer_relevancy`: Rolling average answer relevancy.
  * `ragops_sampled_queries_total`: Total count of sampled queries evaluated.

### (b) Embedding Drift Detection with Evidently (`src/monitoring/embedding_drift.py`)
* Compares high-dimensional embedding spaces between a reference window and current production windows for:
  * **Document Chunks**: Baseline index embeddings vs newly added/modified documents.
  * **Query Traffic**: Baseline golden queries vs incoming user traffic.
* Computes:
  * Top Principal Component Analysis (PCA) projections and Centroid Cosine Distance distribution shift.
  * Two-sample Kolmogorov-Smirnov test and Evidently Data Drift reports.
* Generates interactive reports in `data/monitoring/`:
  * `embedding_drift_doc.html` & `embedding_drift_doc_metrics.json`
  * `embedding_drift_query.html` & `embedding_drift_query_metrics.json`
* Exposes Prometheus gauges:
  * `ragops_embedding_drift_score`: Share of drifted dimensions / distance shift.
  * `ragops_embedding_drift_detected`: Boolean flag (1.0 = drift alert, 0.0 = normal).
  * `ragops_embedding_distance_shift`: Mean cosine distance shift.

### (c) Knowledge-Base Drift & Index Staleness (`src/monitoring/kb_drift.py`)
* Computes cryptographic SHA-256 digests across raw markdown documentation trees.
* Generates and persists versioned manifests (`data/monitoring/manifests/manifest_{version}.json`).
* Detects:
  * **New files** added to the documentation.
  * **Modified files** whose contents or APIs changed.
  * **Removed files** deleted from documentation.
* Marks the ChromaDB vector index as **STALE** (`is_stale=True`) when drift is detected.
* Exposes Prometheus gauges:
  * `ragops_kb_index_stale`: 1.0 if index is stale/drifted, 0.0 if fresh.
  * `ragops_kb_files_changed_total`: Count of added, modified, or removed files.

### Simulating & Demoing Drift (`scripts/simulate_drift.py`)
Run the end-to-end drift simulator to observe the system detect documentation updates, embedding shifts, and live query scoring:

```bash
# Run drift simulation across all 3 monitoring layers
python scripts/simulate_drift.py --num-queries 5 --mock-judge

# Or run with live Gemini Judge API (requires GEMINI_API_KEY)
python scripts/simulate_drift.py --num-queries 5
```

---

## 11. Running Tests & Linting
```bash
# Run all 36 unit tests across ingestion, chunking, embedding, vectorstore, rag, evaluation, api, and monitoring
python -m unittest discover -s tests

# Run ruff code linter
ruff check .
```

