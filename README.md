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
Benchmark the Gemini Judge against 30 curated, manually annotated samples covering high-quality, hallucinated, off-topic, and out-of-context edge cases:
```bash
python scripts/judge_calibration.py --dataset eval/calibration_samples_30.jsonl --tolerance 0.20
```

---

## 8. Running Tests
```bash
python -m unittest discover -s tests
```
