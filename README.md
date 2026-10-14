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

## 5. Running Tests
```bash
python -m unittest discover -s tests
```
