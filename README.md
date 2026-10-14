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

### Running Tests
```bash
python -m unittest discover -s tests
```
