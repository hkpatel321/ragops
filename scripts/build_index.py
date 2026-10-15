"""
Build and version the ChromaDB index for RAGOps.
Chunks versioned documentation, generates dense embeddings, persists vectors,
and logs all parameters, metrics, and index artifacts to MLflow.

Usage:
    python scripts/build_index.py --data-version v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from src.chunking.markdown_chunker import MarkdownChunker
from src.config import cfg
from src.ingestion.loader import load_documents_from_directory
from src.logger import logger
from src.vectorstore.store import ChromaStore, compute_config_hash, compute_version_id


def build_index(data_version: str = "v1") -> Dict[str, Any]:
    """
    Build, index, and log ChromaDB collection for a specified data version.

    Args:
        data_version: Version of the raw data to index ('v1' or 'v2').

    Returns:
        Dictionary summarizing index build metadata and metrics.
    """
    start_time = time.time()
    logger.info(f"Starting index build for data version '{data_version}'...")

    # 1. Load documents
    raw_dir = Path("data/raw") / data_version
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    documents = load_documents_from_directory(raw_dir, version=data_version)
    if not documents:
        raise ValueError(f"No documents found in {raw_dir}")

    doc_count = len(documents)
    total_raw_chars = sum(len(d.content) for d in documents)
    logger.info(f"Loaded {doc_count} documents ({total_raw_chars} total chars).")

    # 2. Chunk documents
    chunker = MarkdownChunker(
        chunk_size=cfg.chunking.chunk_size,
        overlap=cfg.chunking.overlap,
    )
    chunks = chunker.chunk_documents(documents)
    chunk_count = len(chunks)
    avg_chunk_len = sum(len(c.text) for c in chunks) / max(chunk_count, 1)

    # 3. Compute version ID
    cfg_hash = compute_config_hash()
    version_id = compute_version_id(data_version)
    logger.info(f"Generated Version ID: {version_id} (Config Hash: {cfg_hash})")

    # 4. Embed & Index into ChromaDB
    store = ChromaStore(
        persist_directory=cfg.vectorstore.persist_directory,
        collection_name=cfg.vectorstore.collection_name,
    )
    indexed_count = store.add_chunks(chunks=chunks, version_id=version_id)

    duration = round(time.time() - start_time, 2)
    logger.info(f"Index build completed in {duration}s. Indexed {indexed_count} vectors.")

    # 5. Prepare index build metadata artifact
    build_summary = {
        "data_version": data_version,
        "version_id": version_id,
        "config_hash": cfg_hash,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "avg_chunk_chars": round(avg_chunk_len, 1),
        "chunk_size": cfg.chunking.chunk_size,
        "overlap": cfg.chunking.overlap,
        "embedding_model": cfg.embedding.model_name,
        "distance_metric": cfg.vectorstore.distance_metric,
        "build_duration_sec": duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save summary artifact
    artifact_dir = Path("data/artifacts") / version_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "index_summary.json"
    summary_path.write_text(json.dumps(build_summary, indent=2), encoding="utf-8")

    # 6. Log parameters, metrics, and index artifact to MLflow
    try:
        # Set local sqlite tracking uri if remote server is not running
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        run_name = f"index-build-{version_id}"
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params(
                {
                    "data_version": data_version,
                    "version_id": version_id,
                    "config_hash": cfg_hash,
                    "chunk_size": cfg.chunking.chunk_size,
                    "overlap": cfg.chunking.overlap,
                    "embedding_model": cfg.embedding.model_name,
                    "distance_metric": cfg.vectorstore.distance_metric,
                    "collection_name": f"{cfg.vectorstore.collection_name}_{version_id}",
                }
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "doc_count": doc_count,
                    "chunk_count": chunk_count,
                    "avg_chunk_chars": avg_chunk_len,
                    "build_duration_sec": duration,
                }
            )

            # Log tags
            mlflow.set_tags(
                {
                    "stage": "index_build",
                    "version_id": version_id,
                    "data_version": data_version,
                }
            )

            # Log summary artifact
            mlflow.log_artifact(str(summary_path), artifact_path="metadata")

            # Log ChromaDB store artifact directory
            chroma_dir = Path(cfg.vectorstore.persist_directory)
            if chroma_dir.exists():
                mlflow.log_artifacts(str(chroma_dir), artifact_path="chroma_store")

            logger.info(f"Successfully logged index build run '{run_name}' to MLflow ({mlflow_uri}).")

    except Exception as exc:
        logger.warning(f"MLflow logging encountered notice: {exc}. Local index build succeeded.")

    return build_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and version ChromaDB index for RAGOps")
    parser.add_argument(
        "--data-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Raw data version to index (v1 or v2)",
    )
    args = parser.parse_args()

    summary = build_index(data_version=args.data_version)
    print("\n==========================================")
    print(f"Index Build Summary for {args.data_version.upper()}")
    print("==========================================")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
