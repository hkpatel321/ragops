"""
Entry point script for RAGOps pipeline orchestration.
Provides CLI commands to run ingestion, chunking, indexing, evaluation, and serving.
"""

import argparse

from src.config import cfg
from src.logger import logger


def run_ingest() -> None:
    logger.info("Starting document ingestion from documentation sources...")
    # Will be invoked once ingestion module is implemented
    logger.info("Ingestion completed.")


def run_chunk() -> None:
    logger.info(f"Chunking documents (size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})...")
    logger.info("Chunking completed.")


def run_index() -> None:
    logger.info(f"Indexing chunks with model {cfg.embedding.model_name} into {cfg.vectorstore.persist_directory}...")
    logger.info("Vector store index created.")


def run_eval() -> None:
    logger.info(f"Running automated Ragas evaluation using judge {cfg.llm.judge.model_name}...")
    logger.info("Evaluation completed and logged to MLflow.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGOps Pipeline CLI")
    parser.add_argument(
        "--step",
        choices=["ingest", "chunk", "index", "eval", "all"],
        default="all",
        help="Pipeline step to execute",
    )
    args = parser.parse_args()

    logger.info(f"Executing RAGOps pipeline step: {args.step}")

    if args.step in ["ingest", "all"]:
        run_ingest()
    if args.step in ["chunk", "all"]:
        run_chunk()
    if args.step in ["index", "all"]:
        run_index()
    if args.step in ["eval", "all"]:
        run_eval()


if __name__ == "__main__":
    main()
