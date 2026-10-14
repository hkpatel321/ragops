"""
Configuration loader and schema definitions for RAGOps.
Loads YAML configuration with environment variable overrides and validates with Pydantic.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()


class ProjectConfig(BaseModel):
    name: str = "ragops"
    version: str = "1.0.0"
    description: str = "RAGOps Pipeline"


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    overlap: int = 64
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", " ", ""])


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    persist_directory: str = "data/chroma_db"
    collection_name: str = "ragops_docs"
    top_k: int = 4
    distance_metric: str = "cosine"


class ModelParameters(BaseModel):
    model_name: str
    temperature: float = 0.2
    max_output_tokens: int = 1024
    top_p: float = 0.95


class LLMConfig(BaseModel):
    generator: ModelParameters = Field(
        default_factory=lambda: ModelParameters(
            model_name="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=1024,
            top_p=0.95,
        )
    )
    judge: ModelParameters = Field(
        default_factory=lambda: ModelParameters(
            model_name="gemini-2.5-pro",
            temperature=0.0,
            max_output_tokens=2048,
            top_p=0.90,
        )
    )
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))


class ThresholdsConfig(BaseModel):
    faithfulness: float = 0.80
    answer_relevance: float = 0.85
    context_precision: float = 0.75
    context_recall: float = 0.75
    max_latency_seconds: float = 3.0


class MonitoringConfig(BaseModel):
    reference_data_path: str = "data/test_sets/reference_queries.json"
    drift_report_path: str = "data/monitoring/drift_report.html"
    drift_metrics_path: str = "data/monitoring/drift_metrics.json"
    embedding_drift_threshold: float = 0.15
    response_length_drift_threshold: float = 0.20


class MLflowConfig(BaseModel):
    tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "RAGOps-Evaluation")
    )


class APIConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", 8000)))
    reload: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """
    Load YAML configuration file and populate into strongly typed AppConfig model.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Ensure environment variables take priority for sensitive keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        config.llm.api_key = gemini_key

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        config.mlflow.tracking_uri = mlflow_uri

    return config


# Singleton cached configuration instance
cfg: AppConfig = load_config()
