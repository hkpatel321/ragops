"""
Embedding Drift Detection for RAGOps using Evidently.
Monitors semantic vector spaces for:
- Query embedding drift (reference baseline queries vs current incoming queries)
- Document chunk embedding drift (reference knowledge base vs current knowledge base)
Exposes metrics to Prometheus and generates Evidently HTML & JSON drift reports.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from prometheus_client import Gauge
from scipy import stats
from sklearn.decomposition import PCA

from src.logger import logger

# Prometheus Gauges for Embedding Drift
EMBEDDING_DRIFT_SCORE = Gauge(
    name="ragops_embedding_drift_score",
    documentation="Embedding drift metric score (share of drifted components or distance shift)",
    labelnames=["embedding_type"],
)

EMBEDDING_DRIFT_DETECTED = Gauge(
    name="ragops_embedding_drift_detected",
    documentation="Indicator if embedding drift was detected (1.0 = drift, 0.0 = stable)",
    labelnames=["embedding_type"],
)

EMBEDDING_DISTANCE_SHIFT = Gauge(
    name="ragops_embedding_distance_shift",
    documentation="Mean cosine distance shift between reference and current embedding distributions",
    labelnames=["embedding_type"],
)


@dataclass
class EmbeddingDriftResult:
    """
    Structured summary of embedding drift detection results.
    """
    embedding_type: str  # "query" or "doc"
    drift_detected: bool
    drift_score: float
    drift_share: float
    mean_cosine_shift: float
    p_value: float
    reference_count: int
    current_count: int
    report_html_path: Optional[str] = None
    metrics_json_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbeddingDriftDetector:
    """
    Analyzes embedding distributions across reference and current windows.
    Generates Evidently reports and updates Prometheus gauges.
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        pca_components: int = 5,
        reports_dir: str = "data/monitoring",
    ) -> None:
        self.drift_threshold = drift_threshold
        self.pca_components = pca_components
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors along axis 1."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _prepare_tabular_data(
        self,
        ref_embeddings: np.ndarray,
        curr_embeddings: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Transform high-dimensional embeddings into low-dimensional representation:
        - Centroid cosine distance feature
        - Top PCA projection features
        Returns (ref_df, curr_df, mean_cosine_shift).
        """
        ref_norm = self._normalize(ref_embeddings)
        curr_norm = self._normalize(curr_embeddings)

        # 1. Reference centroid
        ref_centroid = np.mean(ref_norm, axis=0, keepdims=True)
        ref_centroid = ref_centroid / max(np.linalg.norm(ref_centroid), 1e-9)

        # Centroid cosine distances: 1 - cosine_sim
        ref_dist_to_centroid = 1.0 - np.dot(ref_norm, ref_centroid.T).flatten()
        curr_dist_to_centroid = 1.0 - np.dot(curr_norm, ref_centroid.T).flatten()

        mean_cosine_shift = float(abs(np.mean(curr_dist_to_centroid) - np.mean(ref_dist_to_centroid)))

        # 2. PCA projections on reference fitted space
        n_samples_min = min(len(ref_embeddings), len(curr_embeddings))
        n_comp = min(self.pca_components, n_samples_min - 1) if n_samples_min > 1 else 1

        pca = PCA(n_components=n_comp, random_state=42)
        ref_pca = pca.fit_transform(ref_norm)
        curr_pca = pca.transform(curr_norm)

        ref_dict: Dict[str, Any] = {"centroid_distance": ref_dist_to_centroid}
        curr_dict: Dict[str, Any] = {"centroid_distance": curr_dist_to_centroid}

        for i in range(n_comp):
            ref_dict[f"pca_dim_{i}"] = ref_pca[:, i]
            curr_dict[f"pca_dim_{i}"] = curr_pca[:, i]

        return pd.DataFrame(ref_dict), pd.DataFrame(curr_dict), mean_cosine_shift

    def detect_drift(
        self,
        reference_embeddings: Union[List[List[float]], np.ndarray],
        current_embeddings: Union[List[List[float]], np.ndarray],
        embedding_type: str = "query",
        report_filename: Optional[str] = None,
    ) -> EmbeddingDriftResult:
        """
        Compare reference vs current embeddings using Evidently data drift metrics.
        """
        ref_arr = np.array(reference_embeddings, dtype=float)
        curr_arr = np.array(current_embeddings, dtype=float)

        if ref_arr.ndim != 2 or curr_arr.ndim != 2:
            raise ValueError("Embeddings must be 2-dimensional (samples, dimensions).")

        if len(ref_arr) == 0 or len(curr_arr) == 0:
            raise ValueError("Reference and current embeddings must not be empty.")

        ref_df, curr_df, mean_cosine_shift = self._prepare_tabular_data(ref_arr, curr_arr)

        drift_detected = False
        drift_share = 0.0
        drift_score = 0.0
        min_p_value = 1.0
        html_path = None
        json_path = None

        # Attempt to run Evidently Report
        evidently_success = False
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(reference_data=ref_df, current_data=curr_df)

            base_name = report_filename or f"embedding_drift_{embedding_type}"
            html_file = self.reports_dir / f"{base_name}.html"
            json_file = self.reports_dir / f"{base_name}_metrics.json"

            snapshot.save_html(str(html_file))
            snapshot.save_json(str(json_file))

            html_path = str(html_file)
            json_path = str(json_file)

            snapshot_dict = snapshot.dict()
            metrics_data = snapshot_dict.get("metrics", [])
            for m in metrics_data:
                val = m.get("value")
                if isinstance(val, dict) and "share" in val:
                    drift_share = float(val.get("share", 0.0))
                    drift_detected = (drift_share >= self.drift_threshold) or (mean_cosine_shift >= self.drift_threshold)
                    drift_score = max(drift_share, mean_cosine_shift)
                    break
                elif "DriftedColumnsCount" in m.get("metric_name", "") and isinstance(val, dict):
                    drift_share = float(val.get("share", 0.0))
                    drift_detected = (drift_share >= self.drift_threshold) or (mean_cosine_shift >= self.drift_threshold)
                    drift_score = max(drift_share, mean_cosine_shift)
                    break

            evidently_success = True
            logger.info(
                f"Evidently report generated for [{embedding_type}]: "
                f"drift_share={drift_share:.3f}, drift_detected={drift_detected}"
            )

        except Exception as e:
            logger.warning(f"Evidently report generation skipped or fallback used: {e}")

        # Fallback / statistical verification using 2-sample Kolmogorov-Smirnov test
        if not evidently_success:
            drifted_cols = 0
            p_values = []
            for col in ref_df.columns:
                res = stats.ks_2samp(ref_df[col], curr_df[col])
                p_values.append(res.pvalue)
                if res.pvalue < 0.05:
                    drifted_cols += 1

            drift_share = float(drifted_cols / len(ref_df.columns))
            min_p_value = float(min(p_values)) if p_values else 1.0
            drift_detected = (drift_share >= self.drift_threshold) or (mean_cosine_shift >= self.drift_threshold)
            drift_score = max(drift_share, mean_cosine_shift)

            # Save fallback JSON metrics
            json_file = self.reports_dir / f"{report_filename or f'embedding_drift_{embedding_type}'}_metrics.json"
            fallback_metrics = {
                "embedding_type": embedding_type,
                "dataset_drift": drift_detected,
                "drift_share": drift_share,
                "drift_score": drift_score,
                "mean_cosine_shift": mean_cosine_shift,
                "min_p_value": min_p_value,
                "reference_samples": len(ref_arr),
                "current_samples": len(curr_arr),
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(fallback_metrics, f, indent=2)
            json_path = str(json_file)

        # Update Prometheus Gauges
        EMBEDDING_DRIFT_SCORE.labels(embedding_type=embedding_type).set(round(drift_score, 4))
        EMBEDDING_DRIFT_DETECTED.labels(embedding_type=embedding_type).set(1.0 if drift_detected else 0.0)
        EMBEDDING_DISTANCE_SHIFT.labels(embedding_type=embedding_type).set(round(mean_cosine_shift, 4))

        logger.info(
            f"Embedding drift evaluated [{embedding_type}]: "
            f"Detected={drift_detected} | Drift Score={drift_score:.4f} | "
            f"Cosine Shift={mean_cosine_shift:.4f}"
        )

        return EmbeddingDriftResult(
            embedding_type=embedding_type,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_share=drift_share,
            mean_cosine_shift=mean_cosine_shift,
            p_value=min_p_value,
            reference_count=len(ref_arr),
            current_count=len(curr_arr),
            report_html_path=html_path,
            metrics_json_path=json_path,
        )
