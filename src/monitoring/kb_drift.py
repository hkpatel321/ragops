"""
Knowledge Base Drift & Staleness Detection for RAGOps.
Detects file additions, modifications, and deletions in raw document sets:
- Computes SHA256 cryptographic hashes for all markdown files
- Compares current documentation state against indexed baseline manifest
- Flags vector index as STALE when drift is detected
- Exposes staleness state and file change counts to Prometheus
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prometheus_client import Gauge

from src.logger import logger

# Prometheus Gauges for Knowledge Base Staleness
KB_INDEX_STALE = Gauge(
    name="ragops_kb_index_stale",
    documentation="Knowledge base index staleness indicator (1.0 = stale/drifted, 0.0 = fresh)",
    labelnames=["data_version"],
)

KB_FILES_CHANGED = Gauge(
    name="ragops_kb_files_changed_total",
    documentation="Count of added, modified, or removed documentation files",
    labelnames=["change_type", "data_version"],
)


@dataclass
class KBDriftReport:
    """
    Structured report describing differences between knowledge base state and indexed baseline.
    """
    data_version: str
    is_stale: bool
    drift_detected: bool
    total_reference_files: int
    total_current_files: int
    new_files: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def total_changes(self) -> int:
        return len(self.new_files) + len(self.changed_files) + len(self.removed_files)


class KBDriftDetector:
    """
    Detects documentation drift by hashing document trees and comparing against manifests.
    """

    def __init__(self, manifests_dir: str = "data/artifacts/manifests") -> None:
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hex digest of file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_manifest(self, docs_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Scan a documentation directory and compute SHA256 hashes for all markdown files.
        Returns mapping of relative_path -> sha256_hash.
        """
        base_path = Path(docs_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Documentation directory does not exist: {docs_dir}")

        manifest: Dict[str, str] = {}
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".md", ".markdown"]:
                rel_path = file_path.relative_to(base_path).as_posix()
                manifest[rel_path] = self.compute_file_hash(file_path)

        return dict(sorted(manifest.items()))

    def save_manifest(self, manifest: Dict[str, str], version_tag: str) -> Path:
        """Save a generated manifest to disk."""
        out_file = self.manifests_dir / f"manifest_{version_tag}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved manifest for version '{version_tag}' ({len(manifest)} files) to {out_file}.")
        return out_file

    def load_manifest(self, version_tag: str) -> Optional[Dict[str, str]]:
        """Load manifest for a given version tag if it exists."""
        in_file = self.manifests_dir / f"manifest_{version_tag}.json"
        if in_file.is_file():
            try:
                with open(in_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest from {in_file}: {e}")
        return None

    def detect_drift(
        self,
        current_dir: Union[str, Path],
        data_version: str = "v1",
        reference_manifest: Optional[Dict[str, str]] = None,
    ) -> KBDriftReport:
        """
        Compare current document state against reference manifest.
        If reference_manifest is not provided, loads from disk or generates baseline.
        """
        current_manifest = self.build_manifest(current_dir)

        if reference_manifest is None:
            reference_manifest = self.load_manifest(data_version)

        if reference_manifest is None:
            # First time seeing this version: save manifest as reference baseline
            logger.info(f"No existing manifest found for version '{data_version}'. Creating baseline manifest...")
            self.save_manifest(current_manifest, version_tag=data_version)
            reference_manifest = current_manifest

        curr_keys = set(current_manifest.keys())
        ref_keys = set(reference_manifest.keys())

        new_files = sorted(curr_keys - ref_keys)
        removed_files = sorted(ref_keys - curr_keys)
        changed_files = sorted(f for f in (curr_keys & ref_keys) if current_manifest[f] != reference_manifest[f])
        unchanged_files = sorted(f for f in (curr_keys & ref_keys) if current_manifest[f] == reference_manifest[f])

        is_stale = bool(new_files or changed_files or removed_files)

        # Update Prometheus Gauges
        KB_INDEX_STALE.labels(data_version=data_version).set(1.0 if is_stale else 0.0)
        KB_FILES_CHANGED.labels(change_type="new", data_version=data_version).set(len(new_files))
        KB_FILES_CHANGED.labels(change_type="changed", data_version=data_version).set(len(changed_files))
        KB_FILES_CHANGED.labels(change_type="removed", data_version=data_version).set(len(removed_files))

        report = KBDriftReport(
            data_version=data_version,
            is_stale=is_stale,
            drift_detected=is_stale,
            total_reference_files=len(reference_manifest),
            total_current_files=len(current_manifest),
            new_files=new_files,
            changed_files=changed_files,
            removed_files=removed_files,
            unchanged_files=unchanged_files,
        )

        if is_stale:
            logger.warning(
                f"Knowledge Base Drift Detected for [{data_version}]! "
                f"New: {len(new_files)}, Changed: {len(changed_files)}, Removed: {len(removed_files)}. "
                f"Index marked as STALE."
            )
        else:
            logger.info(f"Knowledge Base [{data_version}] is FRESH. No file drift detected.")

        return report
