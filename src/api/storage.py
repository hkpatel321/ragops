"""
Persistent Query Telemetry Logging for RAGOps API.
Maintains dual persistence:
1. SQLite Database: Queryable indexed relational store (data/logs/query_log.db).
2. JSONL Log: Append-only streaming logs for downstream batch ETL and drift monitoring (data/logs/query_log.jsonl).
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import logger


class QueryStore:
    """
    Thread-safe query logger writing to both SQLite and JSONL files.
    """

    def __init__(
        self,
        db_path: str = "data/logs/query_log.db",
        jsonl_path: str = "data/logs/query_log.jsonl",
    ) -> None:
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self._lock = threading.Lock()

        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queries (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        query TEXT NOT NULL,
                        contexts TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        latency REAL NOT NULL,
                        prompt_tokens INTEGER DEFAULT 0,
                        completion_tokens INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        index_version TEXT NOT NULL,
                        model_name TEXT,
                        top_k INTEGER,
                        retrieved_chunk_ids TEXT
                    );
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_version ON queries(index_version);")
                conn.commit()
            finally:
                conn.close()

    def log_query(
        self,
        request_id: str,
        query: str,
        contexts: List[str],
        answer: str,
        latency: float,
        token_counts: Dict[str, int],
        index_version: str,
        model_name: str,
        top_k: int,
        retrieved_chunk_ids: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Record a query to both SQLite database and JSONL file.
        """
        ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
        chunk_ids = retrieved_chunk_ids or []
        prompt_tokens = token_counts.get("prompt_tokens", 0)
        completion_tokens = token_counts.get("completion_tokens", 0)
        total_tokens = token_counts.get("total_tokens", prompt_tokens + completion_tokens)
        contexts_json = json.dumps(contexts, ensure_ascii=False)
        chunk_ids_json = json.dumps(chunk_ids, ensure_ascii=False)

        record_dict = {
            "id": request_id,
            "timestamp": ts,
            "query": query,
            "contexts": contexts,
            "answer": answer,
            "latency": latency,
            "token_counts": token_counts,
            "index_version": index_version,
            "model_name": model_name,
            "top_k": top_k,
            "retrieved_chunk_ids": chunk_ids,
        }

        with self._lock:
            # 1. Write to SQLite
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO queries (
                        id, timestamp, query, contexts, answer, latency,
                        prompt_tokens, completion_tokens, total_tokens,
                        index_version, model_name, top_k, retrieved_chunk_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        ts,
                        query,
                        contexts_json,
                        answer,
                        latency,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        index_version,
                        model_name,
                        top_k,
                        chunk_ids_json,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to write query record to SQLite ({self.db_path}): {e}")

            # 2. Append to JSONL
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to append query record to JSONL ({self.jsonl_path}): {e}")

    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent queries from SQLite store.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, query, contexts, answer, latency,
                       prompt_tokens, completion_tokens, total_tokens,
                       index_version, model_name, top_k, retrieved_chunk_ids
                FROM queries
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append(
                    {
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "query": r["query"],
                        "contexts": json.loads(r["contexts"]) if r["contexts"] else [],
                        "answer": r["answer"],
                        "latency": r["latency"],
                        "token_counts": {
                            "prompt_tokens": r["prompt_tokens"],
                            "completion_tokens": r["completion_tokens"],
                            "total_tokens": r["total_tokens"],
                        },
                        "index_version": r["index_version"],
                        "model_name": r["model_name"],
                        "top_k": r["top_k"],
                        "retrieved_chunk_ids": json.loads(r["retrieved_chunk_ids"])
                        if r["retrieved_chunk_ids"]
                        else [],
                    }
                )
            conn.close()
            return results

    def get_total_count(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM queries")
            count = cursor.fetchone()[0]
            conn.close()
            return int(count)
