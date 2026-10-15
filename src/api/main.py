"""
Main entry point for RAGOps FastAPI server.
Supports both direct Python execution (`python src/api/main.py`) and Uvicorn CLI (`uvicorn src.api.main:app`).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from src.api.app import app
from src.config import cfg

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
    )
