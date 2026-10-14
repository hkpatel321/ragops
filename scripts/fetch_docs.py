"""
Fetch versioned markdown documentation for MLflow and FastAPI from GitHub.

Downloads release-tagged docs:
- Version 1 (older): FastAPI 0.100.0, MLflow v2.10.0
- Version 2 (newer): FastAPI 0.115.0, MLflow v2.20.0

Saves documents under:
- data/raw/v1/fastapi/ and data/raw/v1/mlflow/
- data/raw/v2/fastapi/ and data/raw/v2/mlflow/
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry-enabled session
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "RAGOps-DocFetcher/1.0"})

# Versions mapping
VERSIONS = {
    "v1": {
        "fastapi": "0.100.0",
        "mlflow": "v2.10.0",
    },
    "v2": {
        "fastapi": "0.115.0",
        "mlflow": "v2.20.0",
    },
}

# FastAPI target markdown files
FASTAPI_DOCS: List[str] = [
    "docs/en/docs/index.md",
    "docs/en/docs/tutorial/first-steps.md",
    "docs/en/docs/tutorial/path-params.md",
    "docs/en/docs/tutorial/query-params.md",
    "docs/en/docs/tutorial/body.md",
    "docs/en/docs/tutorial/response-model.md",
    "docs/en/docs/tutorial/handling-errors.md",
    "docs/en/docs/tutorial/dependencies/index.md",
    "docs/en/docs/tutorial/cors.md",
    "docs/en/docs/deployment/docker.md",
    "docs/en/docs/deployment/versions.md",
    "docs/en/docs/async.md",
]

# MLflow target files (markdown and core rst converted to md)
MLFLOW_DOCS: List[str] = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/source/llms/index.rst",
    "docs/source/llms/rag/index.rst",
    "docs/source/tracking.rst",
    "docs/source/model-registry.rst",
    "docs/source/projects.rst",
]


def rst_to_markdown_simple(rst_text: str) -> str:
    """
    Lightweight conversion of reStructuredText headers and directives into clean Markdown.
    """
    lines = rst_text.splitlines()
    md_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for title underlines (===, ---, ~~~)
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if len(next_line) >= 3 and len(next_line) == len(line.strip()):
                if next_line.startswith("="):
                    md_lines.append(f"# {line.strip()}")
                    i += 2
                    continue
                elif next_line.startswith("-"):
                    md_lines.append(f"## {line.strip()}")
                    i += 2
                    continue
                elif next_line.startswith("~") or next_line.startswith("^"):
                    md_lines.append(f"### {line.strip()}")
                    i += 2
                    continue

        # Strip rst directives like .. note::, .. code-block:: python
        code_match = re.match(r"^\s*\.\.\s+code-block::\s*(\w+)?", line)
        if code_match:
            lang = code_match.group(1) or ""
            md_lines.append(f"```{lang}")
            i += 1
            continue

        if re.match(r"^\s*\.\.\s+[a-zA-Z0-9_-]+::", line):
            i += 1
            continue

        # Convert rst inline backticks `:code:\`foo\`` -> `foo`
        cleaned = re.sub(r":[a-z]+:`([^`]+)`", r"`\1`", line)
        md_lines.append(cleaned)
        i += 1

    return "\n".join(md_lines)


def download_github_file(repo: str, tag: str, file_path: str) -> Optional[str]:
    """
    Download raw content of a specific file from GitHub.
    """
    url = f"https://raw.githubusercontent.com/{repo}/{tag}/{file_path}"
    try:
        resp = session.get(url, timeout=(5.0, 15.0))
        if resp.status_code == 200:
            return resp.text
        # Fallback if README.md vs README.rst
        if file_path == "README.md":
            fallback_url = f"https://raw.githubusercontent.com/{repo}/{tag}/README.rst"
            fallback_resp = session.get(fallback_url, timeout=(5.0, 15.0))
            if fallback_resp.status_code == 200:
                return rst_to_markdown_simple(fallback_resp.text)
        return None
    except Exception as exc:
        print(f"Error fetching {url}: {exc}")
        return None


def fetch_docs() -> None:
    """
    Download all defined docs for v1 and v2 for both FastAPI and MLflow.
    """
    base_dir = Path("data/raw")

    for version_key, repo_tags in VERSIONS.items():
        print(f"\n==========================================")
        print(f"Fetching docs for {version_key.upper()}...")
        print(f"==========================================")

        # 1. FastAPI
        fastapi_tag = repo_tags["fastapi"]
        fastapi_dir = base_dir / version_key / "fastapi"
        fastapi_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[FastAPI] Version {fastapi_tag} -> {fastapi_dir}")

        for doc_path in FASTAPI_DOCS:
            out_name = doc_path.replace("docs/en/docs/", "").replace("/", "_")
            out_file = fastapi_dir / out_name
            content = download_github_file("fastapi/fastapi", fastapi_tag, doc_path)
            if content:
                out_file.write_text(content, encoding="utf-8")
                print(f"  [OK] Saved {doc_path} -> {out_name} ({len(content)} chars)")
            else:
                print(f"  [FAILED] Failed to download {doc_path}")

        # 2. MLflow
        mlflow_tag = repo_tags["mlflow"]
        mlflow_dir = base_dir / version_key / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[MLflow] Version {mlflow_tag} -> {mlflow_dir}")

        for doc_path in MLFLOW_DOCS:
            out_name = doc_path.replace("docs/source/", "").replace("/", "_")
            if out_name.endswith(".rst"):
                out_name = out_name[:-4] + ".md"

            out_file = mlflow_dir / out_name
            content = download_github_file("mlflow/mlflow", mlflow_tag, doc_path)
            if content:
                if doc_path.endswith(".rst"):
                    content = rst_to_markdown_simple(content)
                out_file.write_text(content, encoding="utf-8")
                print(f"  [OK] Saved {doc_path} -> {out_name} ({len(content)} chars)")
            else:
                print(f"  [FAILED] Failed to download {doc_path}")

    print("\nDocumentation fetch completed successfully.")


if __name__ == "__main__":
    fetch_docs()
