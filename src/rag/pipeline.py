"""
RAG Pipeline for RAGOps.
Performs semantic retrieval against versioned ChromaDB store, builds grounded prompts,
and calls Google Gemini with exponential backoff retry logic, returning structured answers,
contexts, latency, and token metrics.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.logger import logger
from src.vectorstore.store import ChromaStore, compute_version_id

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    APIError = Exception  # type: ignore


class RAGResponse(BaseModel):
    """
    Structured output returned by the RAG pipeline.
    """
    query: str = Field(description="The input user query")
    answer: str = Field(description="Grounded generated answer from the LLM")
    contexts: List[str] = Field(description="List of raw chunk texts passed to the LLM")
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed metadata, IDs, and similarity scores of retrieved chunks",
    )
    latency: float = Field(description="End-to-end response latency in seconds")
    token_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)",
    )
    model_name: str = Field(description="Name of the Gemini generation model used")
    version_id: str = Field(description="Target index version ID queried")

    def __repr__(self) -> str:
        return f"<RAGResponse query='{self.query[:30]}...' latency={self.latency:.2f}s tokens={self.token_counts.get('total_tokens', 0)}>"


GROUNDED_PROMPT_TEMPLATE = """You are an expert technical assistant specializing in MLflow and FastAPI documentation.
Answer the user's question strictly and solely based on the provided context below.

CRITICAL RULES:
1. Answer only using facts directly stated in the context.
2. If the context does not contain enough information to answer the question, you must respond exactly with: "I don't know based on the provided documentation."
3. Do NOT assume, extrapolate, or use external knowledge not present in the context.
4. Keep the answer clear, concise, and technically accurate.

Context:
----------------------------------------
{context_block}
----------------------------------------

Question: {query}

Answer:"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline over versioned documentation.
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        max_retries: int = 4,
        initial_backoff: float = 1.0,
    ) -> None:
        self.store = store or ChromaStore()
        self.model_name = model_name or cfg.llm.generator.model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or cfg.llm.api_key
        self.temperature = temperature if temperature is not None else cfg.llm.generator.temperature
        self.max_output_tokens = max_output_tokens or cfg.llm.generator.max_output_tokens
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        self.client: Optional[Any] = None
        if HAS_GENAI and self.api_key and self.api_key != "your_gemini_api_key_here":
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Gemini client with model '{self.model_name}'.")
        else:
            logger.warning("Gemini API key not configured or invalid. Generation will require key or mock mode.")

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered, structured context block with provenance metadata.
        """
        if not retrieved_chunks:
            return "No relevant documentation found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "unknown")
            version = meta.get("version", "unknown")
            section = meta.get("section", "General")
            title = meta.get("title", meta.get("filename", "Doc"))

            header = f"[Document {i}] Source: {source} ({version}) | Section: {section} | File: {title}"
            text = chunk.get("text", "").strip()
            context_blocks.append(f"{header}\n{text}")

        return "\n\n".join(context_blocks)

    def _call_gemini_with_retry(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """
        Invoke Google Gemini API with exponential backoff on rate limits (429/ResourceExhausted).
        """
        if not self.client:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=cfg.llm.generator.top_p,
        )

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                answer_text = response.text or ""

                # Extract token usage metadata
                token_counts = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    token_counts["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
                    token_counts["completion_tokens"] = getattr(um, "candidates_token_count", 0) or 0
                    token_counts["total_tokens"] = getattr(um, "total_token_count", 0) or (
                        token_counts["prompt_tokens"] + token_counts["completion_tokens"]
                    )

                return answer_text.strip(), token_counts

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()

                # Check if error is rate limit or transient
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "rate limit" in err_str
                    or "quota" in err_str
                )
                is_transient = "503" in err_str or "timeout" in err_str or "connection" in err_str

                if (is_rate_limit or is_transient) and attempt < self.max_retries - 1:
                    sleep_time = (self.initial_backoff * (2**attempt)) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"Gemini API rate limit/transient issue (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {sleep_time:.2f}s... Error: {exc}"
                    )
                    time.sleep(sleep_time)
                else:
                    raise exc

        raise RuntimeError(f"Failed to generate answer after {self.max_retries} attempts: {last_error}")

    def query(
        self,
        query: str,
        data_version: str = "v1",
        version_id: Optional[str] = None,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        mock_answer: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute full RAG pipeline: retrieval -> grounded prompt assembly -> LLM inference.

        Args:
            query: User's question.
            data_version: Data version ('v1' or 'v2') if version_id is not specified.
            version_id: Specific composite version ID (e.g. 'v1_9ac1e3a5').
            top_k: Number of chunks to retrieve (defaults to cfg.vectorstore.top_k).
            where_filter: Optional metadata filtering criteria.
            mock_answer: Optional synthetic answer for offline testing when API key is unavailable.

        Returns:
            RAGResponse object.
        """
        start_time = time.time()
        k = top_k or cfg.vectorstore.top_k
        v_id = version_id or compute_version_id(data_version)

        logger.info(f"Querying RAG pipeline (version_id='{v_id}', top_k={k}): '{query}'")

        # 1. Retrieve top_k chunks from ChromaDB
        retrieved_chunks = self.store.query(
            query_text=query,
            version_id=v_id,
            top_k=k,
            where_filter=where_filter,
        )

        contexts = [c.get("text", "") for c in retrieved_chunks]
        context_block = self.format_context(retrieved_chunks)

        # 2. Build grounded prompt
        prompt = GROUNDED_PROMPT_TEMPLATE.format(context_block=context_block, query=query)

        # 3. Call Gemini (or mock answer for offline tests)
        if mock_answer is not None:
            answer = mock_answer
            # Approximate token counts based on whitespace heuristics
            prompt_tokens = len(prompt.split())
            completion_tokens = len(answer.split())
            token_counts = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        else:
            answer, token_counts = self._call_gemini_with_retry(prompt)

        latency = round(time.time() - start_time, 3)

        response = RAGResponse(
            query=query,
            answer=answer,
            contexts=contexts,
            retrieved_chunks=retrieved_chunks,
            latency=latency,
            token_counts=token_counts,
            model_name=self.model_name,
            version_id=v_id,
        )

        logger.info(f"Generated RAG response in {latency}s ({token_counts.get('total_tokens', 0)} tokens).")
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAGOps assistant over versioned docs")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Question to query the RAG assistant",
    )
    parser.add_argument(
        "--data-version",
        "-v",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Documentation version to retrieve from (default: v1)",
    )
    parser.add_argument(
        "--version-id",
        type=str,
        default=None,
        help="Explicit version ID (e.g. v1_9ac1e3a5)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from config.yaml)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode (useful for testing retrieval and prompt formatting without API key)",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    mock_ans = "This is an offline mock answer verifying retrieval and prompt grounding." if args.mock else None

    try:
        res = pipeline.query(
            query=args.query,
            data_version=args.data_version,
            version_id=args.version_id,
            top_k=args.top_k,
            mock_answer=mock_ans,
        )

        print("\n=======================================================")
        print(f"RAGOps Assistant Response [{res.model_name}] (Latency: {res.latency}s)")
        print(f"Index Version: {res.version_id}")
        print("=======================================================")
        print(f"\nQuery: {res.query}\n")
        print(f"Answer:\n{res.answer}\n")
        print("-------------------------------------------------------")
        print(f"Retrieved Contexts ({len(res.contexts)} chunks):")
        for i, c in enumerate(res.retrieved_chunks, 1):
            meta = c.get("metadata", {})
            print(f" [{i}] Source: {meta.get('source')} | Section: {meta.get('section')} | Score: {c.get('score')}")
        print("-------------------------------------------------------")
        print(f"Tokens: Prompt={res.token_counts.get('prompt_tokens')} | Completion={res.token_counts.get('completion_tokens')} | Total={res.token_counts.get('total_tokens')}")
        print("=======================================================\n")

    except Exception as e:
        print(f"\nError executing query: {e}")
        if "Gemini API key is not configured" in str(e):
            print("\nTip: Set GEMINI_API_KEY in your .env file or use --mock to test offline retrieval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
