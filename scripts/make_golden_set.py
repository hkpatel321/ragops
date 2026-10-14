"""
Generate golden evaluation dataset from v1 documentation chunks for RAGOps.
Produces 60-100 question/ground-truth-answer pairs with source chunk IDs as retrieval ground truth.
Includes a balanced distribution of factual, how-to, and version-specific questions.
Saves output to eval/golden_v1.jsonl.

Usage:
    python scripts/make_golden_set.py [--count 75] [--offline]
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.markdown_chunker import MarkdownChunker, TextChunk
from src.config import cfg
from src.ingestion.loader import load_documents_from_directory
from src.logger import logger

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


GEMINI_QNA_PROMPT = """You are an AI evaluation benchmark specialist designing a golden question-answering evaluation dataset for RAG pipelines.
Given the documentation chunk below from {source} (version 1), create ONE high-quality question and its corresponding authoritative ground-truth answer.

Requirements:
1. The question MUST be strictly and completely answerable using ONLY the provided chunk.
2. Question type to generate: '{question_type}'
   - 'factual': Inquire about concepts, definitions, components, return values, or architecture.
   - 'how_to': Inquire about practical implementation steps, code syntax, function signatures, or configuration.
   - 'version_specific': Frame the question with reference to v1 conventions, specific versioned APIs, or behavior in this release.
3. The ground-truth answer must be thorough, factual, directly citing terms/code from the chunk, without assuming outside knowledge.

Chunk Metadata:
- Source: {source} (v1)
- Section: {section}
- File: {filename}

Chunk Content:
\"\"\"
{chunk_text}
\"\"\"

Return your response in STRICT valid JSON format with no Markdown formatting:
{{
  "question": "<the generated question>",
  "ground_truth_answer": "<the factual, comprehensive ground-truth answer>",
  "question_type": "{question_type}"
}}"""


def call_gemini_for_qna(
    client: Any,
    chunk: TextChunk,
    question_type: str,
    model_name: str = "gemini-2.5-flash",
) -> Optional[Dict[str, str]]:
    """
    Use Gemini to generate a question/ground-truth answer pair for a chunk.
    """
    prompt = GEMINI_QNA_PROMPT.format(
        source=chunk.source.upper(),
        section=chunk.metadata.get("section", "General"),
        filename=chunk.metadata.get("filename", ""),
        chunk_text=chunk.text,
        question_type=question_type,
    )

    for attempt in range(3):
        try:
            config = types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw_text = (response.text or "").strip()
            # Clean possible markdown wrapping if any
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            if "question" in data and "ground_truth_answer" in data:
                return {
                    "question": data["question"].strip(),
                    "ground_truth_answer": data["ground_truth_answer"].strip(),
                    "question_type": question_type,
                }
        except Exception as exc:
            logger.warning(f"Gemini Q&A generation attempt {attempt + 1} failed: {exc}")
            time.sleep(1.0 * (attempt + 1))

    return None


def generate_synthetic_qna(chunk: TextChunk, question_type: str) -> Dict[str, str]:
    """
    High-quality heuristic question/answer generator for offline execution or when API key is unavailable.
    Synthesizes grounded Q&A pairs directly from section titles, code blocks, and key factual sentences.
    """
    text = chunk.text.strip()
    source = chunk.source
    section = chunk.metadata.get("section", "General")
    filename = chunk.metadata.get("filename", "")

    # Extract code blocks if any
    code_matches = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

    # 1. How-To Question
    if question_type == "how_to":
        if code_matches:
            question = f"In {source.title()} v1 ({section}), how can you implement this functionality in Python?"
            answer = f"According to {source.title()} v1 documentation on '{section}', you can use the following pattern:\n\n```python\n{code_matches[0].strip()[:400]}\n```\n\nKey context: {sentences[0] if sentences else text[:250]}"
        elif "first-steps" in filename or "tutorial" in filename:
            question = f"How do you configure and execute '{section}' according to the {source.title()} v1 guide?"
            answer = f"In the {source.title()} v1 guide for '{section}', the procedure is described as follows: {sentences[0] if sentences else text[:300]}"
        else:
            question = f"How does one use or configure '{section}' in {source.title()} v1?"
            answer = f"In {source.title()} v1, '{section}' is configured and used as follows: {' '.join(sentences[:2]) if len(sentences) >= 2 else text[:300]}"

    # 2. Version-Specific Question
    elif question_type == "version_specific":
        if source == "fastapi":
            question = f"In FastAPI v1 (release 0.100.0), what is documented regarding '{section}'?"
            answer = f"In FastAPI 0.100.0 documentation for '{section}': {' '.join(sentences[:2]) if len(sentences) >= 2 else text[:300]}"
        else:
            question = f"In MLflow v1 (release 2.10.0), what behavior or specification is defined for '{section}'?"
            answer = f"In MLflow 2.10.0 documentation for '{section}': {' '.join(sentences[:2]) if len(sentences) >= 2 else text[:300]}"

    # 3. Factual Question
    else:
        question_type = "factual"
        if sentences:
            lead = sentences[0]
            question = f"What is the primary function or purpose of '{section}' in {source.title()} v1?"
            answer = f"In {source.title()} v1, {lead} Furthermore: {' '.join(sentences[1:3]) if len(sentences) > 1 else ''}".strip()
        else:
            question = f"What does the {source.title()} v1 documentation explain regarding '{section}'?"
            answer = f"According to the {source.title()} v1 documentation for '{section}': {text[:350]}"

    return {
        "question": question,
        "ground_truth_answer": answer,
        "question_type": question_type,
    }


def make_golden_set(
    target_count: int = 75,
    offline_mode: bool = False,
    output_path: str = "eval/golden_v1.jsonl",
) -> List[Dict[str, Any]]:
    """
    Generate golden evaluation dataset from v1 documentation chunks.

    Args:
        target_count: Total number of Q&A pairs to generate (between 60 and 100).
        offline_mode: Force offline heuristic generator without making external API calls.
        output_path: Destination JSONL file path.

    Returns:
        List of golden set dictionary records.
    """
    logger.info(f"Generating golden evaluation dataset (target_count={target_count})...")

    # 1. Load and chunk v1 raw docs
    v1_docs = load_documents_from_directory("data/raw/v1", version="v1")
    if not v1_docs:
        raise FileNotFoundError("No v1 raw documents found in data/raw/v1. Run scripts/fetch_docs.py first.")

    chunker = MarkdownChunker(
        chunk_size=cfg.chunking.chunk_size,
        overlap=cfg.chunking.overlap,
    )
    all_chunks = chunker.chunk_documents(v1_docs)
    logger.info(f"Loaded {len(all_chunks)} candidate chunks from v1 documentation.")

    # 2. Filter informative chunks (> 150 chars, substantive content)
    substantive_chunks = [
        c for c in all_chunks if len(c.text.strip()) >= 150 and not c.text.strip().startswith("# Index")
    ]
    logger.info(f"Identified {len(substantive_chunks)} substantive chunks suitable for Q&A generation.")

    # Split evenly across sources
    fastapi_chunks = [c for c in substantive_chunks if c.source == "fastapi"]
    mlflow_chunks = [c for c in substantive_chunks if c.source == "mlflow"]

    random.seed(42)
    random.shuffle(fastapi_chunks)
    random.shuffle(mlflow_chunks)

    half_count = target_count // 2
    selected_chunks = (
        fastapi_chunks[:half_count] + mlflow_chunks[: target_count - half_count]
    )
    random.shuffle(selected_chunks)

    # 3. Check for Gemini client
    api_key = os.getenv("GEMINI_API_KEY") or cfg.llm.api_key
    use_gemini = (
        not offline_mode
        and HAS_GENAI
        and api_key
        and api_key != "your_gemini_api_key_here"
    )

    client = None
    if use_gemini:
        try:
            client = genai.Client(api_key=api_key)
            logger.info(f"Using Gemini API ({cfg.llm.generator.model_name}) for golden set generation.")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini Client: {e}. Falling back to offline generator.")
            use_gemini = False
    else:
        logger.info("Running in offline generation mode (high-quality grounded synthesis).")

    # 4. Generate Q&A pairs with balanced question types
    question_types = ["factual", "how_to", "version_specific"]
    golden_records: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(selected_chunks, start=1):
        q_type = question_types[(idx - 1) % len(question_types)]
        qna = None

        if use_gemini and client:
            qna = call_gemini_for_qna(
                client=client,
                chunk=chunk,
                question_type=q_type,
                model_name=cfg.llm.generator.model_name,
            )
            # Sleep slightly to respect free tier rate limits
            time.sleep(1.0)

        if not qna:
            qna = generate_synthetic_qna(chunk=chunk, question_type=q_type)

        record = {
            "id": f"q_v1_{idx:03d}",
            "question": qna["question"],
            "ground_truth_answer": qna["ground_truth_answer"],
            "ground_truth_chunk_id": chunk.chunk_id,
            "ground_truth_context": chunk.text,
            "question_type": qna["question_type"],
            "source": chunk.source,
            "version": "v1",
            "section": chunk.metadata.get("section", "General"),
            "file": chunk.metadata.get("filename", ""),
            "metadata": chunk.metadata,
        }
        golden_records.append(record)

        if idx % 15 == 0 or idx == len(selected_chunks):
            logger.info(f"Generated {idx}/{len(selected_chunks)} golden set records...")

    # 5. Save to eval/golden_v1.jsonl
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in golden_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(golden_records)} golden records to {out_path}.")
    return golden_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden test set from v1 docs")
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=75,
        help="Number of question-answer pairs to generate (60-100, default: 75)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline generation without calling Gemini API",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="eval/golden_v1.jsonl",
        help="Output path for the golden set JSONL file",
    )
    args = parser.parse_args()

    count = max(60, min(100, args.count))
    records = make_golden_set(
        target_count=count,
        offline_mode=args.offline,
        output_path=args.output,
    )

    type_counts = {}
    source_counts = {}
    for r in records:
        type_counts[r["question_type"]] = type_counts.get(r["question_type"], 0) + 1
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

    print("\n==========================================")
    print("Golden Evaluation Dataset Generated (v1)")
    print("==========================================")
    print(f"Total Pairs Generated : {len(records)}")
    print(f"Target Output Path    : {args.output}")
    print(f"Question Types        : {type_counts}")
    print(f"Sources               : {source_counts}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
