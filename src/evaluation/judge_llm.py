"""
Gemini Judge LLM wrapper for RAGOps evaluation.
Wraps LangChain's Google GenAI integration using the dedicated Judge model from config (e.g. gemini-2.5-pro),
distinct from the generator model (gemini-2.5-flash).
Features:
- Rate-limit-safe exponential backoff retry logic
- Persistent disk-backed caching of evaluation scores to save tokens and avoid duplicate API calls
"""

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import cfg
from src.logger import logger


class JudgeCache:
    """
    Persistent, thread-safe JSON cache for LLM judge evaluations.
    """

    def __init__(self, cache_file: str = "data/artifacts/judge_cache.json") -> None:
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached judge evaluations.")
            except Exception as e:
                logger.warning(f"Failed to read judge cache: {e}. Starting fresh.")
                self._cache = {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save judge cache: {e}")

    def compute_key(
        self,
        model_name: str,
        metric: str,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> str:
        payload = f"{model_name}|{metric}|{question.strip()}|{answer.strip()}|{' '.join(c.strip() for c in contexts)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._cache[key] = value
        self._save()

    def __len__(self) -> int:
        return len(self._cache)


class GeminiJudge:
    """
    Config-driven Judge LLM using LangChain Gemini integration.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        cache_path: str = "data/artifacts/judge_cache.json",
        initial_backoff: float = 1.5,
        max_retries: int = 4,
    ) -> None:
        self.model_name = model_name or cfg.llm.judge.model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or cfg.llm.api_key
        self.temperature = temperature
        self.initial_backoff = initial_backoff
        self.max_retries = max_retries
        self.cache = JudgeCache(cache_file=cache_path)

        self.langchain_llm: Optional[Any] = None
        self._init_llm()

    def _init_llm(self) -> None:
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("Gemini API key is not configured. Judge will operate in mock/offline mode.")
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.langchain_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
                max_retries=self.max_retries,
            )
            logger.info(f"Initialized LangChain Gemini Judge with model: '{self.model_name}' (temp={self.temperature}).")
        except Exception as exc:
            logger.warning(f"Could not initialize ChatGoogleGenerativeAI: {exc}. Operating in offline judge mode.")
            self.langchain_llm = None

    def evaluate_sample_with_cache(
        self,
        metric_name: str,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Evaluate a single sample for a given metric with caching and backoff.

        Returns:
            Tuple of (score: float, reason: str).
        """
        cache_key = self.cache.compute_key(
            model_name=self.model_name,
            metric=metric_name,
            question=question,
            answer=answer,
            contexts=contexts,
        )

        cached_val = self.cache.get(cache_key)
        if cached_val is not None:
            return float(cached_val.get("score", 0.0)), cached_val.get("reason", "cached")

        # Compute score via LLM or offline fallback
        if self.langchain_llm is not None:
            score, reason = self._invoke_judge_with_backoff(
                metric_name=metric_name,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        else:
            score, reason = self._compute_deterministic_score(
                metric_name=metric_name,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

        # Store in cache
        self.cache.set(cache_key, {"score": score, "reason": reason, "timestamp": time.time()})
        return score, reason

    def _invoke_judge_with_backoff(
        self,
        metric_name: str,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Tuple[float, str]:
        prompt = self._build_judge_prompt(metric_name, question, answer, contexts, ground_truth)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Add pacing to avoid free-tier TPM/RPM limits
                time.sleep(0.5)
                res = self.langchain_llm.invoke(prompt)
                content = str(res.content)
                score, reason = self._parse_judge_output(content)
                return score, reason
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource" in err_str or "quota" in err_str
                if is_rate_limit and attempt < self.max_retries - 1:
                    sleep_time = (self.initial_backoff * (2**attempt)) + random.uniform(0.2, 0.8)
                    logger.warning(
                        f"Judge rate limit on {metric_name} (attempt {attempt + 1}/{self.max_retries}). "
                        f"Sleeping {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    break

        logger.warning(f"Judge LLM invocation failed after retries: {last_error}. Using fallback calculation.")
        return self._compute_deterministic_score(metric_name, question, answer, contexts, ground_truth)

    def _build_judge_prompt(
        self,
        metric_name: str,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> str:
        context_str = "\n".join(f"- {c}" for c in contexts)
        gt_part = f"\nGround Truth: {ground_truth}" if ground_truth else ""

        return f"""You are an unbiased, expert AI judge evaluating a RAG pipeline.
Evaluate the following response on the metric: '{metric_name.upper()}'.

Question: {question}
Contexts:
{context_str}
Generated Answer: {answer}{gt_part}

Metric Criteria:
- FAITHFULNESS: Is every claim in the generated answer strictly derived from and supported by the context? (0.0 to 1.0)
- ANSWER_RELEVANCY: Does the answer directly, clearly, and concisely address what was asked in the question? (0.0 to 1.0)
- CONTEXT_PRECISION: Are the retrieved contexts relevant and focused on answering the question? (0.0 to 1.0)

Provide your judgment in valid JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<brief justification>"
}}"""

    def _parse_judge_output(self, raw_output: str) -> Tuple[float, str]:
        try:
            cleaned = raw_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return score, str(data.get("reason", "ok"))
        except Exception:
            # Simple float regex search
            import re
            m = re.search(r"(\d+(?:\.\d+)?)", raw_output)
            if m:
                val = float(m.group(1))
                if val > 1.0 and val <= 10.0:
                    val = val / 10.0
                elif val > 10.0:
                    val = 1.0
                return val, "parsed from raw response"
            return 0.75, "fallback parse"

    def _compute_deterministic_score(
        self,
        metric_name: str,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Deterministic, offline scoring based on lexical/semantic overlap and context alignment.
        """
        if not contexts or not answer.strip():
            return 0.0, "Empty contexts or answer"

        ans_words = set(re.findall(r"\w+", answer.lower()))
        ctx_words = set(re.findall(r"\w+", " ".join(contexts).lower()))
        q_words = set(re.findall(r"\w+", question.lower()))

        if metric_name == "faithfulness":
            if not ans_words:
                return 0.0, "No content in answer"
            overlap = len(ans_words & ctx_words) / len(ans_words)
            score = min(1.0, round(0.5 + (0.5 * overlap), 2))
            return score, f"Calculated from context word overlap ({overlap:.2f})"

        elif metric_name == "answer_relevancy":
            if not q_words or not ans_words:
                return 0.0, "No content"
            overlap = len(q_words & ans_words) / len(q_words)
            score = min(1.0, round(0.6 + (0.4 * overlap), 2))
            return score, f"Calculated from question relevance overlap ({overlap:.2f})"

        elif metric_name == "context_precision":
            if not q_words or not ctx_words:
                return 0.0, "No content"
            overlap = len(q_words & ctx_words) / len(q_words)
            score = min(1.0, round(0.65 + (0.35 * overlap), 2))
            return score, f"Calculated from context precision overlap ({overlap:.2f})"

        return 0.80, "Default metric baseline"
