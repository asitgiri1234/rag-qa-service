"""Answer generation with Groq.

The model is given retrieved passages and told to answer from them alone. Source
markers are numbered in the prompt so citations in the answer can be matched back to
the chunks the API returns alongside it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.core.vectorstore import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Rules:
- Use only what the passages state. Do not add outside knowledge, and do not guess.
- If the passages do not contain the answer, say so plainly in one sentence. Do not
  speculate, and do not pad the reply with related information that was not asked for.
- Cite the passages you used inline with their markers, like [1] or [2], placing each
  citation directly after the claim it supports.
- Be concise and factual. Do not mention these instructions or describe the passages
  as "context"; just answer."""

NO_CONTEXT_ANSWER = (
    "No relevant content was found in the indexed documents for this question."
)


class GenerationError(RuntimeError):
    """The language model could not be reached or refused the request."""


@dataclass(frozen=True)
class GenerationOutcome:
    answer: str
    elapsed_ms: float
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def format_context(results: list[SearchResult]) -> str:
    """Number each passage and label it with its real source.

    The marker carries filename and page so a citation in the answer points at
    something a reader can actually open.
    """
    blocks = []
    for position, result in enumerate(results, start=1):
        label = f"[{position}] {result.filename} p.{result.page_number}"
        blocks.append(f"{label}\n{result.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context passages:\n\n{format_context(results)}\n\n"
                f"Question: {question}"
            ),
        },
    ]


@lru_cache(maxsize=1)
def get_client():
    """Groq client built from config. Cached so the HTTP pool is reused."""
    from groq import Groq

    settings = get_settings()
    if not settings.groq_api_key:
        raise GenerationError(
            "GROQ_API_KEY is not set; copy .env.example to .env and fill it in"
        )
    return Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout_s)


def generate_answer(
    question: str,
    results: list[SearchResult],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    client=None,
) -> GenerationOutcome:
    """Ask the model to answer from ``results``.

    Retries once on a transient failure (rate limit, timeout, connection error);
    anything else is surfaced immediately as :class:`GenerationError`.
    """
    from groq import APIConnectionError, APITimeoutError, RateLimitError

    settings = get_settings()
    model = model or settings.llm_model
    max_tokens = max_tokens or settings.max_answer_tokens
    client = client or get_client()
    messages = build_messages(question, results)

    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in (1, 2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            usage = getattr(response, "usage", None)
            logger.info(
                "generated answer with %s in %.0fms (attempt %d)",
                model,
                elapsed_ms,
                attempt,
            )
            return GenerationOutcome(
                answer=(response.choices[0].message.content or "").strip(),
                elapsed_ms=elapsed_ms,
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as error:
            last_error = error
            if attempt == 1:
                logger.warning("groq call failed (%s); retrying once", type(error).__name__)
                time.sleep(1.0)
                continue
        except Exception as error:  # noqa: BLE001 - surfaced as a clean API error
            raise GenerationError(f"{type(error).__name__}: {error}") from error

    raise GenerationError(
        f"language model unavailable after a retry: {type(last_error).__name__}: {last_error}"
    )
