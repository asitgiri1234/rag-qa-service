"""Query-side retrieval.

The question is embedded with the *same* embedder used at ingestion. Encoding
documents with one model and questions with another puts the two sets of vectors in
unrelated spaces; nothing raises, scores simply collapse toward noise. That is why
there is one embedder in the process and it is reached through one accessor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.embeddings import Embedder, get_embedder
from app.core.vectorstore import SearchResult, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalOutcome:
    """What retrieval returned, plus the numbers needed to judge it."""

    results: list[SearchResult]
    elapsed_ms: float
    candidates: int
    dropped_below_floor: int
    #: The configured floor, kept distinct from min_similarity below, which is a
    #: property of the results rather than of the configuration.
    similarity_floor: float

    @property
    def top_similarity(self) -> float | None:
        return self.results[0].similarity if self.results else None

    @property
    def min_similarity(self) -> float | None:
        """Weakest score among the returned chunks."""
        return min((r.similarity for r in self.results), default=None)

    @property
    def mean_similarity(self) -> float | None:
        if not self.results:
            return None
        return sum(result.similarity for result in self.results) / len(self.results)


def retrieve(
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
    *,
    min_similarity: float = 0.0,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> RetrievalOutcome:
    """Embed the question, search, and drop anything below the similarity floor.

    Returns an empty result list when nothing clears the floor. The caller must not
    fall through to the language model in that case: answering from context that was
    already judged irrelevant produces a confident wrong answer.
    """
    embedder = embedder or get_embedder()
    store = store or get_vector_store()

    started = time.perf_counter()
    query_vector = embedder.embed_query(query)
    candidates = store.search(query_vector, top_k, document_ids)
    kept = [result for result in candidates if result.similarity >= min_similarity]
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Constraint: every retrieval logs its latency and the similarity of each hit.
    logger.info(
        "retrieval for %r: %d kept / %d candidates in %.1fms (floor=%.2f); scores=[%s]",
        query[:80],
        len(kept),
        len(candidates),
        elapsed_ms,
        min_similarity,
        ", ".join(f"{result.similarity:.4f}" for result in candidates),
    )
    if candidates and not kept:
        logger.warning(
            "no chunk cleared the %.2f floor for %r; best was %.4f -- refusing to answer",
            min_similarity,
            query[:80],
            candidates[0].similarity,
        )

    return RetrievalOutcome(
        results=kept,
        elapsed_ms=elapsed_ms,
        candidates=len(candidates),
        dropped_below_floor=len(candidates) - len(kept),
        similarity_floor=min_similarity,
    )
