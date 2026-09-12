"""The query endpoint: retrieve, then generate.

Similarity scores are returned with every answer on purpose. Without them a bad
answer and a bad retrieval look identical from outside the service, and the only way
to tell them apart is to read the server logs.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from app.api.limits import QUERY_LIMIT, limiter
from app.config import Settings, get_settings
from app.core.generation import NO_CONTEXT_ANSWER, GenerationError, generate_answer
from app.core.retrieval import retrieve
from app.models.schemas import QueryRequest, QueryResponse, Source

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

#: How much of each chunk to echo back as evidence. Enough to judge the match
#: without turning every response into a copy of the corpus.
SNIPPET_CHARS = 400


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
@limiter.limit(QUERY_LIMIT)
def query(request: Request, payload: QueryRequest) -> QueryResponse:
    settings = _settings(request)
    top_k = payload.top_k or settings.top_k
    started = time.perf_counter()

    metrics_path = settings.metrics_path

    outcome = retrieve(
        payload.question,
        top_k,
        payload.document_ids,
        min_similarity=settings.min_similarity,
        store=getattr(request.app.state, "vector_store", None),
    )

    # Nothing cleared the floor: answer honestly instead of handing the model
    # context that retrieval already judged irrelevant.
    if not outcome.results:
        total_ms = (time.perf_counter() - started) * 1000
        _record(
            payload.question,
            top_k,
            outcome,
            generation_ms=0.0,
            total_ms=total_ms,
            answered=False,
            path=metrics_path,
        )
        return QueryResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            retrieval_ms=outcome.elapsed_ms,
            generation_ms=0.0,
            total_ms=total_ms,
        )

    try:
        generated = generate_answer(payload.question, outcome.results)
    except GenerationError as error:
        logger.error("generation failed: %s", error)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"answer generation is unavailable: {error}",
        ) from error

    total_ms = (time.perf_counter() - started) * 1000
    _record(
        payload.question,
        top_k,
        outcome,
        generation_ms=generated.elapsed_ms,
        total_ms=total_ms,
        answered=True,
        path=metrics_path,
    )

    return QueryResponse(
        answer=generated.answer,
        sources=[
            Source(
                filename=result.filename,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                similarity_score=round(result.similarity, 4),
                text=result.text[:SNIPPET_CHARS],
            )
            for result in outcome.results
        ],
        retrieval_ms=round(outcome.elapsed_ms, 2),
        generation_ms=round(generated.elapsed_ms, 2),
        total_ms=round(total_ms, 2),
    )


def _record(question, top_k, outcome, *, generation_ms, total_ms, answered, path=None) -> None:
    """Hand the query to the metrics recorder; never let it break a response."""
    try:
        from app.core import metrics
    except ImportError:
        return
    try:
        metrics.record_query(
            question=question,
            top_k=top_k,
            outcome=outcome,
            generation_ms=generation_ms,
            total_ms=total_ms,
            answered=answered,
            path=path,
        )
    except Exception:
        logger.exception("failed to record query metrics")
