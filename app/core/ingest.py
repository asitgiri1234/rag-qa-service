"""The ingestion pipeline: parse -> chunk -> embed -> store.

Runs on the worker thread, never on a request. Every phase is timed separately so
the metrics stage has real per-phase numbers rather than one opaque total, and any
failure is recorded as a terminal ``failed`` status: a job must never be left at
``processing`` with nothing working on it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.core.chunking import chunk_pages
from app.core.embeddings import Embedder, get_embedder
from app.core.parsers import parse_document
from app.core.tokenization import Tokenizer, load_tokenizer
from app.core.vectorstore import VectorStore, get_vector_store
from app.storage import db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionMetrics:
    """Per-phase timings for one document, in milliseconds."""

    document_id: str
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int
    parse_ms: float
    chunk_ms: float
    embed_ms: float
    store_ms: float
    total_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


class EmptyDocumentError(ValueError):
    """The file parsed cleanly but yielded no text (e.g. a scanned PDF)."""


def ingest_document(
    document_id: str,
    file_path: str | Path,
    *,
    filename: str | None = None,
    settings: Settings | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IngestionMetrics:
    """Run the full pipeline for one document and record the outcome in SQLite.

    Dependencies are injectable so the evaluation sweep can point the same pipeline
    at a throwaway collection and a different chunk configuration.
    """
    settings = settings or get_settings()
    tokenizer = tokenizer or load_tokenizer(settings.embedding_model)
    embedder = embedder or get_embedder()
    store = store or get_vector_store()

    path = Path(file_path)
    display_name = filename or path.name
    started = time.perf_counter()

    db.mark_processing(settings.sqlite_path, document_id)
    logger.info("ingest %s started (%s)", document_id, display_name)

    try:
        phase = time.perf_counter()
        pages = parse_document(path)
        parse_ms = (time.perf_counter() - phase) * 1000
        if not pages:
            raise EmptyDocumentError(
                "no extractable text found; if this is a scanned PDF it needs OCR"
            )

        phase = time.perf_counter()
        chunks = chunk_pages(
            pages,
            tokenizer=tokenizer,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        chunk_ms = (time.perf_counter() - phase) * 1000
        if not chunks:
            raise EmptyDocumentError("document produced no chunks")

        phase = time.perf_counter()
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        embed_ms = (time.perf_counter() - phase) * 1000

        phase = time.perf_counter()
        store.add_chunks(document_id, display_name, chunks, embeddings)
        store_ms = (time.perf_counter() - phase) * 1000

        total_ms = (time.perf_counter() - started) * 1000
        db.mark_completed(settings.sqlite_path, document_id, chunk_count=len(chunks))

        metrics = IngestionMetrics(
            document_id=document_id,
            filename=display_name,
            size_bytes=path.stat().st_size if path.exists() else 0,
            page_count=len(pages),
            chunk_count=len(chunks),
            parse_ms=parse_ms,
            chunk_ms=chunk_ms,
            embed_ms=embed_ms,
            store_ms=store_ms,
            total_ms=total_ms,
        )
        logger.info(
            "ingest %s completed: %d pages -> %d chunks in %.0fms "
            "(parse %.0f, chunk %.0f, embed %.0f, store %.0f)",
            document_id,
            len(pages),
            len(chunks),
            total_ms,
            parse_ms,
            chunk_ms,
            embed_ms,
            store_ms,
        )
        _record(metrics)
        return metrics

    except Exception as error:
        # Any failure is terminal and must be visible through the status endpoint.
        message = f"{type(error).__name__}: {error}"
        logger.exception("ingest %s failed: %s", document_id, message)
        try:
            db.mark_failed(settings.sqlite_path, document_id, message)
        except Exception:
            logger.exception("could not record failure for %s", document_id)
        raise


def _record(metrics: IngestionMetrics) -> None:
    """Hand metrics to the recorder, if one is installed (see Stage 6)."""
    try:
        from app.core import metrics as metrics_module
    except ImportError:
        return
    try:
        metrics_module.record_ingestion(metrics)
    except Exception:
        logger.exception("failed to record ingestion metrics for %s", metrics.document_id)
