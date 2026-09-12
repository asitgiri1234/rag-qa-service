"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi.middleware import SlowAPIMiddleware

from app import __version__
from app.api import documents, metrics, query
from app.api.errors import register_error_handlers
from app.api.limits import limiter
from app.config import Settings, get_settings
from app.core import worker
from app.models.health import HealthResponse
from app.storage import db

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    db.init_db(settings.sqlite_path)
    # The queue does not survive a restart, so anything still marked processing has
    # no worker behind it; fail those rows rather than let them hang forever.
    orphaned = db.reset_stale_processing(settings.sqlite_path)
    if orphaned:
        logger.warning("failed %d document(s) orphaned by a previous shutdown", orphaned)

    # Built here from this app's settings so the API and the worker agree on one
    # collection even when settings are overridden (tests, the eval sweep).
    from app.core.vectorstore import ChromaVectorStore

    app.state.vector_store = ChromaVectorStore(settings.chroma_path)

    # Load the embedding model now rather than on the first job. Otherwise the
    # first upload silently pays ~30s of model load, and a broken model name is
    # not discovered until a document is already queued.
    if settings.warm_start:
        from app.core.embeddings import get_embedder

        get_embedder()

    worker.start_worker()
    logger.info(
        "ready: chunk_size=%d overlap=%d top_k=%d min_similarity=%.2f",
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
        settings.top_k,
        settings.min_similarity,
    )
    try:
        yield
    finally:
        worker.stop_worker()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    application = FastAPI(
        title="RAG QA Service",
        version=__version__,
        description=(
            "Retrieval-augmented question answering. Ingestion is asynchronous; "
            "queries are synchronous and return the similarity score of every "
            "source used."
        ),
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.limiter = limiter
    application.add_middleware(SlowAPIMiddleware)

    register_error_handlers(application)
    application.include_router(documents.router)
    application.include_router(query.router)
    application.include_router(metrics.router)

    @application.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return application


configure_logging()
app = create_app()
