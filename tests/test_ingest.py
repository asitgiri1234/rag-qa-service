import pytest

from app.config import Settings
from app.core.ingest import ingest_document
from app.core.vectorstore import ChromaVectorStore
from app.models.schemas import DocumentStatus
from app.storage import db

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.slow


@pytest.fixture
def settings(tmp_path):
    """Isolated settings so a test never touches the real data/ directory."""
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "meta.db",
        chroma_path=tmp_path / "chroma",
        metrics_path=tmp_path / "metrics.jsonl",
        chunk_size_tokens=180,
        chunk_overlap_tokens=40,
    )


@pytest.fixture
def store(settings):
    return ChromaVectorStore(settings.chroma_path)


def register(settings, document_id, path):
    db.init_db(settings.sqlite_path)
    db.insert_document(
        settings.sqlite_path,
        document_id=document_id,
        filename=path.name,
        content_type="application/octet-stream",
        size_bytes=path.stat().st_size,
    )


def test_ingests_a_real_pdf_end_to_end(settings, store, embedder, tokenizer):
    path = FIXTURES / "rag_notes.pdf"
    register(settings, "pdf-1", path)

    metrics = ingest_document(
        "pdf-1",
        path,
        settings=settings,
        tokenizer=tokenizer,
        embedder=embedder,
        store=store,
    )

    assert metrics.page_count == 3
    assert metrics.chunk_count > 0
    assert store.count() == metrics.chunk_count

    document = db.get_document(settings.sqlite_path, "pdf-1")
    assert document.status is DocumentStatus.COMPLETED
    assert document.chunk_count == metrics.chunk_count
    assert document.error is None
    assert document.completed_at is not None

    # Every phase must be timed for the metrics stage.
    assert metrics.parse_ms > 0
    assert metrics.embed_ms > 0
    assert metrics.total_ms >= metrics.embed_ms


def test_ingests_a_real_text_file(settings, store, embedder, tokenizer, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "Retrieval quality depends on chunk size.\n\n"
        "The embedding model truncates at 256 word-piece tokens, so a chunk larger "
        "than that ceiling loses its tail before the vector is ever produced.\n\n"
        "Overlap protects facts that straddle a chunk boundary.",
        encoding="utf-8",
    )
    register(settings, "txt-1", path)

    metrics = ingest_document(
        "txt-1", path, settings=settings, tokenizer=tokenizer, embedder=embedder, store=store
    )

    assert metrics.chunk_count > 0
    assert db.get_document(settings.sqlite_path, "txt-1").status is DocumentStatus.COMPLETED


def test_page_numbers_survive_into_stored_metadata(settings, store, embedder, tokenizer):
    path = FIXTURES / "rag_notes.pdf"
    register(settings, "pdf-2", path)

    ingest_document(
        "pdf-2", path, settings=settings, tokenizer=tokenizer, embedder=embedder, store=store
    )

    # The 256-token ceiling is discussed on page 1 of the fixture.
    results = store.search(embedder.embed_query("How many tokens before truncation?"), top_k=3)
    assert results
    assert {result.page_number for result in results} <= {1, 2, 3}
    assert all(result.filename == "rag_notes.pdf" for result in results)


def test_a_failure_is_terminal_and_never_leaves_status_processing(
    settings, store, embedder, tokenizer, tmp_path
):
    """A crashed job must record the error, not hang at processing forever."""
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  \n", encoding="utf-8")
    register(settings, "bad-1", path)

    with pytest.raises(Exception):
        ingest_document(
            "bad-1", path, settings=settings, tokenizer=tokenizer, embedder=embedder, store=store
        )

    document = db.get_document(settings.sqlite_path, "bad-1")
    assert document.status is DocumentStatus.FAILED
    assert document.status is not DocumentStatus.PROCESSING
    assert "no extractable text" in document.error
    assert document.completed_at is not None


def test_unsupported_format_fails_the_job_cleanly(
    settings, store, embedder, tokenizer, tmp_path
):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04not really a zip")
    register(settings, "bad-2", path)

    with pytest.raises(Exception):
        ingest_document(
            "bad-2", path, settings=settings, tokenizer=tokenizer, embedder=embedder, store=store
        )

    document = db.get_document(settings.sqlite_path, "bad-2")
    assert document.status is DocumentStatus.FAILED
    assert "UnsupportedFormatError" in document.error


def test_worker_thread_drains_the_queue(settings, store, embedder, tokenizer, monkeypatch):
    """The worker runs jobs off the request thread and records their outcome."""
    from app.core import worker

    path = FIXTURES / "rag_notes.pdf"
    register(settings, "queued-1", path)

    # Point the worker's pipeline at the isolated fixtures.
    import app.core.worker as worker_module

    def run(document_id, file_path, filename=None):
        return ingest_document(
            document_id,
            file_path,
            filename=filename,
            settings=settings,
            tokenizer=tokenizer,
            embedder=embedder,
            store=store,
        )

    monkeypatch.setattr(worker_module, "ingest_document", run)

    worker.start_worker()
    try:
        worker.enqueue_ingestion("queued-1", path, path.name)
        worker.drain()
    finally:
        worker.stop_worker()

    document = db.get_document(settings.sqlite_path, "queued-1")
    assert document.status is DocumentStatus.COMPLETED
    assert document.chunk_count > 0


def test_worker_survives_a_failing_job(settings, store, embedder, tokenizer, monkeypatch, tmp_path):
    """One bad document must not kill the worker for every later document."""
    from app.core import worker
    import app.core.worker as worker_module

    bad = tmp_path / "bad.txt"
    bad.write_text("  \n", encoding="utf-8")
    good = FIXTURES / "rag_notes.pdf"
    register(settings, "bad-3", bad)
    db.insert_document(
        settings.sqlite_path,
        document_id="good-1",
        filename=good.name,
        content_type="application/pdf",
        size_bytes=good.stat().st_size,
    )

    def run(document_id, file_path, filename=None):
        return ingest_document(
            document_id,
            file_path,
            filename=filename,
            settings=settings,
            tokenizer=tokenizer,
            embedder=embedder,
            store=store,
        )

    monkeypatch.setattr(worker_module, "ingest_document", run)

    worker.start_worker()
    try:
        worker.enqueue_ingestion("bad-3", bad, bad.name)
        worker.enqueue_ingestion("good-1", good, good.name)
        worker.drain()
    finally:
        worker.stop_worker()

    assert db.get_document(settings.sqlite_path, "bad-3").status is DocumentStatus.FAILED
    assert db.get_document(settings.sqlite_path, "good-1").status is DocumentStatus.COMPLETED
