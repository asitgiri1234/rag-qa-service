import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.limits import limiter
from app.config import Settings
from app.main import create_app
from app.models.schemas import DocumentStatus
from app.storage import db

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "meta.db",
        chroma_path=tmp_path / "chroma",
        metrics_path=tmp_path / "metrics.jsonl",
        max_upload_mb=1,
        warm_start=False,
    )


@pytest.fixture(autouse=True)
def disable_rate_limiting():
    """Off by default so unrelated tests are not throttled; the 429 test re-enables."""
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client(settings, monkeypatch):
    """App with ingestion stubbed out, for tests about the HTTP layer only."""
    queued: list[tuple] = []
    monkeypatch.setattr(
        "app.api.documents.worker.enqueue_ingestion",
        lambda document_id, path, filename: queued.append((document_id, path, filename)),
    )
    with TestClient(create_app(settings)) as test_client:
        test_client.queued = queued
        yield test_client


def upload(client, name, content, content_type="text/plain"):
    return client.post("/documents", files={"file": (name, content, content_type)})


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_upload_returns_202_and_queues_the_job(client, settings):
    response = upload(client, "notes.txt", b"Chunking matters a great deal.")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == DocumentStatus.PENDING.value
    document_id = body["document_id"]

    # The row exists before any processing happens, so status is pollable at once.
    stored = db.get_document(settings.sqlite_path, document_id)
    assert stored.status is DocumentStatus.PENDING
    assert stored.filename == "notes.txt"
    assert len(client.queued) == 1


def test_upload_rejects_unsupported_extension(client):
    response = upload(client, "archive.zip", b"PK\x03\x04", "application/zip")

    assert response.status_code == 415
    assert response.json()["error"]["type"] == "unsupported_media_type"


def test_upload_rejects_an_empty_file(client):
    response = upload(client, "empty.txt", b"")

    assert response.status_code == 400
    assert "empty" in response.json()["error"]["message"]


def test_upload_sniffs_pdf_magic_bytes(client):
    """A .pdf extension is not evidence the bytes are a PDF."""
    response = upload(client, "fake.pdf", b"this is not a pdf at all", "application/pdf")

    assert response.status_code == 400
    assert "magic bytes" in response.json()["error"]["message"]


def test_upload_accepts_a_real_pdf(client):
    response = upload(
        client, "rag_notes.pdf", (FIXTURES / "rag_notes.pdf").read_bytes(), "application/pdf"
    )

    assert response.status_code == 202


def test_upload_rejects_a_file_over_the_size_limit(client):
    oversized = b"x" * (2 * 1024 * 1024)  # limit is 1 MB in these settings

    response = upload(client, "big.txt", oversized)

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "payload_too_large"


def test_oversized_upload_leaves_no_file_behind(client, settings):
    upload(client, "big.txt", b"x" * (2 * 1024 * 1024))

    uploads = Path(settings.data_dir) / "uploads"
    assert not list(uploads.glob("*")) if uploads.exists() else True


def test_unknown_document_returns_a_structured_404(client):
    response = client.get("/documents/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"type": "not_found", "message": "no document does-not-exist"}
    }


def test_listing_is_paginated(client, settings):
    for index in range(3):
        upload(client, f"doc{index}.txt", b"content here")

    response = client.get("/documents", params={"limit": 2, "offset": 0})

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 3
    assert len(body["documents"]) == 2
    assert body["limit"] == 2


def test_invalid_pagination_is_rejected(client):
    response = client.get("/documents", params={"limit": 0})

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_rate_limit_returns_a_clean_json_429(settings, monkeypatch):
    """Deliberately exceed the upload limit and confirm the body is JSON, not HTML."""
    monkeypatch.setattr(
        "app.api.documents.worker.enqueue_ingestion", lambda *args, **kwargs: None
    )
    limiter.enabled = True
    limiter.reset()

    with TestClient(create_app(settings)) as client:
        statuses = [
            upload(client, f"n{index}.txt", b"content").status_code for index in range(12)
        ]

    assert 202 in statuses
    assert 429 in statuses, f"never hit the limit: {statuses}"

    limiter.reset()
    limiter.enabled = True
    with TestClient(create_app(settings)) as client:
        for index in range(11):
            response = upload(client, f"m{index}.txt", b"content")
            if response.status_code == 429:
                break
    body = response.json()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert "rate limit exceeded" in body["error"]["message"]
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.slow
def test_full_upload_to_completed_flow(settings, embedder, tokenizer, monkeypatch):
    """Upload, let the real worker run, and poll until the document completes."""
    import app.core.worker as worker_module
    from app.core.ingest import ingest_document
    from app.core.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(settings.chroma_path)

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

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "rag_notes.pdf",
                    (FIXTURES / "rag_notes.pdf").read_bytes(),
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]

        deadline = time.time() + 120
        while time.time() < deadline:
            body = client.get(f"/documents/{document_id}").json()
            if body["status"] in {"completed", "failed"}:
                break
            time.sleep(0.5)

        assert body["status"] == "completed", body
        assert body["chunk_count"] > 0
        assert body["completed_at"] is not None
        assert body["error"] is None

        deleted = client.delete(f"/documents/{document_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_chunks"] == body["chunk_count"]
        assert client.get(f"/documents/{document_id}").status_code == 404
