import pytest

from app.models.schemas import DocumentStatus
from app.storage import db


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "meta.db"
    db.init_db(path)
    return path


def insert(database, document_id="doc-1", filename="a.pdf", size=1234):
    return db.insert_document(
        database,
        document_id=document_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=size,
    )


def test_init_db_is_idempotent(tmp_path):
    path = tmp_path / "meta.db"
    db.init_db(path)
    db.init_db(path)
    assert db.count_documents(path) == 0


def test_new_document_starts_pending(database):
    created = insert(database)

    assert created.status is DocumentStatus.PENDING
    assert created.chunk_count == 0
    assert created.completed_at is None

    stored = db.get_document(database, "doc-1")
    assert stored == created


def test_unknown_document_is_none(database):
    assert db.get_document(database, "nope") is None


def test_lifecycle_pending_to_completed(database):
    insert(database)

    db.mark_processing(database, "doc-1")
    assert db.get_document(database, "doc-1").status is DocumentStatus.PROCESSING

    db.mark_completed(database, "doc-1", chunk_count=42)

    stored = db.get_document(database, "doc-1")
    assert stored.status is DocumentStatus.COMPLETED
    assert stored.chunk_count == 42
    assert stored.completed_at is not None
    assert stored.error is None


def test_failure_records_the_error_and_is_terminal(database):
    insert(database)
    db.mark_processing(database, "doc-1")

    db.mark_failed(database, "doc-1", "pypdf could not read the file")

    stored = db.get_document(database, "doc-1")
    assert stored.status is DocumentStatus.FAILED
    assert stored.error == "pypdf could not read the file"
    assert stored.completed_at is not None


def test_long_errors_are_truncated(database):
    insert(database)

    db.mark_failed(database, "doc-1", "x" * 5000)

    assert len(db.get_document(database, "doc-1").error) == 2000


def test_completion_clears_a_previous_error(database):
    insert(database)
    db.mark_failed(database, "doc-1", "transient")

    db.mark_completed(database, "doc-1", chunk_count=3)

    assert db.get_document(database, "doc-1").error is None


def test_listing_is_paginated_and_newest_first(database):
    for index in range(5):
        insert(database, document_id=f"doc-{index}", filename=f"{index}.pdf")

    first_page = db.list_documents(database, limit=2, offset=0)
    second_page = db.list_documents(database, limit=2, offset=2)

    assert len(first_page) == 2
    assert len(second_page) == 2
    assert {d.id for d in first_page} & {d.id for d in second_page} == set()
    assert db.count_documents(database) == 5


def test_delete_reports_whether_a_row_went(database):
    insert(database)

    assert db.delete_document(database, "doc-1") is True
    assert db.delete_document(database, "doc-1") is False
    assert db.count_documents(database) == 0


def test_stale_processing_rows_are_failed_at_startup(database):
    """A restart must never leave a job stuck at processing."""
    insert(database, document_id="stuck")
    insert(database, document_id="done")
    db.mark_processing(database, "stuck")
    db.mark_completed(database, "done", chunk_count=1)

    reset = db.reset_stale_processing(database)

    assert reset == 1
    stuck = db.get_document(database, "stuck")
    assert stuck.status is DocumentStatus.FAILED
    assert "restarted" in stuck.error
    assert db.get_document(database, "done").status is DocumentStatus.COMPLETED
