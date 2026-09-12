"""SQLite metadata layer -- stdlib ``sqlite3``, no ORM.

Chroma holds the vectors; this holds the job state that makes asynchronous ingestion
observable from the API. A connection is opened per operation rather than shared,
because the worker thread and the request threads both write here and SQLite
connections are not safe to pass between threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.models.schemas import DocumentStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    content_type  TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    status        TEXT NOT NULL,
    error         TEXT,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    completed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
"""


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    chunk_count: int
    created_at: str
    completed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Document":
        return cls(
            id=row["id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            status=DocumentStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with WAL enabled, committing on clean exit."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        # WAL lets the worker write while requests read; busy_timeout absorbs the
        # brief lock contention that still occurs on write-write overlap.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: str | Path) -> None:
    """Create the schema if it is not already there. Safe to call repeatedly."""
    with connect(path) as connection:
        connection.executescript(SCHEMA)


def insert_document(
    path: str | Path,
    *,
    document_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
) -> Document:
    """Record a new upload in ``pending``."""
    created_at = utcnow()
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO documents
                (id, filename, content_type, size_bytes, status, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                document_id,
                filename,
                content_type,
                size_bytes,
                DocumentStatus.PENDING.value,
                created_at,
            ),
        )
    return Document(
        id=document_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        status=DocumentStatus.PENDING,
        error=None,
        chunk_count=0,
        created_at=created_at,
        completed_at=None,
    )


def mark_processing(path: str | Path, document_id: str) -> None:
    _update_status(path, document_id, DocumentStatus.PROCESSING)


def mark_completed(path: str | Path, document_id: str, chunk_count: int) -> None:
    with connect(path) as connection:
        connection.execute(
            """
            UPDATE documents
               SET status = ?, chunk_count = ?, error = NULL, completed_at = ?
             WHERE id = ?
            """,
            (DocumentStatus.COMPLETED.value, chunk_count, utcnow(), document_id),
        )


def mark_failed(path: str | Path, document_id: str, error: str) -> None:
    """Record a terminal failure. Truncated so a huge traceback cannot bloat a row."""
    with connect(path) as connection:
        connection.execute(
            """
            UPDATE documents
               SET status = ?, error = ?, completed_at = ?
             WHERE id = ?
            """,
            (DocumentStatus.FAILED.value, error[:2000], utcnow(), document_id),
        )


def _update_status(path: str | Path, document_id: str, status: DocumentStatus) -> None:
    with connect(path) as connection:
        connection.execute(
            "UPDATE documents SET status = ? WHERE id = ?", (status.value, document_id)
        )


def get_document(path: str | Path, document_id: str) -> Document | None:
    with connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return Document.from_row(row) if row else None


def list_documents(
    path: str | Path, *, limit: int = 50, offset: int = 0
) -> list[Document]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [Document.from_row(row) for row in rows]


def count_documents(path: str | Path) -> int:
    with connect(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def delete_document(path: str | Path, document_id: str) -> bool:
    with connect(path) as connection:
        cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0


def reset_stale_processing(path: str | Path) -> int:
    """Fail any job left mid-flight by a process that died.

    Called at startup: the in-process queue does not survive a restart, so anything
    still ``processing`` has no worker behind it and would otherwise hang forever.
    """
    with connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE documents
               SET status = ?, error = ?, completed_at = ?
             WHERE status = ?
            """,
            (
                DocumentStatus.FAILED.value,
                "interrupted: the service restarted while this document was processing",
                utcnow(),
                DocumentStatus.PROCESSING.value,
            ),
        )
        return cursor.rowcount
