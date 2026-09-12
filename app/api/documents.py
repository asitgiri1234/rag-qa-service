"""Document upload and management endpoints.

Upload validates, persists, enqueues and returns ``202`` without waiting for the
pipeline. The work happens on the worker thread; progress is read back through
``GET /documents/{id}``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from app.config import Settings, get_settings
from app.core import worker
from app.models.schemas import (
    DeleteResponse,
    DocumentList,
    DocumentStatus,
    DocumentSummary,
    UploadResponse,
)
from app.storage import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

#: Declared types we accept per extension. Browsers and curl disagree about
#: markdown and plain text, so the set is deliberately permissive -- the real
#: gatekeeping is the extension plus the magic-byte check below.
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/octet-stream",
    "",
}

PDF_MAGIC = b"%PDF-"
READ_CHUNK = 1024 * 1024


def _settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for asynchronous ingestion",
)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = _settings(request)
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no filename supplied")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported extension {suffix or '(none)'}; "
            f"allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if (file.content_type or "") not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type {file.content_type!r}",
        )

    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{document_id}{suffix}"

    size_bytes = await _save_upload(file, destination, settings.max_upload_mb)

    if size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "uploaded file is empty")

    # Sniff rather than trust: a .pdf extension on a non-PDF would otherwise fail
    # deep inside the worker, long after the client got its 202.
    if suffix == ".pdf":
        with destination.open("rb") as handle:
            if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "file does not begin with the PDF magic bytes despite its .pdf extension",
                )

    db.insert_document(
        settings.sqlite_path,
        document_id=document_id,
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=size_bytes,
    )
    worker.enqueue_ingestion(document_id, destination, filename)

    return UploadResponse(
        document_id=document_id,
        status=DocumentStatus.PENDING,
        message="accepted for processing; poll GET /documents/{id} for status",
    )


async def _save_upload(file: UploadFile, destination: Path, max_mb: int) -> int:
    """Stream to disk, aborting past the size limit rather than buffering it all."""
    limit = max_mb * 1024 * 1024
    size = 0
    try:
        with destination.open("wb") as handle:
            while data := await file.read(READ_CHUNK):
                size += len(data)
                if size > limit:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"file exceeds the {max_mb} MB limit",
                    )
                handle.write(data)
    except HTTPException:
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return size


@router.get("/{document_id}", response_model=DocumentSummary)
def get_document(request: Request, document_id: str) -> DocumentSummary:
    settings = _settings(request)
    document = db.get_document(settings.sqlite_path, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id}")
    return _to_summary(document)


@router.get("", response_model=DocumentList)
def list_documents(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DocumentList:
    settings = _settings(request)
    documents = db.list_documents(settings.sqlite_path, limit=limit, offset=offset)
    return DocumentList(
        documents=[_to_summary(document) for document in documents],
        total=db.count_documents(settings.sqlite_path),
        limit=limit,
        offset=offset,
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(request: Request, document_id: str) -> DeleteResponse:
    settings = _settings(request)
    document = db.get_document(settings.sqlite_path, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id}")

    from app.core.vectorstore import get_vector_store

    removed = get_vector_store().delete_document(document_id)
    db.delete_document(settings.sqlite_path, document_id)

    upload = next(
        (p for p in (Path(settings.data_dir) / "uploads").glob(f"{document_id}.*")), None
    )
    if upload is not None:
        upload.unlink(missing_ok=True)

    return DeleteResponse(
        document_id=document_id,
        deleted_chunks=removed,
        message="document and its chunks removed",
    )


def _to_summary(document: db.Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        chunk_count=document.chunk_count,
        error=document.error,
        created_at=document.created_at,
        completed_at=document.completed_at,
    )
