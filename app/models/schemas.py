"""Shared schemas and enums.

The status enum lives here rather than in the storage layer so the database, the
worker and the API all agree on one vocabulary.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle of an uploaded document.

    A document is created ``PENDING``, moves to ``PROCESSING`` when the worker picks
    it up, and ends at ``COMPLETED`` or ``FAILED``. It must never be left at
    ``PROCESSING``: a crashed job records the error and moves to ``FAILED``.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- documents ----------------------------------------------------------------


class UploadResponse(BaseModel):
    document_id: str = Field(..., description="Poll GET /documents/{id} for progress")
    status: DocumentStatus
    message: str


class DocumentSummary(BaseModel):
    """A document's stored state, as returned by the status and list endpoints."""

    document_id: str
    filename: str
    content_type: str
    size_bytes: int = Field(..., ge=0)
    status: DocumentStatus
    chunk_count: int = Field(0, ge=0)
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class DocumentList(BaseModel):
    documents: list[DocumentSummary]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int = Field(..., ge=0)
    message: str


# --- query --------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20, description="defaults to config top_k")
    document_ids: list[str] | None = Field(
        None, description="restrict retrieval to these documents"
    )


class Source(BaseModel):
    """One retrieved chunk, with the score that put it in the answer.

    The similarity is returned deliberately: without it a bad answer is
    indistinguishable from a bad retrieval from outside the service.
    """

    filename: str
    page_number: int
    chunk_index: int
    similarity_score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_ms: float = Field(..., ge=0)
    generation_ms: float = Field(..., ge=0)
    total_ms: float = Field(..., ge=0)


# --- errors -------------------------------------------------------------------


class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
