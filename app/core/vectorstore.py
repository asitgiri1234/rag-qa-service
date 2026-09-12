"""Vector storage behind a narrow interface.

Chroma is an implementation detail. Everything above this module talks to the
:class:`VectorStore` protocol, so swapping in FAISS or pgvector later is a new class
rather than a rewrite of the retrieval path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.core.chunking import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with its provenance and score.

    ``similarity`` is in [0, 1] for normalised vectors: 1.0 is an exact match.
    """

    chunk_id: str
    text: str
    similarity: float
    document_id: str
    filename: str
    chunk_index: int
    page_number: int
    token_count: int


class VectorStore(Protocol):
    """The storage operations the ingestion and query paths depend on."""

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int: ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Persistent Chroma collection using cosine distance.

    No embedding function is registered on the collection. Chroma would otherwise
    install its own default model and silently embed anything passed as raw text,
    which would mean two different models in one system -- exactly the failure the
    explicit-embeddings rule exists to prevent. Every write and every query here
    passes vectors we produced.
    """

    def __init__(self, path: str | Path, collection_name: str = COLLECTION_NAME) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.info(
            "opened chroma collection %r at %s (%d chunks)",
            collection_name,
            path,
            self._collection.count(),
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return 0

        started = time.perf_counter()
        # upsert, not add: re-ingesting a document must replace its chunks rather
        # than collide on the stable ids or silently duplicate them.
        self._collection.upsert(
            ids=[chunk_id(document_id, chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "token_count": chunk.token_count,
                }
                for chunk in chunks
            ],
        )
        logger.info(
            "stored %d chunks for document %s in %.3fs",
            len(chunks),
            document_id,
            time.perf_counter() - started,
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": list(document_ids)}}

        started = time.perf_counter()
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        results = [
            SearchResult(
                chunk_id=chunk_id_value,
                text=document,
                # Chroma returns cosine DISTANCE. With L2-normalised vectors that is
                # 1 - cosine_similarity, so similarity is recovered exactly by
                # inverting it. Every score reported by the API means this.
                similarity=1.0 - float(distance),
                document_id=str(metadata.get("document_id", "")),
                filename=str(metadata.get("filename", "")),
                chunk_index=int(metadata.get("chunk_index", -1)),
                page_number=int(metadata.get("page_number", -1)),
                token_count=int(metadata.get("token_count", -1)),
            )
            for chunk_id_value, document, metadata, distance in zip(
                _first(response, "ids"),
                _first(response, "documents"),
                _first(response, "metadatas"),
                _first(response, "distances"),
            )
        ]

        # Required by the project constraints: every retrieval logs its latency and
        # the similarity of each hit.
        logger.info(
            "vector search returned %d/%d chunks in %.1fms; scores=[%s]%s",
            len(results),
            top_k,
            elapsed_ms,
            ", ".join(f"{result.similarity:.4f}" for result in results),
            f" filtered to {len(document_ids)} document(s)" if document_ids else "",
        )
        return results

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(
            where={"document_id": document_id}, include=[]
        )
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        logger.info("deleted %d chunks for document %s", len(ids), document_id)
        return len(ids)

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop every chunk. Used by the evaluation sweep between configurations."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )


def chunk_id(document_id: str, chunk: Chunk) -> str:
    """Stable id so re-ingesting a document overwrites rather than duplicates."""
    return f"{document_id}:{chunk.chunk_index}"


def _first(response: dict[str, Any], key: str) -> list[Any]:
    """Chroma nests one list per query; we always send exactly one."""
    value = response.get(key)
    if not value:
        return []
    return value[0] or []


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Process-wide store, opened from config on first use."""
    from app.config import get_settings

    return ChromaVectorStore(get_settings().chroma_path)
