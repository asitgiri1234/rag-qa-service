"""Embedding generation.

The same :class:`Embedder` must serve both ingestion and query. Encoding a corpus
with one model and questions with another puts the two sets of vectors in unrelated
spaces, and retrieval degrades to noise without raising any error -- so the model
name comes from config in exactly one place and is logged on load.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Wraps a sentence-transformers model, loaded once and reused.

    Vectors are L2-normalised at encode time, which makes cosine distance and dot
    product equivalent and lets the vector store convert distance to similarity with
    a plain ``1 - distance`` (see :mod:`app.core.vectorstore`).
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        # Imported here rather than at module scope: pulling in torch costs seconds,
        # and modules that only need the tokenizer should not pay for it.
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        logger.info(
            "loaded embedding model %s in %.2fs (dim=%d, max_seq_length=%d)",
            model_name,
            time.perf_counter() - started,
            self.dimension,
            self.max_seq_length,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector width; the collection is pinned to this."""
        # sentence-transformers 6.x renamed this; support both.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        """Word-piece tokens the model accepts before it truncates.

        This is the number the chunker's ceiling is derived from -- 256 for
        all-MiniLM-L6-v2. Read it rather than trusting the tokenizer's
        ``model_max_length``, which reports the base BERT value of 512.
        """
        return int(self._model.max_seq_length)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch, logging throughput for the metrics stage."""
        if not texts:
            return []

        started = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "embedded %d texts in %.3fs (%.1f texts/s, batch_size=%d)",
            len(texts),
            elapsed,
            len(texts) / elapsed if elapsed else float("inf"),
            self._batch_size,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Encode a single question with the ingestion-side model."""
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide embedder, built from config on first use."""
    settings = get_settings()
    return Embedder(settings.embedding_model)
