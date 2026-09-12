"""Word-piece tokenization.

Isolated from :mod:`app.core.chunking` because loading a tokenizer touches the
network and the filesystem (the HuggingFace cache). Chunking itself stays pure by
receiving a :class:`Tokenizer` as an argument.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Tokenizer(Protocol):
    """The slice of tokenizer behaviour the chunker depends on."""

    def count(self, text: str) -> int:
        """Number of word-piece tokens in ``text``, excluding special tokens."""

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """``(char_start, char_end)`` for each word-piece token in ``text``."""


class WordPieceTokenizer:
    """Adapter over a HuggingFace fast tokenizer.

    Special tokens are excluded from every count: the chunker measures content,
    and the two sequence markers are accounted for by the caller's ceiling (see
    ``CONTENT_TOKEN_BUDGET`` in :mod:`app.core.chunking`).
    """

    def __init__(self, hf_tokenizer) -> None:
        if not getattr(hf_tokenizer, "is_fast", False):
            raise ValueError(
                "a fast tokenizer is required: offset mapping is used to split "
                "oversized sentences at exact token boundaries"
            )
        self._tokenizer = hf_tokenizer

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self._encode(text)["input_ids"])

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        if not text.strip():
            return []
        offsets = self._encode(text)["offset_mapping"]
        # Guard against degenerate (0, 0) spans some tokenizers emit.
        return [(int(s), int(e)) for s, e in offsets if e > s]

    def _encode(self, text: str):
        # verbose=False suppresses the "sequence longer than model_max_length"
        # warning; we never truncate here, we measure and split ourselves.
        return self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str = DEFAULT_EMBEDDING_MODEL) -> WordPieceTokenizer:
    """Load and cache the tokenizer for ``model_name``.

    Performs network I/O on first use (downloads into the HuggingFace cache), so
    call it once at application startup and pass the result down.
    """
    from transformers import AutoTokenizer

    return WordPieceTokenizer(AutoTokenizer.from_pretrained(model_name))
