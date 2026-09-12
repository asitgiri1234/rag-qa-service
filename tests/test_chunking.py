import pytest

from app.core.chunking import (
    MODEL_MAX_TOKENS,
    Chunk,
    chunk_pages,
    chunk_text,
)
from tests.conftest import build_text


def test_overlap_actually_overlaps(tokenizer):
    """Chunk N's tail must reappear at the head of chunk N+1."""
    text = build_text(tokenizer, 400)

    chunks = chunk_text(
        text, tokenizer=tokenizer, chunk_size_tokens=60, chunk_overlap_tokens=20
    )

    assert len(chunks) > 2, "need several chunks for overlap to be observable"
    for previous, following in zip(chunks, chunks[1:]):
        assert following.char_start < previous.char_end, (
            f"chunk {following.chunk_index} starts at {following.char_start}, "
            f"after chunk {previous.chunk_index} ends at {previous.char_end}: no overlap"
        )
        shared = text[following.char_start : previous.char_end]
        assert shared.strip(), "overlapping region is whitespace only"
        assert shared in previous.text
        assert shared in following.text


def test_no_chunk_exceeds_the_token_ceiling(tokenizer):
    """The hard ceiling holds across configurations, measured on the emitted text."""
    text = build_text(tokenizer, 1200)

    for size, overlap in [(100, 20), (180, 40), (254, 60)]:
        chunks = chunk_text(
            text,
            tokenizer=tokenizer,
            chunk_size_tokens=size,
            chunk_overlap_tokens=overlap,
        )
        assert chunks
        for chunk in chunks:
            assert chunk.token_count <= size
            assert chunk.token_count <= MODEL_MAX_TOKENS
            # token_count must describe the text that will reach the embedder
            assert chunk.token_count == tokenizer.count(chunk.text)
            assert chunk.text == text[chunk.char_start : chunk.char_end]


def test_three_paragraphs_split_on_paragraph_boundaries(tokenizer):
    """Paragraph breaks win over sentence and word boundaries."""
    paragraphs = [
        "Dense retrieval encodes a passage into a single vector. "
        "That vector is compared against the encoded question by cosine similarity.",
        "Chunk size controls how much context each vector must represent. "
        "Passages that are too long dilute the signal of any single fact.",
        "Overlap exists so a fact sitting on a boundary is not lost. "
        "Without it, a sentence split across two chunks may match neither.",
    ]
    text = "\n\n".join(paragraphs)
    budget = max(tokenizer.count(p) for p in paragraphs) + 5

    chunks = chunk_text(
        text,
        tokenizer=tokenizer,
        chunk_size_tokens=budget,
        chunk_overlap_tokens=0,
    )

    assert [c.text for c in chunks] == paragraphs
    for chunk in chunks:
        assert chunk.text.endswith("."), "chunk ends mid-sentence"


def test_single_long_sentence_is_split_not_truncated(tokenizer):
    """A 500-token sentence exceeds every structural boundary and must still split."""
    words: list[str] = []
    while tokenizer.count(" ".join(words)) < 500:
        words.append("segmentation")
        words.append("strategy")
        words.append("matters")
    sentence = " ".join(words) + "."
    total_tokens = tokenizer.count(sentence)
    assert total_tokens >= 500

    chunks = chunk_text(
        sentence, tokenizer=tokenizer, chunk_size_tokens=180, chunk_overlap_tokens=0
    )

    assert len(chunks) >= 3
    assert all(c.token_count <= 180 for c in chunks)
    # Nothing may be dropped: the chunks must cover the whole sentence.
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(sentence)
    for previous, following in zip(chunks, chunks[1:]):
        gap = sentence[previous.char_end : following.char_start]
        assert not gap.strip(), f"content lost between chunks: {gap!r}"
    recovered = sum(c.token_count for c in chunks)
    assert recovered >= total_tokens - len(chunks), "substantial content went missing"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n", " "])
def test_empty_and_whitespace_only_input_returns_nothing(tokenizer, text):
    assert chunk_text(text, tokenizer=tokenizer) == []


def test_chunk_indices_and_pages_are_continuous(tokenizer):
    """chunk_pages numbers chunks across the document, keeping per-page provenance."""
    pages = [(1, build_text(tokenizer, 300)), (7, build_text(tokenizer, 300))]

    chunks = chunk_pages(
        pages, tokenizer=tokenizer, chunk_size_tokens=100, chunk_overlap_tokens=20
    )

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 7}
    for chunk in chunks:
        source = next(text for number, text in pages if number == chunk.page_number)
        assert chunk.text == source[chunk.char_start : chunk.char_end]


def test_chunk_size_above_the_model_ceiling_is_rejected(tokenizer):
    """Configuration that would silently truncate is refused, not accepted."""
    with pytest.raises(ValueError, match="exceeds the model ceiling"):
        chunk_text("hello world", tokenizer=tokenizer, chunk_size_tokens=400)


def test_overlap_must_be_smaller_than_chunk_size(tokenizer):
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(
            "hello world",
            tokenizer=tokenizer,
            chunk_size_tokens=50,
            chunk_overlap_tokens=50,
        )


def test_returns_chunk_dataclass_instances(tokenizer):
    chunks = chunk_text("A short document.", tokenizer=tokenizer)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "A short document."
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
