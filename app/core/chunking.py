"""Structure-aware chunking measured in word-piece tokens.

Why tokens and not characters: ``all-MiniLM-L6-v2`` truncates its input at 256
word-piece tokens. A character budget maps to a wildly variable token count -- code,
tables and non-English text tokenize far denser than prose -- so a character-based
chunker silently emits chunks whose tails never reach the embedding. Every length in
this module is a word-piece count from the embedding model's own tokenizer.

This module is pure: it performs no I/O and reads no globals. The tokenizer and every
tuning parameter arrive as arguments so a sweep script can vary them freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.tokenization import Tokenizer

#: Sequence length at which all-MiniLM-L6-v2 truncates, including the [CLS] and [SEP]
#: markers the model adds. Chunks are measured without those markers.
MODEL_MAX_TOKENS = 256

#: Tokens actually available to content, once [CLS] and [SEP] are accounted for.
#: Staying at or below this is what genuinely guarantees no truncation.
CONTENT_TOKEN_BUDGET = MODEL_MAX_TOKENS - 2

DEFAULT_CHUNK_SIZE_TOKENS = 180
DEFAULT_CHUNK_OVERLAP_TOKENS = 40


@dataclass(frozen=True)
class Chunk:
    """One embeddable span of a single page.

    ``char_start``/``char_end`` index the *normalized* text of that page, and
    ``text`` is exactly ``page_text[char_start:char_end]``.
    """

    text: str
    token_count: int
    chunk_index: int
    page_number: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _Unit:
    """An indivisible span that fits the budget, carrying absolute offsets."""

    start: int
    end: int
    tokens: int


# --- boundary detection -------------------------------------------------------

_PARAGRAPH = re.compile(r"\n{2,}")
_WHITESPACE = re.compile(r"\s+")

# End punctuation, any closing quotes/brackets, then whitespace or end-of-text. The
# lookahead is what keeps "3.14" and "file.txt" from being read as sentence ends.
_SENTENCE_END = re.compile("[.!?…]+[\"'”’)\\]]*(?=\\s|$)")

# Abbreviations whose trailing period is not a sentence end. Single letters (initials,
# and the trailing "g" of "e.g.") are handled separately.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st rev hon gen col capt lt sgt
    vs etc al eg ie cf ca approx est
    fig figs eq eqs ref refs no nos vol vols ch chap sec secs pp
    inc ltd co corp dept univ jan feb mar apr jun jul aug sep sept oct nov dec""".split()
)


def _sentence_spans(text: str, base: int) -> list[tuple[int, int]]:
    """Trimmed ``(start, end)`` spans of the sentences in ``text``."""
    cuts: list[int] = []
    for match in _SENTENCE_END.finditer(text):
        if _is_abbreviation(text, match.start()):
            continue
        cuts.append(match.end())

    spans: list[tuple[int, int]] = []
    position = 0
    for cut in [*cuts, len(text)]:
        if cut <= position:
            continue
        span = _trim(text, base, position, cut)
        if span:
            spans.append(span)
        position = cut
    return spans


def _is_abbreviation(text: str, punct_start: int) -> bool:
    """True when the period at ``punct_start`` closes an abbreviation."""
    if text[punct_start] != ".":
        return False
    word = ""
    index = punct_start - 1
    while index >= 0 and (text[index].isalnum() or text[index] == "."):
        word = text[index] + word
        index -= 1
    if not word:
        return False
    # A lone letter is an initial ("J. Smith") or the tail of "e.g." / "i.e.".
    if len(word.replace(".", "")) == 1:
        return True
    return word.lower().strip(".") in _ABBREVIATIONS


def _split_on(text: str, base: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for match in pattern.finditer(text):
        span = _trim(text, base, position, match.start())
        if span:
            spans.append(span)
        position = match.end()
    span = _trim(text, base, position, len(text))
    if span:
        spans.append(span)
    return spans


def _trim(text: str, base: int, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` past surrounding whitespace; None if empty."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (base + start, base + end) if end > start else None


# --- recursive splitting ------------------------------------------------------

_PARAGRAPHS, _SENTENCES, _WORDS, _HARD = range(4)


def _to_units(
    source: str, start: int, end: int, budget: int, tokenizer: Tokenizer, level: int
) -> list[_Unit]:
    """Split ``source[start:end]`` into units that each fit ``budget`` tokens.

    Descends paragraphs -> sentences -> words, and only cuts mid-sentence at exact
    token boundaries once every structural boundary has been exhausted.
    """
    fragment = source[start:end]
    tokens = tokenizer.count(fragment)
    if tokens == 0:
        return []
    if tokens <= budget:
        return [_Unit(start, end, tokens)]

    if level == _PARAGRAPHS:
        spans = _split_on(fragment, start, _PARAGRAPH)
    elif level == _SENTENCES:
        spans = _sentence_spans(fragment, start)
    elif level == _WORDS:
        spans = _split_on(fragment, start, _WHITESPACE)
    else:
        return _hard_split(source, start, end, budget, tokenizer)

    # This level found no boundary; try the next one rather than looping.
    if len(spans) <= 1:
        return _to_units(source, start, end, budget, tokenizer, level + 1)

    units: list[_Unit] = []
    for span_start, span_end in spans:
        units.extend(
            _to_units(source, span_start, span_end, budget, tokenizer, level + 1)
        )
    return units


def _hard_split(
    source: str, start: int, end: int, budget: int, tokenizer: Tokenizer
) -> list[_Unit]:
    """Last resort: cut at exact word-piece boundaries using offset mapping.

    Reached only by a single sentence that exceeds the budget on its own, so the cut
    lands on a real token edge rather than mid-word.
    """
    fragment = source[start:end]
    spans = tokenizer.token_spans(fragment)
    if not spans:
        return []

    units: list[_Unit] = []
    for offset in range(0, len(spans), budget):
        window = spans[offset : offset + budget]
        piece_start = start + window[0][0]
        piece_end = start + window[-1][1]
        units.append(_Unit(piece_start, piece_end, len(window)))
    return units


# --- assembly -----------------------------------------------------------------


def _carry_overlap(units: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Trailing units totalling at most ``overlap_tokens``.

    Always leaves at least one unit behind, which is what guarantees the assembly
    loop makes forward progress.
    """
    if overlap_tokens <= 0 or len(units) <= 1:
        return []
    kept: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.tokens > overlap_tokens:
            break
        kept.insert(0, unit)
        total += unit.tokens
    if len(kept) >= len(units):
        kept = kept[1:]
    return kept


def chunk_text(
    text: str,
    *,
    tokenizer: Tokenizer,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    page_number: int = 1,
    start_index: int = 0,
    hard_max_tokens: int = MODEL_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one page of normalized text.

    Returns ``[]`` for empty or whitespace-only input. Every parameter is an argument
    so a sweep script can vary the configuration without touching module state.
    """
    if chunk_size_tokens < 1:
        raise ValueError("chunk_size_tokens must be >= 1")
    if not 0 <= chunk_overlap_tokens < chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must satisfy 0 <= overlap < chunk_size")
    if chunk_size_tokens > hard_max_tokens:
        raise ValueError(
            f"chunk_size_tokens ({chunk_size_tokens}) exceeds the model ceiling "
            f"({hard_max_tokens}); chunks would be truncated before embedding"
        )
    if not text or not text.strip():
        return []

    units = _to_units(text, 0, len(text), chunk_size_tokens, tokenizer, _PARAGRAPHS)
    if not units:
        return []

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_tokens = 0
    index = start_index
    position = 0

    while position < len(units):
        unit = units[position]
        if current and current_tokens + unit.tokens > chunk_size_tokens:
            chunks.append(
                _emit(text, current, index, page_number, tokenizer, hard_max_tokens)
            )
            index += 1
            current = _carry_overlap(current, chunk_overlap_tokens)
            current_tokens = sum(u.tokens for u in current)
            # Drop carried context from the front until the next unit fits, so the
            # overlap can never cause a duplicate chunk to be emitted.
            while current and current_tokens + unit.tokens > chunk_size_tokens:
                current_tokens -= current.pop(0).tokens
            continue
        current.append(unit)
        current_tokens += unit.tokens
        position += 1

    if current:
        chunks.append(
            _emit(text, current, index, page_number, tokenizer, hard_max_tokens)
        )
    return chunks


def _emit(
    source: str,
    units: list[_Unit],
    index: int,
    page_number: int,
    tokenizer: Tokenizer,
    hard_max_tokens: int,
) -> Chunk:
    char_start = units[0].start
    char_end = units[-1].end
    body = source[char_start:char_end]
    # Counted on the emitted text, not summed from the units, so the assertion below
    # tests what will actually be handed to the embedder.
    token_count = tokenizer.count(body)
    assert token_count <= hard_max_tokens, (
        f"chunk {index} is {token_count} tokens, over the {hard_max_tokens}-token "
        f"ceiling; it would be truncated before embedding"
    )
    return Chunk(
        text=body,
        token_count=token_count,
        chunk_index=index,
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
    )


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    tokenizer: Tokenizer,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    hard_max_tokens: int = MODEL_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk parser output, numbering chunks continuously across pages."""
    chunks: list[Chunk] = []
    for page_number, page_text in pages:
        chunks.extend(
            chunk_text(
                page_text,
                tokenizer=tokenizer,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                page_number=page_number,
                start_index=len(chunks),
                hard_max_tokens=hard_max_tokens,
            )
        )
    return chunks
