"""Document parsers.

Every parser returns ``list[tuple[int, str]]`` -- ``(page_number, text)`` pairs with
1-based page numbers, so page provenance survives into chunk metadata. Blank pages
are dropped; their numbers are simply absent from the result.

Pure functions: the only I/O is reading the file that was handed to us.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


class UnsupportedFormatError(ValueError):
    """Raised when a file's extension has no registered parser."""


# --- whitespace normalisation -------------------------------------------------

# Characters that carry no meaning once the text is extracted.
_INVISIBLE = str.maketrans(
    {
        "\u00ad": "",  # soft hyphen: a hyphenation artefact by definition
        "\u200b": "",  # zero-width space
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u00a0": " ",  # non-breaking space
        "\u2007": " ",
        "\u202f": " ",
        "\v": "\n",
        "\f": "\n\n",  # form feed is a page break
    }
)

# "hyphen- \n ation" -> "hyphenation". Restricted to a letter on both sides and a
# lowercase continuation, which is the overwhelmingly common case for a word broken
# across lines. Known limitation: a genuine compound broken at its hyphen
# ("state-\nof-the-art") loses that hyphen.
_HYPHEN_LINEBREAK = re.compile(r"(\w)[-\u2010\u2011]\n[ \t]*([a-z\u00df-\u024f])")

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
_HORIZONTAL_WS = re.compile(r"[ \t]+")

# Lines that begin a new markdown block and must keep their own line.
_BLOCK_START = re.compile(r"(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|~~~|:{3})")


def normalize_text(raw: str, *, keep_block_starts: bool = False) -> str:
    """Collapse extraction noise while preserving paragraph structure.

    - runs of spaces/tabs collapse to a single space
    - words hyphenated across a line break are rejoined
    - single newlines inside a paragraph become spaces
    - paragraph breaks are preserved as exactly ``\n\n``

    ``keep_block_starts`` keeps a line break before markdown block markers (headings,
    list items, quotes, table rows) so lists do not collapse into one run-on line.
    """
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n").translate(_INVISIBLE)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    paragraphs = []
    for block in _PARAGRAPH_BREAK.split(text):
        joined = _join_lines(block, keep_block_starts=keep_block_starts)
        joined = _HORIZONTAL_WS.sub(" ", joined).strip()
        if joined:
            paragraphs.append(joined)

    return "\n\n".join(paragraphs)


def _join_lines(block: str, *, keep_block_starts: bool) -> str:
    lines = [line.strip() for line in block.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    if not keep_block_starts:
        return " ".join(lines)

    out = lines[0]
    for line in lines[1:]:
        out += ("\n" if _BLOCK_START.match(line) else " ") + line
    return out


# --- parsers ------------------------------------------------------------------


def parse_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text per page with pypdf. Pages yielding no text are skipped."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def parse_txt(path: str | Path) -> list[tuple[int, str]]:
    """Plain text is a single logical page."""
    text = normalize_text(_read(path))
    return [(1, text)] if text else []


def parse_markdown(path: str | Path) -> list[tuple[int, str]]:
    """Markdown is a single logical page; block structure is preserved."""
    text = normalize_text(_read(path), keep_block_starts=True)
    return [(1, text)] if text else []


def _read(path: str | Path) -> str:
    # utf-8 with replacement: a stray bad byte should not fail an upload.
    return Path(path).read_text(encoding="utf-8", errors="replace")


_PARSERS = {
    ".pdf": parse_pdf,
    ".txt": parse_txt,
    ".text": parse_txt,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}

SUPPORTED_EXTENSIONS = frozenset(_PARSERS)


def parse_document(path: str | Path) -> list[tuple[int, str]]:
    """Dispatch to the parser registered for ``path``'s extension.

    Raises :class:`UnsupportedFormatError` for anything else.
    """
    suffix = Path(path).suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"unsupported file type {suffix or '(none)'!r}; supported: {supported}"
        )
    return parser(path)
