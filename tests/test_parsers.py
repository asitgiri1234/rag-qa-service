import pytest

from app.core.parsers import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFormatError,
    normalize_text,
    parse_document,
    parse_markdown,
    parse_txt,
)


def test_collapses_runs_of_spaces():
    assert normalize_text("too    many\t\tspaces") == "too many spaces"


def test_rejoins_words_hyphenated_across_a_line_break():
    assert normalize_text("hyphen-\nation is an artefact") == "hyphenation is an artefact"


def test_keeps_hyphen_when_the_line_break_is_a_paragraph_break():
    assert normalize_text("well-\n\nknown") == "well-\n\nknown"


def test_preserves_paragraph_breaks_as_double_newline():
    raw = "First paragraph\nwrapped over lines.\n\n\n  Second paragraph."
    assert normalize_text(raw) == "First paragraph wrapped over lines.\n\nSecond paragraph."


def test_single_newlines_inside_a_paragraph_become_spaces():
    assert normalize_text("line one\nline two") == "line one line two"


def test_strips_invisible_characters():
    assert normalize_text("soft­hyphen and nbsp") == "softhyphen and nbsp"


def test_markdown_keeps_list_items_on_their_own_lines(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n- first item\n- second item\n", encoding="utf-8")

    (page_number, text), = parse_markdown(path)

    assert page_number == 1
    assert text == "# Title\n- first item\n- second item"


def test_plain_text_collapses_those_same_lines(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Title\nfirst line\nsecond line\n", encoding="utf-8")

    assert parse_txt(path) == [(1, "Title first line second line")]


def test_blank_file_yields_no_pages(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n\t\n", encoding="utf-8")

    assert parse_txt(path) == []


def test_dispatch_routes_by_extension(tmp_path):
    path = tmp_path / "doc.MD"
    path.write_text("# Heading", encoding="utf-8")

    assert parse_document(path) == [(1, "# Heading")]


def test_dispatch_rejects_unknown_extensions(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")

    with pytest.raises(UnsupportedFormatError, match="unsupported file type"):
        parse_document(path)


def test_supported_extensions_are_advertised():
    assert {".pdf", ".txt", ".md"} <= SUPPORTED_EXTENSIONS
