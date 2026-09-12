"""Ingest a file straight from the command line, bypassing the API.

Exercises the real pipeline (parse -> chunk -> embed -> store) without a running
server, which is how the corpus for the evaluation sweep gets loaded.

    python scripts/ingest_local.py data/samples/paper.pdf
    python scripts/ingest_local.py data/samples/*.txt --chunk-size 300 --overlap 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

# Allow "python scripts/ingest_local.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.ingest import ingest_document  # noqa: E402
from app.core.parsers import SUPPORTED_EXTENSIONS  # noqa: E402
from app.storage import db  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="files to ingest")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="override chunk_size_tokens"
    )
    parser.add_argument(
        "--overlap", type=int, default=None, help="override chunk_overlap_tokens"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = get_settings()
    if args.chunk_size is not None:
        settings = settings.model_copy(update={"chunk_size_tokens": args.chunk_size})
    if args.overlap is not None:
        settings = settings.model_copy(update={"chunk_overlap_tokens": args.overlap})

    db.init_db(settings.sqlite_path)

    failures = 0
    for path in args.paths:
        if not path.is_file():
            print(f"skip {path}: not a file", file=sys.stderr)
            failures += 1
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"skip {path}: unsupported extension", file=sys.stderr)
            failures += 1
            continue

        document_id = str(uuid.uuid4())
        db.insert_document(
            settings.sqlite_path,
            document_id=document_id,
            filename=path.name,
            content_type="application/octet-stream",
            size_bytes=path.stat().st_size,
        )
        try:
            metrics = ingest_document(
                document_id, path, filename=path.name, settings=settings
            )
        except Exception as error:
            print(f"FAILED {path.name}: {error}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"OK {path.name}: {metrics.chunk_count} chunks from {metrics.page_count} "
            f"page(s) in {metrics.total_ms:.0f}ms  "
            f"[parse {metrics.parse_ms:.0f} | chunk {metrics.chunk_ms:.0f} | "
            f"embed {metrics.embed_ms:.0f} | store {metrics.store_ms:.0f}]  "
            f"id={document_id}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
