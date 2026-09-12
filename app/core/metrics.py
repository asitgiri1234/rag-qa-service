"""Metrics recording and aggregation.

Every query and every ingestion appends one JSON line to ``data/metrics.jsonl``.
Log lines alone are not metrics: they cannot be aggregated without grep and
arithmetic, so the records here are structured and the summary endpoint computes
real percentiles over them.

Recording must never break a request. Every write is wrapped, and a failure to
record is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

QUERY = "query"
INGESTION = "ingestion"

# Appends are short and the file is opened per write, but two threads (the request
# thread and the worker) can both record, so serialise them.
_lock = threading.Lock()


def _metrics_path() -> Path:
    from app.config import get_settings

    return Path(get_settings().metrics_path)


def _append(record: dict[str, Any], path: Path | None = None) -> None:
    path = path or _metrics_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        # A metrics failure must never surface to the caller.
        logger.exception("failed to append a metrics record")


def record_query(
    *,
    question: str,
    top_k: int,
    outcome,
    generation_ms: float,
    total_ms: float,
    answered: bool,
    path: Path | None = None,
) -> None:
    """Record one query. ``answered`` is False when we refused for lack of context."""
    _append(
        {
            "kind": QUERY,
            "timestamp": _now(),
            "question": question,
            "top_k": top_k,
            "retrieval_ms": round(outcome.elapsed_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_ms": round(total_ms, 3),
            "returned_chunk_count": len(outcome.results),
            "candidates": outcome.candidates,
            "dropped_below_floor": outcome.dropped_below_floor,
            "similarity_floor": outcome.similarity_floor,
            "top_similarity": _round(outcome.top_similarity),
            "mean_similarity": _round(outcome.mean_similarity),
            "min_similarity": _round(outcome.min_similarity),
            "answered": answered,
        },
        path,
    )


def record_ingestion(metrics, path: Path | None = None) -> None:
    """Record one document ingestion with its per-phase timings."""
    _append(
        {
            "kind": INGESTION,
            "timestamp": _now(),
            "document_id": metrics.document_id,
            "filename": metrics.filename,
            "size_bytes": metrics.size_bytes,
            "page_count": metrics.page_count,
            "chunk_count": metrics.chunk_count,
            "parse_ms": round(metrics.parse_ms, 3),
            "chunk_ms": round(metrics.chunk_ms, 3),
            "embed_ms": round(metrics.embed_ms, 3),
            "store_ms": round(metrics.store_ms, 3),
            "total_ms": round(metrics.total_ms, 3),
        },
        path,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# --- aggregation --------------------------------------------------------------


def read_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Load every record, skipping any line that is not valid JSON."""
    path = path or _metrics_path()
    if not Path(path).exists():
        return []
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A partially written line should not poison the whole summary.
            logger.warning("skipping malformed metrics line")
    return records


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile, in plain Python.

    ``fraction`` is in [0, 1]; 0.95 is p95. Matches numpy's default 'linear'
    method so the numbers are comparable to anything computed elsewhere.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": _r(percentile(values, 0.50)),
        "p95": _r(percentile(values, 0.95)),
        "p99": _r(percentile(values, 0.99)),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _histogram(values: list[float]) -> dict[str, int]:
    """Similarity distribution in fixed 0.1 buckets."""
    buckets = {f"{edge / 10:.1f}-{(edge + 1) / 10:.1f}": 0 for edge in range(10)}
    for value in values:
        index = min(int(max(value, 0.0) * 10), 9)
        buckets[f"{index / 10:.1f}-{(index + 1) / 10:.1f}"] += 1
    return buckets


def summarize(path: Path | None = None) -> dict[str, Any]:
    """Aggregate the recorded metrics into the summary payload."""
    records = read_records(path)
    queries = [r for r in records if r.get("kind") == QUERY]
    ingestions = [r for r in records if r.get("kind") == INGESTION]

    answered = [r for r in queries if r.get("answered")]
    refused = [r for r in queries if not r.get("answered")]
    top_similarities = [
        r["top_similarity"] for r in queries if r.get("top_similarity") is not None
    ]

    # Generation time is only meaningful for queries that actually called the model.
    generation_values = [r["generation_ms"] for r in answered]

    chunks_per_second = []
    embed_ms_per_chunk = []
    for record in ingestions:
        chunk_count = record.get("chunk_count") or 0
        total_ms = record.get("total_ms") or 0
        embed_ms = record.get("embed_ms") or 0
        if chunk_count and total_ms:
            chunks_per_second.append(chunk_count / (total_ms / 1000))
        if chunk_count:
            embed_ms_per_chunk.append(embed_ms / chunk_count)

    return {
        "queries": {
            "count": len(queries),
            "answered": len(answered),
            "refused": len(refused),
            "refusal_rate": (
                round(len(refused) / len(queries), 4) if queries else None
            ),
            "retrieval_ms": _stats([r["retrieval_ms"] for r in queries]),
            "generation_ms": _stats(generation_values),
            "total_ms": _stats([r["total_ms"] for r in queries]),
            "top_similarity": {
                "mean": (
                    round(sum(top_similarities) / len(top_similarities), 4)
                    if top_similarities
                    else None
                ),
                "p50": _round(percentile(top_similarities, 0.50)),
                "p95": _round(percentile(top_similarities, 0.95)),
                "min": round(min(top_similarities), 4) if top_similarities else None,
                "max": round(max(top_similarities), 4) if top_similarities else None,
                "distribution": _histogram(top_similarities),
            },
        },
        "ingestion": {
            "documents": len(ingestions),
            "total_chunks": sum(r.get("chunk_count", 0) for r in ingestions),
            "mean_chunks_per_second": (
                round(sum(chunks_per_second) / len(chunks_per_second), 2)
                if chunks_per_second
                else None
            ),
            "mean_embed_ms_per_chunk": (
                round(sum(embed_ms_per_chunk) / len(embed_ms_per_chunk), 2)
                if embed_ms_per_chunk
                else None
            ),
            "parse_ms": _stats([r["parse_ms"] for r in ingestions]),
            "chunk_ms": _stats([r["chunk_ms"] for r in ingestions]),
            "embed_ms": _stats([r["embed_ms"] for r in ingestions]),
            "store_ms": _stats([r["store_ms"] for r in ingestions]),
        },
    }


def iter_queries(path: Path | None = None) -> Iterator[dict[str, Any]]:
    for record in read_records(path):
        if record.get("kind") == QUERY:
            yield record
