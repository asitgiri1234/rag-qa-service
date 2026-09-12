import json

import pytest

from app.core import metrics
from app.core.ingest import IngestionMetrics


class Outcome:
    def __init__(self, scores, elapsed_ms=5.0, candidates=None):
        from app.core.vectorstore import SearchResult

        self.results = [
            SearchResult(
                chunk_id=f"d:{i}",
                text="t",
                similarity=s,
                document_id="d",
                filename="f.pdf",
                chunk_index=i,
                page_number=1,
                token_count=5,
            )
            for i, s in enumerate(scores)
        ]
        self.elapsed_ms = elapsed_ms
        self.candidates = candidates if candidates is not None else len(scores)
        self.dropped_below_floor = self.candidates - len(scores)
        self.similarity_floor = 0.25
        self.top_similarity = scores[0] if scores else None
        self.mean_similarity = sum(scores) / len(scores) if scores else None
        self.min_similarity = min(scores) if scores else None


@pytest.fixture
def path(tmp_path):
    return tmp_path / "metrics.jsonl"


# --- percentiles --------------------------------------------------------------


def test_percentile_of_empty_is_none():
    assert metrics.percentile([], 0.5) is None


def test_percentile_of_single_value():
    assert metrics.percentile([7.0], 0.95) == 7.0


def test_percentile_matches_linear_interpolation():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert metrics.percentile(values, 0.50) == pytest.approx(5.5)
    assert metrics.percentile(values, 0.0) == 1
    assert metrics.percentile(values, 1.0) == 10
    assert metrics.percentile(values, 0.95) == pytest.approx(9.55)


def test_percentile_ignores_input_order():
    assert metrics.percentile([9, 1, 5], 0.5) == metrics.percentile([1, 5, 9], 0.5)


# --- recording ----------------------------------------------------------------


def test_query_record_has_every_required_field(path):
    metrics.record_query(
        question="how many tokens?",
        top_k=5,
        outcome=Outcome([0.8, 0.6, 0.4]),
        generation_ms=120.0,
        total_ms=180.0,
        answered=True,
        path=path,
    )

    record = json.loads(path.read_text(encoding="utf-8").strip())
    for field in (
        "timestamp",
        "question",
        "top_k",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "returned_chunk_count",
        "top_similarity",
        "mean_similarity",
        "min_similarity",
        "answered",
    ):
        assert field in record, f"missing {field}"
    assert record["returned_chunk_count"] == 3
    assert record["top_similarity"] == pytest.approx(0.8)
    assert record["min_similarity"] == pytest.approx(0.4)
    assert record["answered"] is True


def test_ingestion_record_has_every_required_field(path):
    metrics.record_ingestion(
        IngestionMetrics(
            document_id="doc-1",
            filename="a.pdf",
            size_bytes=4841,
            page_count=3,
            chunk_count=6,
            parse_ms=26.0,
            chunk_ms=24.0,
            embed_ms=185.0,
            store_ms=120.0,
            total_ms=367.0,
        ),
        path=path,
    )

    record = json.loads(path.read_text(encoding="utf-8").strip())
    for field in (
        "document_id",
        "size_bytes",
        "chunk_count",
        "parse_ms",
        "chunk_ms",
        "embed_ms",
        "store_ms",
        "total_ms",
    ):
        assert field in record, f"missing {field}"


def test_records_append_rather_than_overwrite(path):
    for index in range(3):
        metrics.record_query(
            question=f"q{index}",
            top_k=5,
            outcome=Outcome([0.5]),
            generation_ms=1.0,
            total_ms=2.0,
            answered=True,
            path=path,
        )

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_recording_never_raises_even_when_the_path_is_unusable(tmp_path):
    """A metrics failure must not break the request that triggered it."""
    unwritable = tmp_path / "a-file"
    unwritable.write_text("not a directory", encoding="utf-8")

    metrics.record_query(
        question="q",
        top_k=5,
        outcome=Outcome([0.5]),
        generation_ms=1.0,
        total_ms=2.0,
        answered=True,
        path=unwritable / "nested" / "metrics.jsonl",
    )


def test_malformed_lines_are_skipped(path):
    path.write_text('{"kind":"query","retrieval_ms":1,"total_ms":2,"answered":true}\nnot json\n', encoding="utf-8")

    assert len(metrics.read_records(path)) == 1


# --- aggregation --------------------------------------------------------------


def record_queries(path, specs):
    for scores, generation_ms, answered in specs:
        metrics.record_query(
            question="q",
            top_k=5,
            outcome=Outcome(scores),
            generation_ms=generation_ms,
            total_ms=generation_ms + 10,
            answered=answered,
            path=path,
        )


def test_summary_of_an_empty_file_is_well_formed(path):
    summary = metrics.summarize(path)

    assert summary["queries"]["count"] == 0
    assert summary["queries"]["refusal_rate"] is None
    assert summary["ingestion"]["documents"] == 0


def test_summary_computes_counts_and_refusal_rate(path):
    record_queries(
        path,
        [
            ([0.8], 100.0, True),
            ([0.7], 120.0, True),
            ([], 0.0, False),
            ([], 0.0, False),
        ],
    )

    summary = metrics.summarize(path)["queries"]

    assert summary["count"] == 4
    assert summary["answered"] == 2
    assert summary["refused"] == 2
    assert summary["refusal_rate"] == pytest.approx(0.5)


def test_generation_percentiles_exclude_refusals(path):
    """A refusal never calls the model; counting its 0ms would skew the latency."""
    record_queries(path, [([0.8], 100.0, True), ([], 0.0, False)])

    generation = metrics.summarize(path)["queries"]["generation_ms"]

    assert generation["count"] == 1
    assert generation["mean"] == pytest.approx(100.0)


def test_top_similarity_distribution_is_bucketed(path):
    record_queries(path, [([0.85], 1.0, True), ([0.31], 1.0, True), ([0.34], 1.0, True)])

    top = metrics.summarize(path)["queries"]["top_similarity"]

    assert top["distribution"]["0.3-0.4"] == 2
    assert top["distribution"]["0.8-0.9"] == 1
    assert top["mean"] == pytest.approx((0.85 + 0.31 + 0.34) / 3, abs=1e-4)


def test_ingestion_throughput_is_aggregated(path):
    metrics.record_ingestion(
        IngestionMetrics("d1", "a.pdf", 100, 1, 10, 1.0, 1.0, 500.0, 1.0, 1000.0),
        path=path,
    )
    metrics.record_ingestion(
        IngestionMetrics("d2", "b.pdf", 100, 1, 20, 1.0, 1.0, 1000.0, 1.0, 2000.0),
        path=path,
    )

    ingestion = metrics.summarize(path)["ingestion"]

    assert ingestion["documents"] == 2
    assert ingestion["total_chunks"] == 30
    # 10 chunks in 1s and 20 chunks in 2s are both 10/s
    assert ingestion["mean_chunks_per_second"] == pytest.approx(10.0)
    assert ingestion["mean_embed_ms_per_chunk"] == pytest.approx(50.0)


def test_summary_endpoint_returns_the_aggregate(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.limits import limiter
    from app.config import Settings
    from app.main import create_app

    metrics_path = tmp_path / "metrics.jsonl"
    record_queries(metrics_path, [([0.8], 100.0, True), ([], 0.0, False)])

    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "meta.db",
        chroma_path=tmp_path / "chroma",
        metrics_path=metrics_path,
        warm_start=False,
    )
    limiter.enabled = False
    try:
        with TestClient(create_app(settings)) as client:
            response = client.get("/metrics/summary")
    finally:
        limiter.enabled = True

    assert response.status_code == 200
    body = response.json()
    assert body["queries"]["count"] == 2
    assert body["queries"]["refusal_rate"] == pytest.approx(0.5)
    assert "p95" in body["queries"]["retrieval_ms"]
