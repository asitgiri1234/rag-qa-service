import pytest
from fastapi.testclient import TestClient

from app.api.limits import limiter
from app.config import Settings
from app.core.vectorstore import SearchResult
from app.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "meta.db",
        chroma_path=tmp_path / "chroma",
        metrics_path=tmp_path / "metrics.jsonl",
        warm_start=False,
        min_similarity=0.25,
        top_k=5,
    )


@pytest.fixture(autouse=True)
def no_rate_limit():
    limiter.enabled = False
    yield
    limiter.enabled = True


def hit(similarity=0.8, index=0, page=2):
    return SearchResult(
        chunk_id=f"doc:{index}",
        text="The embedding model truncates at 256 word-piece tokens.",
        similarity=similarity,
        document_id="doc",
        filename="notes.pdf",
        chunk_index=index,
        page_number=page,
        token_count=12,
    )


class Outcome:
    """Stand-in for RetrievalOutcome with only what the endpoint reads."""

    def __init__(self, results):
        self.results = results
        self.elapsed_ms = 4.2
        self.candidates = len(results)
        self.dropped_below_floor = 0
        self.min_similarity = 0.25
        self.top_similarity = results[0].similarity if results else None
        self.mean_similarity = (
            sum(r.similarity for r in results) / len(results) if results else None
        )


@pytest.fixture
def client(settings, monkeypatch):
    def fake_retrieve(question, top_k, document_ids=None, **kwargs):
        fake_retrieve.seen = (question, top_k, document_ids)
        return Outcome([hit(0.81, 0), hit(0.44, 1, page=3)])

    def fake_generate(question, results, **kwargs):
        from app.core.generation import GenerationOutcome

        return GenerationOutcome(
            answer="It truncates at 256 tokens [1].", elapsed_ms=12.0, model="fake"
        )

    monkeypatch.setattr("app.api.query.retrieve", fake_retrieve)
    monkeypatch.setattr("app.api.query.generate_answer", fake_generate)
    with TestClient(create_app(settings)) as test_client:
        test_client.fake_retrieve = fake_retrieve
        yield test_client


def test_query_returns_answer_sources_and_timings(client):
    response = client.post("/query", json={"question": "How many tokens?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It truncates at 256 tokens [1]."
    assert len(body["sources"]) == 2
    assert body["retrieval_ms"] >= 0
    assert body["generation_ms"] >= 0
    assert body["total_ms"] >= 0


def test_sources_carry_the_similarity_score(client):
    """Scores are part of the contract: they make a bad retrieval visible."""
    body = client.post("/query", json={"question": "How many tokens?"}).json()

    first = body["sources"][0]
    assert first["similarity_score"] == pytest.approx(0.81)
    assert first["filename"] == "notes.pdf"
    assert first["page_number"] == 2
    assert first["chunk_index"] == 0
    assert first["text"]


def test_top_k_defaults_to_the_configured_value(client):
    client.post("/query", json={"question": "q"})

    assert client.fake_retrieve.seen[1] == 5


def test_top_k_can_be_overridden(client):
    client.post("/query", json={"question": "q", "top_k": 3})

    assert client.fake_retrieve.seen[1] == 3


def test_document_filter_is_passed_through(client):
    client.post("/query", json={"question": "q", "document_ids": ["a", "b"]})

    assert client.fake_retrieve.seen[2] == ["a", "b"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 21},
        {},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_no_relevant_content_refuses_without_calling_the_model(settings, monkeypatch):
    """When nothing clears the floor the LLM must not be called at all."""
    called = []

    monkeypatch.setattr(
        "app.api.query.retrieve", lambda *args, **kwargs: Outcome([])
    )
    monkeypatch.setattr(
        "app.api.query.generate_answer",
        lambda *args, **kwargs: called.append(1),
    )

    with TestClient(create_app(settings)) as client:
        response = client.post("/query", json={"question": "unrelated question"})

    body = response.json()
    assert response.status_code == 200
    assert body["sources"] == []
    assert "no relevant content" in body["answer"].lower()
    assert body["generation_ms"] == 0
    assert called == [], "the model was called despite no usable context"


def test_generation_failure_returns_503(settings, monkeypatch):
    from app.core.generation import GenerationError

    def boom(*args, **kwargs):
        raise GenerationError("groq unavailable")

    monkeypatch.setattr(
        "app.api.query.retrieve", lambda *args, **kwargs: Outcome([hit()])
    )
    monkeypatch.setattr("app.api.query.generate_answer", boom)

    with TestClient(create_app(settings)) as client:
        response = client.post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable"


def test_query_is_rate_limited(settings, monkeypatch):
    monkeypatch.setattr(
        "app.api.query.retrieve", lambda *args, **kwargs: Outcome([])
    )
    limiter.enabled = True
    limiter.reset()

    with TestClient(create_app(settings)) as client:
        statuses = [
            client.post("/query", json={"question": f"q{i}"}).status_code
            for i in range(24)
        ]

    assert 429 in statuses, f"query endpoint was never limited: {statuses}"
    limiter.reset()
