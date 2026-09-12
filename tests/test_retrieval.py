import pytest

from app.core.generation import (
    NO_CONTEXT_ANSWER,
    GenerationError,
    build_messages,
    format_context,
    generate_answer,
)
from app.core.retrieval import retrieve
from app.core.vectorstore import SearchResult


def result(text="a passage", similarity=0.8, *, page=1, index=0, filename="notes.pdf"):
    return SearchResult(
        chunk_id=f"doc:{index}",
        text=text,
        similarity=similarity,
        document_id="doc",
        filename=filename,
        chunk_index=index,
        page_number=page,
        token_count=10,
    )


class FakeStore:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query_embedding, top_k, document_ids=None):
        self.calls.append((top_k, document_ids))
        return self._results[:top_k]


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]


# --- retrieval ----------------------------------------------------------------


def test_retrieve_keeps_results_above_the_floor():
    store = FakeStore([result(similarity=0.9), result(similarity=0.4, index=1)])

    outcome = retrieve("q", 5, min_similarity=0.25, embedder=FakeEmbedder(), store=store)

    assert len(outcome.results) == 2
    assert outcome.dropped_below_floor == 0
    assert outcome.top_similarity == pytest.approx(0.9)


def test_retrieve_drops_results_below_the_floor():
    store = FakeStore([result(similarity=0.9), result(similarity=0.10, index=1)])

    outcome = retrieve("q", 5, min_similarity=0.25, embedder=FakeEmbedder(), store=store)

    assert len(outcome.results) == 1
    assert outcome.dropped_below_floor == 1
    assert outcome.candidates == 2


def test_retrieve_returns_nothing_when_all_results_are_weak():
    """The caller must be able to refuse rather than pass junk to the model."""
    store = FakeStore([result(similarity=0.05), result(similarity=0.02, index=1)])

    outcome = retrieve("q", 5, min_similarity=0.25, embedder=FakeEmbedder(), store=store)

    assert outcome.results == []
    assert outcome.top_similarity is None
    assert outcome.dropped_below_floor == 2


def test_retrieve_uses_the_supplied_embedder_for_the_query():
    """The query must be encoded by the same embedder used at ingestion."""
    embedder = FakeEmbedder()

    retrieve("how many tokens?", 3, embedder=embedder, store=FakeStore([result()]))

    assert embedder.queries == ["how many tokens?"]


def test_retrieve_passes_the_document_filter_through():
    store = FakeStore([result()])

    retrieve("q", 3, ["doc-a"], embedder=FakeEmbedder(), store=store)

    assert store.calls == [(3, ["doc-a"])]


def test_retrieve_reports_mean_similarity():
    store = FakeStore([result(similarity=0.8), result(similarity=0.6, index=1)])

    outcome = retrieve("q", 5, embedder=FakeEmbedder(), store=store)

    assert outcome.mean_similarity == pytest.approx(0.7)


# --- prompt construction ------------------------------------------------------


def test_context_is_numbered_and_labelled_with_real_sources():
    results = [
        result("first passage", page=3, index=0, filename="a.pdf"),
        result("second passage", page=7, index=1, filename="b.pdf"),
    ]

    context = format_context(results)

    assert "[1] a.pdf p.3" in context
    assert "[2] b.pdf p.7" in context
    assert "first passage" in context


def test_system_prompt_forbids_answering_outside_the_context():
    messages = build_messages("q", [result()])

    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "only" in system.lower()
    assert "do not" in system.lower()
    assert "q" in messages[1]["content"]


# --- generation ---------------------------------------------------------------


class FakeCompletions:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        behaviour = self.behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


class FakeClient:
    def __init__(self, behaviours):
        self.chat = type("chat", (), {})()
        self.chat.completions = FakeCompletions(behaviours)


def response(text):
    message = type("m", (), {"content": text})()
    choice = type("c", (), {"message": message})()
    usage = type("u", (), {"prompt_tokens": 10, "completion_tokens": 5})()
    return type("r", (), {"choices": [choice], "usage": usage})()


def test_generate_answer_returns_the_model_text():
    client = FakeClient([response("  The ceiling is 256 tokens [1].  ")])

    outcome = generate_answer("q", [result()], client=client, model="test-model")

    assert outcome.answer == "The ceiling is 256 tokens [1]."
    assert outcome.model == "test-model"
    assert outcome.prompt_tokens == 10


def test_generate_answer_retries_once_on_a_transient_failure():
    from groq import APITimeoutError

    import httpx

    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))
    client = FakeClient([timeout, response("recovered")])

    outcome = generate_answer("q", [result()], client=client, model="test-model")

    assert outcome.answer == "recovered"
    assert client.chat.completions.calls == 2


def test_generate_answer_gives_up_after_one_retry():
    from groq import APITimeoutError

    import httpx

    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))
    client = FakeClient([timeout, timeout])

    with pytest.raises(GenerationError, match="after a retry"):
        generate_answer("q", [result()], client=client, model="test-model")

    assert client.chat.completions.calls == 2


def test_generate_answer_surfaces_permanent_errors_immediately():
    client = FakeClient([ValueError("model does not exist")])

    with pytest.raises(GenerationError, match="ValueError"):
        generate_answer("q", [result()], client=client, model="test-model")

    assert client.chat.completions.calls == 1


def test_no_context_answer_is_a_plain_refusal():
    assert "no relevant content" in NO_CONTEXT_ANSWER.lower()
