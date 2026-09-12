import math

import pytest

from app.core.vectorstore import ChromaVectorStore
from tests.conftest import make_chunk


def unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(tmp_path / "chroma")


def test_round_trip_preserves_text_and_metadata(store):
    chunks = [make_chunk("alpha content", 0, page=3), make_chunk("beta content", 1, page=4)]
    vectors = [unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])]

    added = store.add_chunks("doc-1", "paper.pdf", chunks, vectors)

    assert added == 2
    assert store.count() == 2

    results = store.search(unit([1.0, 0.0, 0.0]), top_k=2)

    top = results[0]
    assert top.text == "alpha content"
    assert top.document_id == "doc-1"
    assert top.filename == "paper.pdf"
    assert top.chunk_index == 0
    assert top.page_number == 3
    assert top.token_count == 2


def test_similarity_is_one_minus_cosine_distance(store):
    """An exact vector match must score ~1.0, not ~0.0."""
    store.add_chunks("doc-1", "a.txt", [make_chunk("only chunk", 0)], [unit([1.0, 0.0, 0.0])])

    (result,) = store.search(unit([1.0, 0.0, 0.0]), top_k=1)

    assert result.similarity == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vector_scores_near_zero(store):
    store.add_chunks("doc-1", "a.txt", [make_chunk("only chunk", 0)], [unit([1.0, 0.0, 0.0])])

    (result,) = store.search(unit([0.0, 1.0, 0.0]), top_k=1)

    assert result.similarity == pytest.approx(0.0, abs=1e-5)


def test_results_are_ranked_by_descending_similarity(store):
    chunks = [make_chunk(f"chunk {i}", i) for i in range(3)]
    vectors = [unit([1.0, 0.0, 0.0]), unit([0.9, 0.4, 0.0]), unit([0.0, 0.0, 1.0])]
    store.add_chunks("doc-1", "a.txt", chunks, vectors)

    results = store.search(unit([1.0, 0.0, 0.0]), top_k=3)

    scores = [result.similarity for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_can_be_filtered_to_specific_documents(store):
    store.add_chunks("doc-1", "a.txt", [make_chunk("alpha", 0)], [unit([1.0, 0.0, 0.0])])
    store.add_chunks("doc-2", "b.txt", [make_chunk("beta", 0)], [unit([0.99, 0.1, 0.0])])

    results = store.search(unit([1.0, 0.0, 0.0]), top_k=5, document_ids=["doc-2"])

    assert [result.document_id for result in results] == ["doc-2"]


def test_delete_document_removes_only_that_document(store):
    store.add_chunks("doc-1", "a.txt", [make_chunk("alpha", 0)], [unit([1.0, 0.0, 0.0])])
    store.add_chunks("doc-2", "b.txt", [make_chunk("beta", 0)], [unit([0.0, 1.0, 0.0])])

    removed = store.delete_document("doc-1")

    assert removed == 1
    assert store.count() == 1
    assert store.search(unit([0.0, 1.0, 0.0]), top_k=5)[0].document_id == "doc-2"


def test_reingesting_a_document_overwrites_rather_than_duplicates(store):
    chunk = make_chunk("alpha", 0)
    store.add_chunks("doc-1", "a.txt", [chunk], [unit([1.0, 0.0, 0.0])])
    store.add_chunks("doc-1", "a.txt", [chunk], [unit([1.0, 0.0, 0.0])])

    assert store.count() == 1


def test_mismatched_chunk_and_embedding_counts_are_rejected(store):
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.add_chunks(
            "doc-1", "a.txt", [make_chunk("a", 0), make_chunk("b", 1)], [unit([1.0, 0.0, 0.0])]
        )


def test_empty_add_is_a_no_op(store):
    assert store.add_chunks("doc-1", "a.txt", [], []) == 0
    assert store.count() == 0


@pytest.mark.slow
def test_nearest_neighbour_for_an_obviously_matching_query(store, embedder):
    """End to end with the real model: the topically matching chunk must win."""
    texts = [
        "The mitochondrion is the organelle that produces ATP inside a cell.",
        "Interest rates set by the central bank influence mortgage costs.",
        "Volcanoes form where tectonic plates diverge or one plate subducts.",
    ]
    chunks = [make_chunk(text, index) for index, text in enumerate(texts)]

    store.add_chunks("doc-1", "mixed.txt", chunks, embedder.embed_texts(texts))
    results = store.search(embedder.embed_query("What part of the cell makes energy?"), top_k=3)

    assert results[0].chunk_index == 0, f"expected the cell-biology chunk, got {results[0].text!r}"
    assert results[0].similarity > results[1].similarity


@pytest.mark.slow
def test_embedder_reports_the_dimension_and_the_256_token_ceiling(embedder):
    """The chunker's ceiling is derived from this number, so pin it down."""
    assert embedder.dimension == 384
    assert embedder.max_seq_length == 256

    vectors = embedder.embed_texts(["one", "two"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    # normalize_embeddings=True is what makes 1 - distance a true cosine similarity
    assert math.sqrt(sum(value * value for value in vectors[0])) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_empty_batch_returns_no_vectors(embedder):
    assert embedder.embed_texts([]) == []
