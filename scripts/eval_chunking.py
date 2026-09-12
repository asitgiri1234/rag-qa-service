"""Sweep chunk configurations and measure what each one does to retrieval.

    python scripts/eval_chunking.py --corpus eval/corpus --questions eval/questions.json

For each configuration the corpus is ingested into its own throwaway Chroma
collection, every question is run, and the results are written to
``eval/eval_results.json`` alongside a printed comparison table.

Note on the oversized configurations: :func:`app.core.chunking.chunk_text` normally
refuses a chunk size above the model's 256-token ceiling. The sweep raises that guard
deliberately so the 300- and 500-token configurations can run, then counts how many
emitted chunks exceed 256 tokens. Those chunks are truncated by the model before the
vector is produced while their stored text stays complete -- the loss is invisible
from the outside, which is the point the count is there to make.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunking import MODEL_MAX_TOKENS, chunk_pages  # noqa: E402
from app.core.embeddings import Embedder  # noqa: E402
from app.core.parsers import parse_document  # noqa: E402
from app.core.tokenization import load_tokenizer  # noqa: E402
from app.core.vectorstore import ChromaVectorStore  # noqa: E402

logger = logging.getLogger("eval")

CONFIGURATIONS = [(100, 20), (180, 40), (300, 60), (500, 100)]
TOP_K = 5


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison, since chunk text is already normalised."""
    return " ".join(text.split()).lower()


def load_corpus(corpus_dir: Path) -> dict[str, list[tuple[int, str]]]:
    documents = {}
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            try:
                documents[path.name] = parse_document(path)
            except Exception as error:
                logger.warning("skipping %s: %s", path.name, error)
    if not documents:
        raise SystemExit(f"no parseable documents in {corpus_dir}")
    return documents


def evaluate(
    size: int,
    overlap: int,
    documents: dict,
    questions: list[dict],
    tokenizer,
    embedder: Embedder,
    workdir: Path,
) -> dict:
    store = ChromaVectorStore(workdir / f"chroma_{size}_{overlap}")

    all_chunks = []
    truncated = 0
    for filename, pages in documents.items():
        chunks = chunk_pages(
            pages,
            tokenizer=tokenizer,
            chunk_size_tokens=size,
            chunk_overlap_tokens=overlap,
            # Raised so configurations above the ceiling can be measured rather
            # than rejected. Truncation is then counted explicitly below.
            hard_max_tokens=max(size, MODEL_MAX_TOKENS),
        )
        for chunk in chunks:
            if chunk.token_count > MODEL_MAX_TOKENS:
                truncated += 1
                logger.warning(
                    "TRUNCATED: %s chunk %d is %d tokens; the model will discard "
                    "%d of them before embedding",
                    filename,
                    chunk.chunk_index,
                    chunk.token_count,
                    chunk.token_count - MODEL_MAX_TOKENS,
                )
        store.add_chunks(
            filename, filename, chunks, embedder.embed_texts([c.text for c in chunks])
        )
        all_chunks.extend(chunks)

    token_counts = [chunk.token_count for chunk in all_chunks]
    per_question = []
    latencies = []

    for case in questions:
        started = time.perf_counter()
        results = store.search(embedder.embed_query(case["question"]), TOP_K)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

        expected = normalise(case["expected_answer_substring"])
        hit_positions = [
            position
            for position, result in enumerate(results, start=1)
            if expected in normalise(result.text)
        ]
        top_result = results[0] if results else None
        per_question.append(
            {
                "question": case["question"],
                "top1_similarity": round(top_result.similarity, 4) if top_result else None,
                "top5_similarities": [round(r.similarity, 4) for r in results],
                "recall_at_5": bool(hit_positions),
                "first_hit_rank": hit_positions[0] if hit_positions else None,
                "expected_file": case.get("expected_source_file"),
                "top1_file": top_result.filename if top_result else None,
                "top1_file_correct": (
                    top_result.filename == case.get("expected_source_file")
                    if top_result
                    else False
                ),
                "retrieval_ms": round(elapsed_ms, 2),
            }
        )

    recalled = sum(1 for q in per_question if q["recall_at_5"])
    top1_correct = sum(1 for q in per_question if q["top1_file_correct"])
    top1_scores = [q["top1_similarity"] for q in per_question if q["top1_similarity"]]

    return {
        "chunk_size_tokens": size,
        "chunk_overlap_tokens": overlap,
        "chunks_emitted": len(all_chunks),
        "mean_chunk_tokens": round(sum(token_counts) / len(token_counts), 1),
        "max_chunk_tokens": max(token_counts),
        "chunks_truncated_by_model": truncated,
        "truncation_rate": round(truncated / len(all_chunks), 4) if all_chunks else 0.0,
        "recall_at_5": round(recalled / len(questions), 4),
        "questions_recalled": recalled,
        "questions_total": len(questions),
        "top1_file_accuracy": round(top1_correct / len(questions), 4),
        "mean_top1_similarity": round(sum(top1_scores) / len(top1_scores), 4),
        "mean_retrieval_ms": round(sum(latencies) / len(latencies), 2),
        "questions": per_question,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'size/overlap':>14} {'chunks':>7} {'mean tok':>9} {'max tok':>8} "
        f"{'truncated':>10} {'recall@5':>9} {'top1 doc':>9} {'mean top1':>10} {'ret ms':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        label = f"{row['chunk_size_tokens']}/{row['chunk_overlap_tokens']}"
        flag = "  <-- over ceiling" if row["chunks_truncated_by_model"] else ""
        print(
            f"{label:>14} {row['chunks_emitted']:>7} {row['mean_chunk_tokens']:>9} "
            f"{row['max_chunk_tokens']:>8} {row['chunks_truncated_by_model']:>10} "
            f"{row['recall_at_5']:>9.3f} {row['top1_file_accuracy']:>9.3f} "
            f"{row['mean_top1_similarity']:>10.4f} {row['mean_retrieval_ms']:>7.1f}{flag}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/eval_results.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    documents = load_corpus(args.corpus)
    tokenizer = load_tokenizer()
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")

    print(f"corpus: {len(documents)} documents, {len(questions)} questions")
    print(f"model ceiling: {MODEL_MAX_TOKENS} word-piece tokens\n")

    workdir = Path(tempfile.mkdtemp(prefix="chunk_sweep_"))
    rows = []
    try:
        for size, overlap in CONFIGURATIONS:
            print(f"evaluating chunk_size={size} overlap={overlap} ...")
            rows.append(
                evaluate(size, overlap, documents, questions, tokenizer, embedder, workdir)
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print_table(rows)

    best = max(rows, key=lambda row: (row["recall_at_5"], row["mean_top1_similarity"]))
    payload = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_max_tokens": MODEL_MAX_TOKENS,
        "top_k": TOP_K,
        "corpus": sorted(documents),
        "question_count": len(questions),
        "best_by_recall": {
            "chunk_size_tokens": best["chunk_size_tokens"],
            "chunk_overlap_tokens": best["chunk_overlap_tokens"],
        },
        "configurations": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
