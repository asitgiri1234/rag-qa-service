"""Run questions designed to break retrieval and dump what came back.

    python scripts/find_failures.py

Three adversarial categories:

* ``vocabulary_mismatch`` -- the question uses synonyms the source text does not
  contain. A bi-encoder compares meanings, but it is not immune to surface form.
* ``boundary_spanning`` -- the answer straddles two chunks, so no single record
  holds all of it.
* ``multi_hop`` -- the answer needs facts from two distant sections, often in two
  different documents.

Writes a readable report to ``eval/failure_report.md`` and the raw data to
``eval/failure_report.json``. Findings here feed the retrieval-failure section of
EXPLANATIONS.md, which must describe an observed failure rather than a theorised one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunking import chunk_pages  # noqa: E402
from app.core.embeddings import Embedder  # noqa: E402
from app.core.parsers import parse_document  # noqa: E402
from app.core.tokenization import load_tokenizer  # noqa: E402
from app.core.vectorstore import ChromaVectorStore  # noqa: E402

TOP_K = 5
SNIPPET = 150


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def build_index(corpus: Path, workdir: Path, size: int, overlap: int):
    tokenizer = load_tokenizer()
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    store = ChromaVectorStore(workdir / "chroma")

    for path in sorted(corpus.iterdir()):
        if not path.is_file():
            continue
        pages = parse_document(path)
        chunks = chunk_pages(
            pages,
            tokenizer=tokenizer,
            chunk_size_tokens=size,
            chunk_overlap_tokens=overlap,
        )
        store.add_chunks(
            path.name, path.name, chunks, embedder.embed_texts([c.text for c in chunks])
        )
    return store, embedder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    parser.add_argument("--questions", type=Path, default=Path("eval/adversarial.json"))
    parser.add_argument("--report", type=Path, default=Path("eval/failure_report.md"))
    parser.add_argument("--json", type=Path, default=Path("eval/failure_report.json"))
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--min-similarity", type=float, default=0.25)
    args = parser.parse_args(argv)

    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    workdir = Path(tempfile.mkdtemp(prefix="failures_"))

    try:
        store, embedder = build_index(args.corpus, workdir, args.chunk_size, args.overlap)

        records = []
        for case in cases:
            results = store.search(embedder.embed_query(case["question"]), TOP_K)
            expected = normalise(case["expected_answer_substring"])
            hits = [
                position
                for position, result in enumerate(results, start=1)
                if expected in normalise(result.text)
            ]
            records.append(
                {
                    "category": case["category"],
                    "question": case["question"],
                    "note": case.get("note", ""),
                    "expected_answer_substring": case["expected_answer_substring"],
                    "expected_source_file": case.get("expected_source_file"),
                    "answer_present_in_top5": bool(hits),
                    "first_hit_rank": hits[0] if hits else None,
                    "below_floor": bool(results) and results[0].similarity < args.min_similarity,
                    "results": [
                        {
                            "rank": position,
                            "similarity": round(result.similarity, 4),
                            "filename": result.filename,
                            "chunk_index": result.chunk_index,
                            "page_number": result.page_number,
                            "contains_answer": expected in normalise(result.text),
                            "snippet": " ".join(result.text.split())[:SNIPPET],
                        }
                        for position, result in enumerate(results, start=1)
                    ],
                }
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    write_report(records, args, cases)
    args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    failed = [r for r in records if not r["answer_present_in_top5"]]
    misranked = [r for r in records if r["answer_present_in_top5"] and r["first_hit_rank"] > 1]
    print(f"\n{len(records)} adversarial questions at chunk_size={args.chunk_size}")
    print(f"  answer absent from top-{TOP_K}: {len(failed)}")
    print(f"  answer present but not ranked first: {len(misranked)}")
    for record in failed:
        print(f"  MISS [{record['category']}] {record['question'][:70]}")
    print(f"\nwrote {args.report} and {args.json}")
    return 0


def write_report(records: list[dict], args, cases) -> None:
    lines = [
        "# Retrieval failure report",
        "",
        f"Corpus `{args.corpus}`, chunk_size={args.chunk_size}, "
        f"overlap={args.overlap}, top_k={TOP_K}, similarity floor {args.min_similarity}.",
        "",
        "Generated by `scripts/find_failures.py`. Every score below is an observed "
        "cosine similarity, not an estimate.",
        "",
    ]

    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)

    lines += ["## Summary", "", "| category | questions | answer in top-5 | answer ranked 1st |", "|---|---|---|---|"]
    for category, group in by_category.items():
        found = sum(1 for r in group if r["answer_present_in_top5"])
        first = sum(1 for r in group if r["first_hit_rank"] == 1)
        lines.append(f"| {category} | {len(group)} | {found}/{len(group)} | {first}/{len(group)} |")
    lines.append("")

    for category, group in by_category.items():
        lines += [f"## {category.replace('_', ' ').title()}", ""]
        for record in group:
            verdict = (
                f"answer found at rank {record['first_hit_rank']}"
                if record["answer_present_in_top5"]
                else "ANSWER NOT RETRIEVED"
            )
            lines += [
                f"### {record['question']}",
                "",
                f"*{record['note']}*",
                "",
                f"**Outcome: {verdict}.** Looking for: `{record['expected_answer_substring']}` "
                f"in `{record['expected_source_file']}`.",
                "",
                "| rank | score | source | chunk | has answer | snippet |",
                "|---|---|---|---|---|---|",
            ]
            for result in record["results"]:
                mark = "yes" if result["contains_answer"] else "no"
                snippet = result["snippet"].replace("|", "\\|")
                lines.append(
                    f"| {result['rank']} | {result['similarity']:.4f} | {result['filename']} "
                    f"| {result['chunk_index']} | {mark} | {snippet} |"
                )
            lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
