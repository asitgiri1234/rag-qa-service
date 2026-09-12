"""Generate the multi-page PDF used by the ingestion tests and the eval corpus.

PDFs cannot be committed as readable source, so the bytes are produced from text
kept here. Run after changing the corpus text:

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent

PAGES: list[list[str]] = [
    [
        "Retrieval Augmented Generation: Operating Notes",
        "A retrieval augmented generation system answers questions by first finding "
        "relevant passages in a document collection and then asking a language model "
        "to compose an answer from those passages alone. The retrieval step is what "
        "grounds the answer, and it is also where most production failures originate.",
        "The embedding model used throughout these notes is all-MiniLM-L6-v2. It "
        "produces vectors of 384 dimensions and truncates its input at 256 word-piece "
        "tokens. Any passage longer than that ceiling is silently cut off before the "
        "vector is produced, so the tail of an oversized chunk contributes nothing to "
        "retrieval while still appearing intact in the stored text.",
        "Cosine similarity is the distance metric. Because the vectors are normalised "
        "to unit length, cosine distance and one minus cosine similarity are the same "
        "quantity, and a score of 1.0 indicates an exact match.",
    ],
    [
        "Chunking Strategy",
        "Chunk size is the single most consequential parameter in the pipeline. Small "
        "chunks produce sharp, specific vectors but scatter a single fact across "
        "several records. Large chunks keep related sentences together but dilute the "
        "signal: one vector must represent everything the passage discusses, so the "
        "match for any individual fact within it weakens.",
        "The chunker used here splits recursively. It attempts paragraph boundaries "
        "first, falls back to sentence boundaries, then to word boundaries, and only "
        "cuts at an arbitrary token position when a single sentence is longer than the "
        "entire budget. This ordering keeps semantic units intact wherever the "
        "document's own structure allows it.",
        "Overlap exists to protect facts that straddle a boundary. A chunk overlap of "
        "40 tokens means the last 40 tokens of one chunk reappear at the start of the "
        "next, so a sentence split across the boundary is fully present in at least "
        "one record. Overlap costs storage and adds near-duplicate results.",
    ],
    [
        "Failure Modes and Metrics",
        "Vocabulary mismatch is the most common retrieval failure. A bi-encoder maps "
        "the question and the passage independently, so a question phrased with "
        "synonyms absent from the source text can score lower against the correct "
        "passage than against an unrelated one that happens to share surface wording.",
        "Multi-hop questions fail for a structural reason: the answer requires facts "
        "from two distant sections, and no single chunk contains both. Retrieval "
        "returns whichever half scores higher, and the model answers from an "
        "incomplete context.",
        "The two metrics worth tracking are retrieval latency and top-1 similarity. "
        "Latency should be reported as percentiles rather than a mean, because the "
        "distribution has a long tail dominated by index warm-up. Top-1 similarity is "
        "the honest early warning for a bad answer: when the best available passage "
        "scores poorly, the generated answer will be poor regardless of the model.",
        "A minimum similarity floor of 0.25 discards results that are almost certainly "
        "irrelevant. When nothing clears the floor the correct behaviour is to refuse "
        "to answer rather than to hand the model junk context and let it improvise.",
    ],
]


def build_pdf(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(destination), pagesize=LETTER, title="RAG Operating Notes"
    )

    flowables = []
    for index, page in enumerate(PAGES):
        heading, *paragraphs = page
        flowables.append(Paragraph(heading, styles["Heading1"]))
        flowables.append(Spacer(1, 12))
        for text in paragraphs:
            flowables.append(Paragraph(text, styles["BodyText"]))
            flowables.append(Spacer(1, 8))
        if index < len(PAGES) - 1:
            flowables.append(PageBreak())

    document.build(flowables)


def main() -> int:
    target = ROOT / "tests" / "fixtures" / "rag_notes.pdf"
    build_pdf(target)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
