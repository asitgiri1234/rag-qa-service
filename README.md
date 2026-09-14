# rag-qa-service

A retrieval-augmented question-answering API: upload PDF, text or markdown documents
and ask questions about them. Documents are parsed, chunked, embedded and indexed on a
background worker, so uploads return immediately and progress is polled. Questions are
answered by a language model restricted to the retrieved passages, and every answer
comes back with its sources and their similarity scores.

Chunking and retrieval are written by hand. There is no LangChain and no LlamaIndex —
see [Design decisions](#design-decisions).

- **[EXPLANATIONS.md](EXPLANATIONS.md)** — chunk-size measurement, an observed
  retrieval failure, and how the similarity floor was set. Every number comes from a
  file in this repo.
- **[docs/architecture.md](docs/architecture.md)** — components and data flow.
- **[eval/](eval/)** — the corpus, question set, sweep results and failure report.

---

## Architecture

Two pipelines that share exactly one component: the embedder.

### Ingestion (asynchronous)

`POST /documents` validates the upload, streams it to disk, writes a `pending` row to
SQLite, puts the job on an in-process queue and returns **202** in milliseconds. A
single daemon worker thread then runs the pipeline off the request path:

```
parse → chunk → embed → store in Chroma → status=completed
```

Each phase is timed separately. Any failure sets `status=failed` with the error
message; a job is never left stuck at `processing`. Progress is read back through
`GET /documents/{id}`.

### Query (synchronous)

`POST /query` embeds the question **with the same model used at ingestion**, searches
Chroma for the top *k* chunks, and discards anything below the similarity floor. If
nothing clears the floor it returns a plain "no relevant content" answer and **does not
call the language model** — generating from context already judged irrelevant produces
a confident wrong answer. Otherwise the surviving chunks are numbered into the prompt
and the model is instructed to answer only from them and cite the markers inline.

The embedder being shared is not an implementation detail. Encoding documents with one
model and questions with another places the two sets of vectors in unrelated spaces;
scores stay plausible, ranking becomes meaningless, and nothing raises an error.

---

## Setup

### Prerequisites

- Python 3.11 or newer
- A Groq API key — <https://console.groq.com/keys>
- ~2 GB of disk for PyTorch and the embedding model (CPU only; no GPU needed)

### Install

```bash
git clone https://github.com/asitgiri1234/rag-qa-service
cd rag-qa-service

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Then set `GROQ_API_KEY` in `.env`. Everything else has a working default.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must be the same for ingestion and query |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `CHUNK_SIZE_TOKENS` | `180` | Word-piece tokens per chunk; must be ≤ 256 |
| `CHUNK_OVERLAP_TOKENS` | `40` | Tokens carried into the next chunk |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MIN_SIMILARITY` | `0.25` | Below this a chunk is discarded |
| `MAX_UPLOAD_MB` | `20` | Upload size cap |
| `MAX_ANSWER_TOKENS` | `700` | Cap on generated answer length |
| `WARM_START` | `true` | Load the embedding model at startup, not on first upload |
| `CHROMA_PATH` / `SQLITE_PATH` / `METRICS_PATH` | under `data/` | Runtime state |

> **On `LLM_MODEL`:** Groq has decommissioned `llama-3.3-70b-versatile`; it now returns
> `404 model_not_found`. The default here was verified against Groq's live `/models`
> endpoint. If your account exposes a different set, override `LLM_MODEL`.

### Run

```bash
uvicorn app.main:app --reload
```

First start downloads the embedding model (~90 MB) into the HuggingFace cache.
Interactive docs at <http://127.0.0.1:8000/docs>.

### Demo UI

Open <http://127.0.0.1:8000/> for a small browser client: upload documents, watch
their status move from `pending` to `completed`, optionally tick documents to restrict
the search, then ask questions and see the answer with each source's page, chunk and
similarity score, plus retrieval and generation timings.

It is a thin client, not part of the system under evaluation: one static file,
[`app/static/index.html`](app/static/index.html), plain HTML/CSS/JS with no build step
and no dependencies, calling only the public endpoints documented below.

### Test

```bash
pytest                 # full suite (109 tests)
pytest -m "not slow"   # 98 tests; skips those that load the embedding model
```

`-m "not slow"` still downloads the tokenizer (~500 KB) on first run, since chunking
is measured in that tokenizer's word-piece tokens. It does not load the ~90 MB
embedding model.

---

## API reference

All errors share one shape:

```json
{ "error": { "type": "not_found", "message": "no document abc" } }
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /documents` — upload

Accepts `.pdf`, `.txt`, `.md`. Rate limit **10/minute**.

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@eval/corpus/vector_stores.md;type=text/markdown"
```

```json
{
  "document_id": "f248696c-f271-4635-b094-ca1603fad244",
  "status": "pending",
  "message": "accepted for processing; poll GET /documents/{id} for status"
}
```

`202 Accepted`. Validation: extension, declared content type, non-empty, size under
`MAX_UPLOAD_MB`, and for PDFs the `%PDF-` magic bytes — the extension is not trusted.

| Failure | Status | `error.type` |
|---|---|---|
| Unsupported extension or content type | 415 | `unsupported_media_type` |
| Empty file | 400 | `bad_request` |
| `.pdf` that is not a PDF | 400 | `bad_request` |
| Over `MAX_UPLOAD_MB` | 413 | `payload_too_large` |
| More than 10 uploads/minute | 429 | `rate_limit_exceeded` |

### `GET /documents/{id}` — status

Rate limit **30/minute**.

```bash
curl http://127.0.0.1:8000/documents/f248696c-f271-4635-b094-ca1603fad244
```

```json
{
  "document_id": "f3104d55-bb92-449f-b0e0-2a3b38f3cffb",
  "filename": "retrieval_evaluation.txt",
  "content_type": "application/octet-stream",
  "size_bytes": 3835,
  "status": "completed",
  "chunk_count": 6,
  "error": null,
  "created_at": "2026-09-12T18:22:51+00:00",
  "completed_at": "2026-09-12T18:22:51+00:00"
}
```

`status` is one of `pending`, `processing`, `completed`, `failed`. On `failed`, `error`
carries the reason. `404` if the id is unknown.

### `GET /documents` — list

```bash
curl "http://127.0.0.1:8000/documents?limit=1&offset=0"
```

```json
{
  "documents": [ { "document_id": "f3104d55-...", "filename": "retrieval_evaluation.txt", "status": "completed", "chunk_count": 6, "error": null, "created_at": "2026-09-12T18:22:51+00:00", "completed_at": "2026-09-12T18:22:51+00:00", "content_type": "application/octet-stream", "size_bytes": 3835 } ],
  "total": 4,
  "limit": 1,
  "offset": 0
}
```

### `DELETE /documents/{id}`

Removes the chunks from Chroma, the row from SQLite, and the uploaded file.

```bash
curl -X DELETE http://127.0.0.1:8000/documents/f248696c-f271-4635-b094-ca1603fad244
```

```json
{
  "document_id": "f248696c-f271-4635-b094-ca1603fad244",
  "deleted_chunks": 6,
  "message": "document and its chunks removed"
}
```

### `POST /query` — ask a question

Rate limit **20/minute**.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What index structure does Chroma use?", "top_k": 2}'
```

```json
{
  "answer": "Chroma uses an HNSW (Hierarchical Navigable Small World) index. [1]",
  "sources": [
    {
      "filename": "vector_stores.md",
      "page_number": 1,
      "chunk_index": 1,
      "similarity_score": 0.7099,
      "text": "## HNSW\n\nThe index used by Chroma is HNSW, Hierarchical Navigable Small World. It builds a layered graph..."
    },
    {
      "filename": "vector_stores.md",
      "page_number": 1,
      "chunk_index": 2,
      "similarity_score": 0.6409,
      "text": "## Chroma\n\nChroma is an embedded vector database. A PersistentClient writes to a local directory..."
    }
  ],
  "retrieval_ms": 37.77,
  "generation_ms": 790.65,
  "total_ms": 1533.88
}
```

Request fields: `question` (1–1000 chars, required), `top_k` (1–20, defaults to
`TOP_K`), `document_ids` (optional, restricts retrieval to those documents).

When nothing clears the similarity floor, the model is not called:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

```json
{
  "answer": "No relevant content was found in the indexed documents for this question.",
  "sources": [],
  "retrieval_ms": 29.3,
  "generation_ms": 0.0,
  "total_ms": 29.4
}
```

`503` with `error.type: service_unavailable` if the language model cannot be reached
after one retry.

### `GET /metrics/summary`

```bash
curl http://127.0.0.1:8000/metrics/summary
```

```json
{
  "queries": {
    "count": 25,
    "answered": 22,
    "refused": 3,
    "refusal_rate": 0.12,
    "retrieval_ms": { "count": 25, "mean": 32.942, "p50": 32.115, "p95": 41.809, "p99": 47.278, "min": 25.45, "max": 48.771 },
    "generation_ms": { "count": 22, "p50": 5287.106, "p95": 8810.322 },
    "top_similarity": {
      "mean": 0.5808, "p50": 0.6097, "p95": 0.7737, "min": 0.2999, "max": 0.7925,
      "distribution": { "0.2-0.3": 1, "0.3-0.4": 1, "0.4-0.5": 5, "0.5-0.6": 2, "0.6-0.7": 10, "0.7-0.8": 3 }
    }
  },
  "ingestion": {
    "documents": 4,
    "total_chunks": 23,
    "mean_chunks_per_second": 21.06,
    "mean_embed_ms_per_chunk": 31.8
  }
}
```

---

## Scripts

```bash
# Ingest files directly, without the API
python scripts/ingest_local.py eval/corpus/*.md

# Sweep chunk configurations -> eval/eval_results.json
python scripts/eval_chunking.py

# Run adversarial questions -> eval/failure_report.md
python scripts/find_failures.py

# Regenerate the PDF test fixture
python scripts/make_fixtures.py
```

---

## Design decisions

**FastAPI.** Pydantic validation is native rather than bolted on, so request and
response shapes are declared once and enforced automatically, and the OpenAPI document
is generated from the same declarations. Async request handling matters specifically
for upload streaming.

**No LangChain, no LlamaIndex — deliberately.** Chunking and retrieval *are* the
deliverable here. A framework would supply a `RecursiveCharacterTextSplitter` and a
`VectorStoreRetriever` and leave nothing to reason about: the chunk boundaries, the
token budget, the score conversion and the similarity floor would all be defaults
chosen by someone else, and the interesting decisions would be invisible. Writing them
by hand is what makes the 256-token ceiling in `chunking.py`, the `1 - distance`
conversion in `vectorstore.py` and the floor in `retrieval.py` explicit, testable and
defensible. The frameworks also carry large dependency trees and a habit of changing
their abstractions between minor versions.

**sentence-transformers rather than an embedding API.** Groq has no embeddings
endpoint, so an API-based embedder would mean a second vendor. Running locally also
removes per-query network latency from retrieval — measured p50 is 32 ms end to end —
and makes the 256-token ceiling inspectable at runtime rather than a documented claim.
`all-MiniLM-L6-v2` runs on CPU at ~21 chunks/second, which is ample here. The cost is
the ~2 GB PyTorch install.

**A worker thread rather than Celery.** `BackgroundTasks` was rejected outright: it
occupies a thread from the pool Starlette uses for every synchronous endpoint, and
CPU-bound embedding there would stall unrelated requests. An explicit
`queue.Queue` plus one daemon thread gives a real out-of-band worker with zero
infrastructure, job state that survives in SQLite, and an observable queue depth.
Celery with Redis or RabbitMQ is the scale-up path — it moves the same contract into
separate processes so work survives a restart and spreads across machines — but it is
a deployment change, not a redesign: only `enqueue_ingestion` would be rewritten. For
a single-instance service it would be infrastructure without benefit.

**Chroma behind a `VectorStore` protocol.** Chroma persists to a local directory with
no server to run, and supports metadata filtering before the vector search — which is
what makes per-document queries correct rather than a post-filter that silently returns
fewer results than requested. But it is reached through four methods (`add_chunks`,
`search`, `delete_document`, `count`), so swapping in FAISS or pgvector is a new class
rather than a rewrite of retrieval. Defining that interface cost close to nothing.

Two Chroma defaults are overridden deliberately: the collection is created with cosine
space (the default is L2), and **no embedding function is registered**, because Chroma
would otherwise attach its own model and embed raw text with it — putting two different
models in one system, which is the exact failure the shared-embedder rule exists to
prevent.

**Chunking measured in word-piece tokens.** A character budget maps to a wildly
variable token count — prose runs ~1.3 tokens per word, dense technical text three to
four — so a character-based chunker silently emits chunks whose tails never reach the
embedding. See [EXPLANATIONS.md](EXPLANATIONS.md#1-chunk-size).

---

## Known limitations

- **Single process, single worker.** The queue is in-memory, so a restart loses queued
  jobs; rows left at `processing` are failed at startup rather than resumed. Rate
  limiting is per-process, so behind multiple workers each enforces its own budget.
- **No authentication.** Every endpoint is open. This is a demonstration service.
- **Scanned PDFs are rejected, not OCR'd.** `pypdf` extracts no text from images, so
  such a document fails with a clear message rather than indexing nothing silently.
- **Dense retrieval only.** No BM25, no reranker. The observed consequence is measured
  in [EXPLANATIONS.md](EXPLANATIONS.md#2-a-retrieval-failure-observed): a question
  phrased in synonyms scored 0.1010 against the chunk that answers it, versus 0.3875
  for the same question in the source's wording.
- **Multi-hop questions are not handled.** A question needing facts from two distant
  sections retrieves whichever half scores higher. One of three multi-hop test
  questions failed outright.
- **The evaluation corpus is small** — 4 documents, 23 chunks, 22 questions. recall@5
  saturates at 1.000 for every chunk configuration and is therefore uninformative at
  this size; the conclusions rest on mean top-1 similarity and top-1 document accuracy
  instead. This is stated plainly rather than papered over.
- **The similarity floor has thin margin.** 0.25 sits 0.05 below the weakest answerable
  question observed, and one paraphrased question already falls below it. It should be
  re-measured on a larger corpus.
- **Generation dominates latency** — p50 5287 ms against retrieval's 32 ms. Responses
  are not streamed, so the client waits for the whole answer.
- **No incremental re-indexing.** Changing the chunk configuration requires
  re-ingesting every document.
