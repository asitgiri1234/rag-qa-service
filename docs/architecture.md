# Architecture

A text description of components and data flow, written so a diagram can be drawn
from it directly.

## The two pipelines

The service has two paths that share exactly one component. Keeping them separate in
the diagram is the point: they run at different times, on different threads, with
different latency budgets.

```
                                ┌──────────────────┐
                                │     Embedder     │
                                │ all-MiniLM-L6-v2 │
                                │  384-dim, 256tok │
                                └──────────────────┘
                                   ▲            ▲
        ingestion (async) ─────────┘            └───────── query (sync)
```

The embedder is drawn **once**, with both lanes touching it. Encoding documents with
one model and questions with another puts the two sets of vectors in unrelated spaces;
retrieval then returns plausible-looking scores that mean nothing, and no component
reports an error. One embedder instance, reached through one accessor, is the
structural guarantee against that.

---

## Pipeline 1 — Ingestion (asynchronous)

```
client
  │  POST /documents  (multipart)
  ▼
┌─────────────────────────────────────────┐
│ app/api/documents.py                    │
│  · extension in {.pdf,.txt,.md}         │
│  · declared content type                │
│  · size <= MAX_UPLOAD_MB (streamed)     │
│  · PDF magic bytes sniffed, not trusted │
└─────────────────────────────────────────┘
  │                              │
  │ write file                   │ insert row (status=pending)
  ▼                              ▼
data/uploads/{uuid}{ext}    ┌──────────────────────┐
  │                         │ SQLite: documents    │
  │                         │ app/storage/db.py    │
  │                         └──────────────────────┘
  │ enqueue(document_id, path)
  ▼
┌──────────────────────┐
│ queue.Queue          │      202 Accepted returns here, immediately
│ app/core/worker.py   │
└──────────────────────┘
  │ single daemon thread
  ▼
┌──────────────────────────────────────────────────────────┐
│ app/core/ingest.py — status=processing                   │
│                                                          │
│  parse   app/core/parsers.py   -> [(page_no, text)]      │
│  chunk   app/core/chunking.py  -> [Chunk]                │
│  embed   app/core/embeddings.py-> [[float] x 384]        │
│  store   app/core/vectorstore.py                         │
│                                                          │
│  each phase timed separately                             │
└──────────────────────────────────────────────────────────┘
  │                    │                      │
  │ vectors + metadata │ status=completed     │ one JSONL line
  ▼                    ▼                      ▼
┌──────────────┐  ┌──────────────┐  ┌────────────────────┐
│ Chroma       │  │ SQLite       │  │ data/metrics.jsonl │
│ data/chroma  │  │ chunk_count  │  │ per-phase timings  │
└──────────────┘  └──────────────┘  └────────────────────┘
```

**Failure path.** Any exception sets `status=failed` and persists the message. A job
is never left at `processing`. On startup, rows still marked `processing` are failed
outright, because the in-process queue does not survive a restart and nothing is
working on them.

**Why a worker thread, not `BackgroundTasks`.** `BackgroundTasks` runs after the
response but inside the request machinery, occupying a thread from the same pool
Starlette uses for every synchronous endpoint. Embedding is CPU-bound and takes
seconds per document, so concurrent uploads would starve that pool and stall unrelated
requests. A queue also gives the job an identity: state in SQLite that
`GET /documents/{id}` can report, and a queue depth that can be observed. One worker
is deliberate — the embedding model is the bottleneck, and a second thread would
contend for the same CPU and the same model instance.

---

## Pipeline 2 — Query (synchronous)

```
client
  │  POST /query  {question, top_k?, document_ids?}
  ▼
┌──────────────────────────────┐
│ app/api/query.py             │  rate limit 20/min
└──────────────────────────────┘
  ▼
┌──────────────────────────────────────────────┐
│ app/core/retrieval.py                        │
│  1. embed the question  ── same Embedder ──▶ │
│  2. search Chroma, top_k, optional filter    │
│  3. drop results below min_similarity (0.25) │
│  4. log latency + every similarity score     │
└──────────────────────────────────────────────┘
  │
  ├── nothing cleared the floor ──▶ return "no relevant content", sources=[],
  │                                 generation_ms=0.   The LLM is NOT called.
  │
  ▼ results survived
┌──────────────────────────────────────────────┐
│ app/core/generation.py                       │
│  · context numbered [1] filename p.3         │
│  · system prompt: answer ONLY from context   │
│  · temperature 0.1, max_tokens capped        │
│  · one retry on rate limit / timeout         │
└──────────────────────────────────────────────┘
  │  Groq API (openai/gpt-oss-120b)
  ▼
┌──────────────────────────────────────────────────────────┐
│ QueryResponse                                            │
│   answer, sources[{filename, page_number, chunk_index,   │
│                    similarity_score, text}],             │
│   retrieval_ms, generation_ms, total_ms                  │
└──────────────────────────────────────────────────────────┘
  │ one JSONL line
  ▼
data/metrics.jsonl
```

The similarity score travels all the way back to the client. Without it, a bad answer
and a bad retrieval are indistinguishable from outside the service.

---

## Components

| Component | Module | Responsibility |
|---|---|---|
| App factory | `app/main.py` | Wiring, lifespan, worker start/stop, model warm-up |
| Settings | `app/config.py` | One source of truth, loaded from `.env` |
| Upload API | `app/api/documents.py` | Validation, persistence, enqueue, status, delete |
| Query API | `app/api/query.py` | Retrieve → generate → respond with scores |
| Metrics API | `app/api/metrics.py` | Aggregated percentiles |
| Rate limiting | `app/api/limits.py` | One shared slowapi limiter |
| Errors | `app/api/errors.py` | `{"error": {type, message}}`, no stack traces |
| Parsers | `app/core/parsers.py` | PDF/TXT/MD → `[(page, text)]`, whitespace normalisation |
| Chunker | `app/core/chunking.py` | Structure-aware splitting, word-piece budget. **Pure** |
| Tokenizer | `app/core/tokenization.py` | Word-piece counting and offsets (isolated I/O) |
| Embedder | `app/core/embeddings.py` | One model, batched, L2-normalised |
| Vector store | `app/core/vectorstore.py` | `VectorStore` protocol + Chroma implementation |
| Retrieval | `app/core/retrieval.py` | Embed, search, apply floor, log scores |
| Generation | `app/core/generation.py` | Prompt assembly, Groq call, retry |
| Ingestion | `app/core/ingest.py` | The pipeline, per-phase timings, terminal failure |
| Worker | `app/core/worker.py` | Queue + daemon thread |
| Metrics | `app/core/metrics.py` | JSONL recording, percentile aggregation |
| Job state | `app/storage/db.py` | SQLite, stdlib `sqlite3`, no ORM |

## Data stores

| Store | Location | Holds |
|---|---|---|
| Chroma | `data/chroma/` | Vectors + chunk text + metadata (document_id, filename, chunk_index, page_number, token_count) |
| SQLite | `data/metadata.db` | Document rows and job status |
| Uploads | `data/uploads/` | Original files, named by document id |
| Metrics | `data/metrics.jsonl` | One JSON line per query and per ingestion |

All of `data/` is gitignored.

## Boundaries worth drawing

- **`VectorStore` is a protocol.** Chroma sits behind four methods — `add_chunks`,
  `search`, `delete_document`, `count`. Swapping in FAISS or pgvector is a new class,
  not a rewrite of retrieval.
- **`chunking.py` is pure.** No I/O, no globals, every parameter an argument. The
  tokenizer — which does network and disk I/O on load — lives in its own module and is
  passed in. This is what lets the evaluation sweep vary the configuration freely.
- **The embedder is shared, deliberately.** Both lanes must touch the same box in the
  diagram.
