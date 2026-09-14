# CLAUDE.md

Project constraints for `rag-qa-service`. These are binding for every session; if a
change would violate one, stop and raise it rather than working around it.

## What this is

A retrieval-augmented question-answering API. Documents are uploaded, chunked,
embedded and indexed on a background worker; questions are answered from retrieved
context by a Groq-hosted LLM, with the similarity score of every source returned.

## Hard constraints

- **Stack is FastAPI + Pydantic.** Settings come from `pydantic-settings`, request and
  response bodies are Pydantic models. No other web framework.
- **No LangChain, no LlamaIndex**, and no equivalent orchestration framework. Do not
  add one as a transitive convenience either.
- **Chunking and retrieval are written by hand.** Splitting, overlap, scoring, ranking
  and context assembly live in `app/core/` as our own code. Vector storage and the
  embedding model are the only parts delegated to libraries (Chroma,
  sentence-transformers).
- **Every retrieval must log latency and similarity scores.** A retrieval call that
  does not emit the per-query elapsed time and the similarity score of each returned
  chunk is incomplete. This is how the system is tuned; it is not optional telemetry.
- **Keep modules small and unit-testable.** One responsibility per module, pure
  functions where the work is a transformation, I/O pushed to the edges. If a module
  cannot be tested without standing up the app, it is too big.

## Invariants that are easy to break

- **One embedding model, both sides.** The same embedder must encode documents at
  ingestion and questions at query time. Two different models produce plausible scores
  and meaningless rankings, and nothing raises an error.
- **256 word-piece tokens is a hard ceiling.** `all-MiniLM-L6-v2` truncates there.
  `chunk_text` raises rather than emit a chunk that would be silently cut off. Measure
  in word-piece tokens from the model's own tokenizer — never characters, never
  whitespace words.
- **Chroma gets no embedding function.** Passing `embedding_function=None` is
  deliberate; the default would embed raw text with a second model.
- **Chroma returns distance, not similarity.** `similarity = 1 - distance`. Inverting
  this reverses the entire ranking.
- **`app/core/chunking.py` stays pure.** No I/O, no globals, every parameter an
  argument. Tokenizer loading lives in `app/core/tokenization.py` for this reason.
- **A job must never be left at `processing`.** Every failure path sets `failed` with
  the error; startup fails rows orphaned by a restart.
- **Never call the LLM with no usable context.** If nothing clears `min_similarity`,
  return the refusal. Generating from irrelevant context yields a confident wrong
  answer.

## Layout

```
app/
  main.py       App factory, lifespan, worker start/stop
  config.py     Settings (pydantic-settings, .env)
  api/          documents.py, query.py, metrics.py, limits.py, errors.py
  core/         parsers, chunking, tokenization, embeddings, vectorstore,
                retrieval, generation, ingest, worker, metrics
  models/       Pydantic schemas
  storage/      db.py -- SQLite job state, stdlib sqlite3, no ORM
  static/       index.html -- demo UI served at /, plain JS, calls only the public API
tests/
scripts/        ingest_local, eval_chunking, find_failures, make_fixtures
eval/           corpus, question sets, committed results
docs/           architecture.md
data/           gitignored: chroma index, sqlite db, uploads, metrics.jsonl
```

## Conventions

- Python 3.11+.
- Dependencies in a plain pinned `requirements.txt`. No Poetry, no PDM.
- Secrets only via `.env`, which is gitignored. Every key must also appear in
  `.env.example` with an empty or non-sensitive default.
- Anything written at runtime goes under `data/`.
- Tests that need the embedding model are marked `@pytest.mark.slow`.

## Gotchas discovered the hard way

- `LLM_MODEL`: Groq decommissioned `llama-3.3-70b-versatile`; it returns
  `404 model_not_found`. Current default is `openai/gpt-oss-120b`. Check Groq's
  `/models` endpoint before assuming a model exists.
- `.gitattributes` marks `*.pdf` binary. Without it a Windows clone converts LF to
  CRLF inside the committed PDF fixture and corrupts it.
- Windows will not unlink an open file: close the handle before deleting.
- Console output is cp1252 by default; set `PYTHONIOENCODING=utf-8` when printing
  model output, which contains non-breaking hyphens and smart quotes.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in GROQ_API_KEY
uvicorn app.main:app --reload
pytest                    # or: pytest -m "not slow"
```
