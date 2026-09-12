# CLAUDE.md

Project constraints for `rag-qa-service`. These are binding for every session; if a
change would violate one, stop and raise it rather than working around it.

## What this is

A retrieval-augmented question-answering API. Documents are uploaded, chunked,
embedded and indexed; questions are answered from retrieved context by a Groq-hosted
LLM.

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

## Layout

```
app/
  main.py       FastAPI app factory; exposes `app` for uvicorn
  config.py     Settings (pydantic-settings, .env)
  api/          routers
  core/         chunking, embedding, vectorstore, retrieval
  models/       Pydantic schemas
  storage/      sqlite metadata layer
tests/
scripts/
data/           gitignored: chroma index, sqlite db, uploads
```

## Conventions

- Python 3.11+.
- Dependencies in a plain `requirements.txt`. No Poetry, no PDM, no lockfile tooling.
- Secrets only via `.env`, which is gitignored. Every key must also appear in
  `.env.example` with an empty or non-sensitive default.
- Anything written at runtime goes under `data/`.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in GROQ_API_KEY
uvicorn app.main:app --reload
pytest
```
