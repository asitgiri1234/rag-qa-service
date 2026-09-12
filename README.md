# rag-qa-service

A retrieval-augmented question-answering API: upload documents, ask questions, get
answers grounded in the retrieved passages.

FastAPI + Pydantic, Chroma for vector storage, sentence-transformers for embeddings,
Groq for generation. Chunking and retrieval are hand-written — see [CLAUDE.md](CLAUDE.md)
for the project constraints.

## Requirements

- Python 3.11+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in GROQ_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0"}
```

Interactive docs at http://127.0.0.1:8000/docs.

## Test

```bash
pytest
```

## Layout

| Path | Contents |
| --- | --- |
| `app/main.py` | FastAPI app factory |
| `app/config.py` | Settings loaded from `.env` |
| `app/api/` | Routers |
| `app/core/` | Chunking, embedding, vector store, retrieval |
| `app/models/` | Pydantic schemas |
| `app/storage/` | SQLite metadata layer |
| `tests/` | Tests |
| `scripts/` | One-off and maintenance scripts |
| `data/` | Chroma index, SQLite db, uploads (gitignored) |
