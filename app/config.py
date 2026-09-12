"""Application settings, loaded from the environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"

    chunk_size_tokens: int = 180
    chunk_overlap_tokens: int = 40
    top_k: int = 5

    #: Retrieved chunks scoring below this are discarded rather than sent to the LLM
    #: as context. Set from the observed score distribution -- see EXPLANATIONS.md.
    min_similarity: float = 0.25

    data_dir: Path = DATA_DIR
    chroma_path: Path = DATA_DIR / "chroma"
    sqlite_path: Path = DATA_DIR / "metadata.db"
    metrics_path: Path = DATA_DIR / "metrics.jsonl"

    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed once per process."""
    return Settings()
