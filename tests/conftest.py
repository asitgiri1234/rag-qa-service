import pytest

from app.core.tokenization import load_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    """The real word-piece tokenizer for the configured embedding model.

    Loaded once per session; the first run downloads it into the HuggingFace cache.
    Tests use the real tokenizer rather than a stand-in because the behaviour under
    test *is* word-piece counting -- a whitespace-word fake would not exercise it.
    """
    return load_tokenizer()


def build_text(tokenizer, target_tokens: int, *, sentence_words: int = 12) -> str:
    """Prose of roughly ``target_tokens`` word-piece tokens, in real sentences."""
    vocabulary = [
        "retrieval", "systems", "depend", "on", "careful", "segmentation", "of",
        "source", "documents", "into", "passages", "that", "preserve", "meaning",
        "while", "remaining", "small", "enough", "for", "the", "encoder", "budget",
    ]
    words: list[str] = []
    sentences: list[str] = []
    index = 0
    while tokenizer.count(" ".join(sentences)) < target_tokens:
        words = [vocabulary[(index + offset) % len(vocabulary)] for offset in range(sentence_words)]
        index += sentence_words
        sentences.append(" ".join(words).capitalize() + ".")
    return " ".join(sentences)
