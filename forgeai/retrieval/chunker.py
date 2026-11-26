"""
chunker.py - splits loaded documents into overlapping chunks of words.

Why chunk?  The model can only read a small amount of text per call, so we
search for the few most relevant pieces and send only those.

Why overlap? If a key sentence falls on the boundary between two chunks,
overlap means it still appears whole in one of them.

Chunks never cross a page boundary. That keeps every citation exact
("leave_policy.pdf, page 2"), at the cost of some short chunks at page ends.
"""

import json
from pathlib import Path

from forgeai import config
from forgeai.retrieval.models import Chunk, LoadedDocument


def split_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """
    Split a list of words into windows of `chunk_size` words, where each window
    starts `chunk_size - overlap` words after the previous one.

    Example with chunk_size=5, overlap=2:
        a b c d e f g h  ->  [a b c d e] [d e f g h]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    if len(words) <= chunk_size:
        return [words] if words else []

    step = chunk_size - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start:start + chunk_size])
        if start + chunk_size >= len(words):  # this window reached the end
            break
    return windows


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Turn every chunkable document into a list of Chunk objects."""
    chunk_size = config.CHUNK_SIZE_WORDS if chunk_size is None else chunk_size
    overlap = config.CHUNK_OVERLAP_WORDS if overlap is None else overlap

    chunks: list[Chunk] = []
    for doc in documents:
        if not doc.chunked:
            continue
        for page in doc.pages:
            # .split() splits on any whitespace and also removes PDF line breaks.
            windows = split_words(page.text.split(), chunk_size, overlap)
            for index, window in enumerate(windows):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.name}::p{page.page}::c{index}",
                        doc_name=doc.name,
                        page=page.page,
                        chunk_index=index,
                        text=" ".join(window),
                        word_count=len(window),
                    )
                )
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Save chunks as a JSON artifact (part of the evidence for each run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump() for chunk in chunks]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunks saved by save_chunks(). Pydantic re-checks every field."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in data]
