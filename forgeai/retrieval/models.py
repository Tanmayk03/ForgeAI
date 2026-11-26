"""
models.py - the data shapes used by document loading and retrieval.

These are Pydantic models: Python classes that check their own fields.
If a field has the wrong type or is missing, Pydantic raises an error
immediately, instead of letting bad data travel through the pipeline.
They can also be saved to and loaded from JSON in one line.
"""

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """The text of one page. Plain-text files count as a single page (page 1)."""

    page: int = Field(ge=1, description="Page number, starting at 1")
    text: str


class LoadedDocument(BaseModel):
    """One file from the document folder, after reading."""

    name: str = Field(
        description="Path relative to the document folder, e.g. 'leave_policy.pdf' or "
        "'hr/leave_policy.pdf'. Used in chunk ids and citations."
    )
    path: str = Field(description="Full path on disk")
    file_type: str = Field(description="'pdf', 'txt', 'md' or 'csv'")
    pages: list[PageText] = Field(default_factory=list)
    # CSV files are recorded but not chunked yet (decision: handle them with the CSV tool later).
    chunked: bool = Field(default=True, description="False if this file is not split into chunks")
    note: str = Field(default="", description="Anything worth reporting, e.g. 'page 3 had no text'")


class SkippedFile(BaseModel):
    """A file that could not be used, and why. Reported, never silently ignored."""

    name: str
    path: str
    reason: str


class LoadResult(BaseModel):
    """Everything the loader found in a folder."""

    folder: str
    documents: list[LoadedDocument] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)


class Chunk(BaseModel):
    """One searchable piece of a document."""

    chunk_id: str = Field(description="Unique id, e.g. 'leave_policy.pdf::p2::c1'")
    doc_name: str
    page: int
    chunk_index: int = Field(description="Position of this chunk within its page, starting at 0")
    text: str
    word_count: int


class SearchResult(BaseModel):
    """One search hit: a chunk plus how well it matched the query."""

    chunk: Chunk
    score: float = Field(description="BM25 score. Higher = better match. Only comparable within one query.")
    rank: int = Field(description="1 = best match")
