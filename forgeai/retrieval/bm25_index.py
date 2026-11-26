"""
bm25_index.py - keyword search over chunks using BM25.

How BM25 works, briefly: a chunk scores higher when it contains the query's
words, especially rare words (a word found in every chunk tells you little)
and when those words appear several times. Long chunks are slightly penalised
so they don't win just by containing more words.

BM25 matches words, not meaning: "time off" will NOT find "leave". That is
a known limitation; the Auditor will show us when it causes retrieval misses.
"""

import math
import re

from rank_bm25 import BM25Okapi

from forgeai import config
from forgeai.retrieval.models import Chunk, SearchResult

# Very common English words that carry no meaning for search. Removing them
# stops a query like "what is the leave policy" matching every chunk on "the".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "of", "on", "or", "the", "to",
    "what", "when", "where", "which", "who", "will", "with", "my", "me", "we",
    "you", "your", "this", "that", "there", "their", "any", "much", "many",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A comma between two digits, as in "6,000" or "1,00,000" (Indian format).
DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")


def tokenize(text: str) -> list[str]:
    """
    Lowercase the text, join numbers written with commas, keep only letters
    and digits, and drop stopwords.
    Example: "What is the Annual-Leave limit? INR 6,000" -> ["annual", "leave", "limit", "inr", "6000"]
    """
    text = DIGIT_COMMA.sub("", text.lower())
    return [token for token in TOKEN_PATTERN.findall(text) if token not in STOPWORDS]


class _PositiveIdfBM25(BM25Okapi):
    """
    BM25Okapi with a word-weight (IDF) formula that is never negative.

    rank_bm25's default gives a NEGATIVE weight to any word found in more than
    half of the chunks. With very few chunks (e.g. one short document) every
    word is "common", every score is negative, and search finds nothing.
    We use the formula from Lucene/Elasticsearch instead:
        idf = log(1 + (N - n + 0.5) / (n + 0.5))
    (N = number of chunks, n = chunks containing the word). Rare words still
    weigh more than common ones, but every weight is above zero.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


class BM25Index:
    """A searchable index built from a list of chunks."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build a search index from zero chunks.")
        self.chunks = chunks
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        # rank_bm25 fails if every chunk is empty after tokenising; guard with a placeholder token.
        self._bm25 = _PositiveIdfBM25([tokens or ["_empty_"] for tokens in tokenized])

    def idf(self, term: str) -> float:
        """How distinctive a word is: high if it appears in few chunks, 0 if in none."""
        return float(self._bm25.idf.get(term, 0.0))

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Return up to `top_k` chunks that best match `query`, best first.

        Chunks with a score of 0 or less (no useful word in common with the query)
        are left out, so the result can be shorter than top_k, or empty.
        An empty result is meaningful: nothing in the documents matches.
        """
        top_k = config.DEFAULT_TOP_K if top_k is None else top_k
        query_tokens = tokenize(query)
        if not query_tokens or top_k <= 0:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Sort by score (highest first); ties keep document order, so results are repeatable.
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])

        results = []
        for i in order:
            if scores[i] <= 0 or len(results) >= top_k:
                break
            results.append(SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=len(results) + 1))
        return results
