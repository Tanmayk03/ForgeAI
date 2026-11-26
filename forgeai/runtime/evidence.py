"""
evidence.py - checks done in plain code (no LLM) on the question, the
retrieved chunks and the answer.

Why this exists: BM25 returns chunks whenever ANY word overlaps, so
"a chunk was found" does not mean "the answer is in the documents".
These checks are cheap, repeatable, and cannot hallucinate.
"""

import re

from forgeai.retrieval.bm25_index import tokenize

# Numbers like 24, 6000, 6,000, 12.5, 2026. Commas inside numbers are removed before comparing.
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")

# Citation markers the model may write inside its answer, e.g. "[1]" or "[1, 3]".
# They are source numbers, not facts, so they are removed before checking numbers.
CITATION_MARKER = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def query_coverage(question: str, chunk_texts: list[str]) -> tuple[list[str], list[str], float]:
    """
    What fraction of the question's key words appear anywhere in the retrieved chunks?

    Returns (query_terms, covered_terms, coverage) where coverage is 0.0-1.0.
    Example: question "Do staff get maternity leave?" -> terms [staff, get, maternity, leave].
    If the chunks contain staff, get and leave but never "maternity", coverage = 3/4 = 0.75.

    This is a rough signal, not proof: high coverage does not mean the answer
    is present (the model must still decide), and a reworded question can score
    low even though the answer exists (a possible wrong refusal, which the
    Auditor will reveal and Repair can tune via min_query_coverage).
    """
    terms = list(dict.fromkeys(tokenize(question)))  # unique, order kept
    if not terms:
        return [], [], 0.0
    chunk_tokens: set[str] = set()
    for text in chunk_texts:
        chunk_tokens.update(tokenize(text))
    covered = [t for t in terms if t in chunk_tokens]
    return terms, covered, len(covered) / len(terms)


def _normalise_number(number: str) -> str:
    """'6,000' -> '6000'; '12.50' -> '12.5'; '24' -> '24'."""
    value = number.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def unsupported_numbers(answer: str, cited_texts: list[str]) -> list[str]:
    """
    Numbers in the answer that appear in none of the cited chunks.

    Invented figures are the most common and most damaging hallucination
    ("24 days" when the policy says 20), so they get their own check.
    Citation markers such as "[1]" are ignored. Whether a flagged answer is
    blocked is decided by the runner (output_contract.block_unsupported_numbers).
    Known trade-off: a correctly reworded number (e.g. "2 days" -> "48 hours")
    is also flagged.
    """
    answer = CITATION_MARKER.sub(" ", answer)
    source_numbers = {
        _normalise_number(n) for text in cited_texts for n in NUMBER_PATTERN.findall(text)
    }
    flagged = []
    for n in NUMBER_PATTERN.findall(answer):
        if _normalise_number(n) not in source_numbers and n not in flagged:
            flagged.append(n)
    return flagged


def estimate_tokens(text: str) -> int:
    """Rough token count: English averages about 1.3 tokens per word. Used only for warnings."""
    return int(len(text.split()) * 1.3) + 1
