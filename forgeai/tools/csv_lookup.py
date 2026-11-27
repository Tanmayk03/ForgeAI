"""
csv_lookup.py - finds rows in the agent's CSV files.

Input: search words, e.g. "fuel injector" or "SP-1002".
A row matches when it contains the query's key words (same tokenizer as
BM25 search). Rows matching at least half as many words as the best row
(and at least 2, when the best row matches 2 or more) are returned, best
first, at most 5, each labelled with its file and row number so answers
can cite them. Returning only the very best row(s) was too strict: for
"pilotage plus night surcharge" it hid the pilotage row needed for the total.
"""

import csv
import io
import math

from forgeai.retrieval.bm25_index import tokenize
from forgeai.retrieval.loader import read_text_file
from forgeai.tools.base import ToolContext, ToolError

MAX_ROWS = 5


def csv_lookup(tool_input: str, context: ToolContext) -> str:
    if not context.csv_paths:
        raise ToolError("this agent has no CSV files")
    query = set(tokenize(tool_input))
    if not query:
        raise ToolError("give some search words, e.g. a part number or name")

    scored = []  # (matching words, file name, row number, row)
    for path in context.csv_paths:
        try:
            reader = csv.DictReader(io.StringIO(read_text_file(path)))
            for number, row in enumerate(reader, start=2):  # row 1 is the header
                row_tokens = set(tokenize(" ".join(str(v) for v in row.values() if v)))
                hits = len(query & row_tokens)
                if hits:
                    scored.append((hits, path.name, number, row))
        except (ValueError, csv.Error) as exc:
            raise ToolError(f"could not read {path.name}: {exc}") from exc

    if not scored:
        return f"No rows match '{tool_input}'."
    best = max(s[0] for s in scored)
    threshold = max(2, math.ceil(best / 2)) if best >= 2 else 1
    kept = sorted((s for s in scored if s[0] >= threshold), key=lambda s: -s[0])
    lines = []
    for hits, name, number, row in kept[:MAX_ROWS]:
        cells = "; ".join(f"{k}={v}" for k, v in row.items() if k)
        lines.append(f"{name} row {number}: {cells}")
    return "\n".join(lines)
