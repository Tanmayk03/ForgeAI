"""
check_retrieval.py - checks document loading, chunking and BM25 search.

Run from the project root (the ForgeAI folder):
    python -m scripts.check_retrieval
    python -m scripts.check_retrieval --docs data\\my_folder --query "how many sick days"
    python -m scripts.check_retrieval --chunk-size 40 --overlap 10

No LLM is used, so this runs in about a second.

Steps:
    1. Load every file in the folder and report what was used or skipped.
    2. Split documents into chunks.
    3. Save the chunks to JSON and load them back (checks the artifact round-trip).
    4. Run search queries and show the best-matching chunks.
"""

import argparse
import sys

from forgeai import config
from forgeai.retrieval.bm25_index import BM25Index
from forgeai.retrieval.chunker import chunk_documents, load_chunks, save_chunks
from forgeai.retrieval.loader import load_documents

DEFAULT_QUERIES = [
    "How many days of annual leave do shore staff get?",
    "Is a medical certificate needed for sick leave?",
    "What is the hotel limit per night for business travel?",
    "What is the weather in Mumbai tomorrow?",  # should find nothing useful
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check document loading and BM25 search.")
    parser.add_argument("--docs", default=str(config.DATA_DIR / "sample_docs"), help="document folder")
    parser.add_argument("--query", action="append", help="search query (repeat for several)")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP_WORDS)
    args = parser.parse_args()

    # 1. Load
    print(f"[1/4] Loading documents from {args.docs}")
    try:
        loaded = load_documents(args.docs)
    except FileNotFoundError as exc:
        print(f"      FAILED: {exc}")
        return 1
    for doc in loaded.documents:
        status = f"{len(doc.pages)} page(s)" if doc.chunked else "recorded, not chunked"
        print(f"      used    {doc.name:28} {doc.file_type:4} {status}")
        if doc.note:
            print(f"              note: {doc.note}")
    for skipped in loaded.skipped:
        print(f"      skipped {skipped.name:28} reason: {skipped.reason}")
    if not any(doc.chunked for doc in loaded.documents):
        print("      FAILED: no searchable documents found.")
        return 1

    # 2. Chunk
    print(f"[2/4] Chunking (size={args.chunk_size} words, overlap={args.overlap})")
    chunks = chunk_documents(loaded.documents, chunk_size=args.chunk_size, overlap=args.overlap)
    sizes = [c.word_count for c in chunks]
    print(f"      {len(chunks)} chunks, words per chunk: min {min(sizes)}, max {max(sizes)}, "
          f"average {sum(sizes) / len(sizes):.0f}")
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        print("      FAILED: chunk ids are not unique.")
        return 1

    # 3. Save and reload
    out_path = config.RUNS_DIR / "retrieval_check" / "chunks.json"
    save_chunks(chunks, out_path)
    if load_chunks(out_path) != chunks:
        print("      FAILED: chunks changed after saving and reloading.")
        return 1
    print(f"[3/4] Saved and reloaded {out_path} - identical")

    # 4. Search
    print(f"[4/4] Searching (top_k={args.top_k})")
    index = BM25Index(chunks)
    for query in args.query or DEFAULT_QUERIES:
        print(f"\n  Q: {query}")
        results = index.search(query, top_k=args.top_k)
        if not results:
            print("     (no matching chunks)")
        for r in results:
            preview = r.chunk.text[:90] + ("..." if len(r.chunk.text) > 90 else "")
            print(f"     #{r.rank} score {r.score:5.2f}  {r.chunk.chunk_id}")
            print(f"        {preview}")

    print("\nRetrieval check finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
