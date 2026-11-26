"""Tests for BM25 search, including the very-small-corpus case that used to return nothing."""

import unittest

from forgeai.retrieval.bm25_index import BM25Index
from forgeai.retrieval.models import Chunk


def chunk(i, text):
    return Chunk(chunk_id=f"doc.txt::p1::c{i}", doc_name="doc.txt", page=1, chunk_index=i,
                 text=text, word_count=len(text.split()))


class TestBM25Index(unittest.TestCase):
    def test_single_chunk_corpus_finds_match(self):
        index = BM25Index([chunk(0, "Mars has two moons, Phobos and Deimos.")])
        results = index.search("How many moons does Mars have?")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].score, 0)

    def test_word_in_every_chunk_still_counts(self):
        index = BM25Index([chunk(0, "leave policy annual"), chunk(1, "leave policy sick")])
        self.assertEqual(len(index.search("leave")), 2)

    def test_rare_word_outranks_common_word(self):
        index = BM25Index([chunk(0, "leave days annual"), chunk(1, "leave days sick"), chunk(2, "leave hotel")])
        self.assertEqual(index.search("hotel")[0].chunk.chunk_index, 2)
        self.assertEqual(index.search("sick leave")[0].chunk.chunk_index, 1)

    def test_idf_ranks_rare_words_higher(self):
        index = BM25Index([chunk(0, "leave days annual"), chunk(1, "leave days sick")])
        self.assertGreater(index.idf("sick"), index.idf("leave"))
        self.assertEqual(index.idf("unknownword"), 0.0)

    def test_unrelated_query_finds_nothing(self):
        index = BM25Index([chunk(0, "Mars has two moons.")])
        self.assertEqual(index.search("weather in Mumbai"), [])


if __name__ == "__main__":
    unittest.main()
