"""
test_context_builder.py
-----------------------
Unit tests for context_builder.py covering:
1. Multiple retrieved chunks.
2. Single retrieved chunk.
3. Empty retrieval results (empty list, None).
4. Missing or malformed chunk fields (missing source_document, missing chunk_id, missing text, non-dict).
"""

import unittest
from context_builder import build_context, format_chunk, NO_CONTEXT_MESSAGE


class TestContextBuilder(unittest.TestCase):

    def test_single_retrieved_chunk(self):
        """Verify context preparation with a single valid chunk."""
        chunks = [
            {
                "chunk_id": "doc1.txt_chunk_1",
                "source_document": "doc1.txt",
                "text": "This is content from chunk 1."
            }
        ]
        context = build_context(chunks)
        self.assertIn("[doc1.txt | Chunk ID: doc1.txt_chunk_1]", context)
        self.assertIn("This is content from chunk 1.", context)

    def test_multiple_retrieved_chunks(self):
        """Verify context preparation preserves all chunks and orders them properly."""
        chunks = [
            {
                "chunk_id": "metrics.txt_chunk_1",
                "source_document": "metrics.txt",
                "text": "Precision measures relevance."
            },
            {
                "chunk_id": "metrics.txt_chunk_2",
                "source_document": "metrics.txt",
                "text": "Recall measures completeness."
            }
        ]
        context = build_context(chunks)
        self.assertIn("[metrics.txt | Chunk ID: metrics.txt_chunk_1]", context)
        self.assertIn("Precision measures relevance.", context)
        self.assertIn("[metrics.txt | Chunk ID: metrics.txt_chunk_2]", context)
        self.assertIn("Recall measures completeness.", context)
        # Verify separator exists between multiple chunks
        self.assertIn("-" * 50, context)

    def test_empty_retrieval_results(self):
        """Verify empty list and None return the clear fallback message."""
        self.assertEqual(build_context([]), NO_CONTEXT_MESSAGE)
        self.assertEqual(build_context(None), NO_CONTEXT_MESSAGE)

    def test_missing_or_malformed_fields(self):
        """
        Verify missing source_document, chunk_id, or text are handled safely
        without inventing false information.
        """
        # Chunk with missing source_document
        chunk_missing_source = {
            "chunk_id": "test_chunk_1",
            "text": "Some text content."
        }
        formatted = format_chunk(chunk_missing_source)
        self.assertIn("[Unknown Source | Chunk ID: test_chunk_1]", formatted)

        # Chunk with missing chunk_id
        chunk_missing_id = {
            "source_document": "sample.txt",
            "text": "Some text content."
        }
        formatted = format_chunk(chunk_missing_id)
        self.assertIn("[sample.txt | Chunk ID: Unknown Chunk ID]", formatted)

        # Chunk with missing/empty text
        chunk_missing_text = {
            "source_document": "sample.txt",
            "chunk_id": "c1",
            "text": None
        }
        formatted = format_chunk(chunk_missing_text)
        self.assertIn("[sample.txt | Chunk ID: c1]", formatted)
        self.assertIn("[No text content]", formatted)

        # Non-dict items in chunk list
        malformed_list = ["not a dict", 42, None]
        context = build_context(malformed_list)
        self.assertEqual(context, NO_CONTEXT_MESSAGE)

    def test_mixed_valid_and_malformed_chunks(self):
        """Verify that valid chunks are preserved even if some entries in the list are malformed."""
        chunks = [
            "invalid entry",
            {
                "chunk_id": "c1",
                "source_document": "docA.txt",
                "text": "Valid text here."
            }
        ]
        context = build_context(chunks)
        self.assertIn("[docA.txt | Chunk ID: c1]", context)
        self.assertIn("Valid text here.", context)


if __name__ == "__main__":
    unittest.main()
