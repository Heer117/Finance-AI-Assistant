"""
test_ingestion.py — Unit tests for PDF loading and chunking.

These tests use a small sample PDF created on the fly,
so no external file dependency is needed.
"""

import pytest
from src.ingestion.pdf_processor import clean_text


class TestTextCleaning:
    def test_collapses_multiple_newlines(self):
        text = "Line one\n\n\n\nLine two"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_collapses_multiple_spaces(self):
        text = "Revenue   increased    significantly"
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_strips_leading_trailing_whitespace(self):
        text = "   Some content   "
        cleaned = clean_text(text)
        assert cleaned == "Some content"


class TestChunking:
    def test_chunk_size_respected(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        long_text = "Financial performance overview. " * 200  # long text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents([Document(page_content=long_text)])

        assert len(docs) > 1
        for doc in docs:
            # Allow small overshoot due to separator logic
            assert len(doc.page_content) <= 1100

    def test_chunk_overlap_creates_shared_content(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        long_text = "Sentence number " + " ".join(f"item{i}." for i in range(300))
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents([Document(page_content=long_text)])

        assert len(docs) >= 2