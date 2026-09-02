"""Unit tests for document splitting."""
import pytest

from doc_intelligence.splitting.splitter import split_documents
from doc_intelligence.types import DocumentType


@pytest.mark.unit
def test_splits_batch_into_documents():
    pages = ["INVOICE number INV-42", "line items continued", "RECEIPT total paid 45"]
    docs = split_documents(pages)
    assert len(docs) == 2
    assert docs[0].page_range == (0, 1)      # invoice spans its continuation page
    assert docs[0].doc_type == DocumentType.INVOICE
    assert docs[1].page_range == (2, 2)
    assert docs[1].doc_type == DocumentType.RECEIPT


@pytest.mark.unit
def test_single_document_batch():
    docs = split_documents(["INVOICE number INV-1", "more lines"])
    assert len(docs) == 1
    assert docs[0].page_range == (0, 1)


@pytest.mark.unit
def test_empty_batch():
    assert split_documents([]) == []
