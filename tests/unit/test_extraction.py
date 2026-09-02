"""Unit tests for ingestion and extraction."""
import pytest

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import infer_type, ingest_document
from doc_intelligence.types import DocumentType


@pytest.mark.unit
def test_infer_invoice_type():
    assert infer_type("INVOICE number 42, amount due 100") == DocumentType.INVOICE


@pytest.mark.unit
def test_ingest_rejects_empty():
    with pytest.raises(ValueError):
        ingest_document("d1", "   ")


@pytest.mark.unit
def test_rule_extractor_pulls_invoice_fields():
    doc = ingest_document("inv1", "INVOICE number INV-42. Total: 1250.00. Due date: 2026-09-15")
    result = RuleExtractor().extract(doc)
    assert result.value("invoice_number") == "INV-42"
    assert result.value("total_amount") == "1250.00"


@pytest.mark.unit
def test_missing_field_is_low_confidence():
    doc = ingest_document("inv1", "INVOICE number INV-42")
    result = RuleExtractor().extract(doc)
    assert result.fields["total_amount"].value is None
    assert result.needs_review
