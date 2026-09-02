"""Unit tests for schema-driven extraction (Swiss invoice fields)."""
import pytest

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.schemas.registry import load_schema, required_fields
from doc_intelligence.types import DocumentType


@pytest.mark.unit
def test_invoice_schema_has_swiss_fields():
    names = [s.name for s in load_schema(DocumentType.INVOICE)]
    assert {"iban", "mwst_amount", "mwst_rate", "currency"} <= set(names)


@pytest.mark.unit
def test_required_fields_from_schema():
    assert set(required_fields(DocumentType.INVOICE)) == {"invoice_number", "total_amount", "invoice_date"}


@pytest.mark.unit
def test_extracts_swiss_invoice_fields():
    text = ("INVOICE number INV-42. Total: CHF 1250.00. MWST rate 8.1%. "
            "IBAN: CH9300762011623852957. Invoice date: 2026-09-15. Currency CHF")
    result = RuleExtractor().extract(ingest_document("inv1", text))
    assert result.value("iban") == "CH9300762011623852957"
    assert result.value("mwst_rate") == "8.1"
    assert result.value("currency") == "CHF"
