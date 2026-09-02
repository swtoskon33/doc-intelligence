"""Unit tests for the evaluation metrics."""
import pytest

from doc_intelligence.eval.metrics import evaluate
from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import ingest_document


@pytest.mark.unit
def test_perfect_extraction_scores_one():
    doc = ingest_document("inv1", "INVOICE number INV-42. Total: 1250.00. Due date: 2026-09-15")
    result = RuleExtractor().extract(doc)
    truth = {"inv1": {"invoice_number": "INV-42", "total_amount": "1250.00", "due_date": "2026-09-15"}}
    report = evaluate([result], truth)
    assert report.overall_accuracy == 1.0
    assert report.per_field["invoice_number"].f1 == 1.0


@pytest.mark.unit
def test_missing_field_hurts_recall():
    doc = ingest_document("inv1", "INVOICE number INV-42")
    result = RuleExtractor().extract(doc)
    truth = {"inv1": {"invoice_number": "INV-42", "total_amount": "999.00"}}
    report = evaluate([result], truth)
    assert report.per_field["total_amount"].recall == 0.0
