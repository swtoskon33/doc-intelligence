"""Tests for the pipeline orchestrator, vendor memory, and trained classifier."""
import pytest
from doc_intelligence.models.classifier import DocumentClassifier

from doc_intelligence.memory.vendor import VendorMemory
from doc_intelligence.pipeline.orchestrator import Decision, DocumentPipeline
from doc_intelligence.types import DocumentType


@pytest.mark.unit
def test_pipeline_splits_batch_and_decides_each_document():
    pages = [
        "INVOICE number INV-1. Total: CHF 100.00. Invoice date: 2026-01-01",
        "RECEIPT. Merchant: Coop. Total: CHF 20.00. Date: 2026-01-02",
    ]
    out = DocumentPipeline().process_batch("b1", pages)
    assert len(out) == 2
    assert out[0].doc_type == DocumentType.INVOICE
    assert out[1].doc_type == DocumentType.RECEIPT
    assert all(d.decision in (Decision.AUTO_ACCEPT, Decision.NEEDS_REVIEW) for d in out)


@pytest.mark.unit
def test_review_decision_carries_reasons():
    out = DocumentPipeline().process_document("d1", "INVOICE number INV-1")
    assert out.decision is Decision.NEEDS_REVIEW
    assert out.reasons


@pytest.mark.unit
def test_vendor_memory_roundtrip_and_fuzzy_lookup():
    m = VendorMemory()
    m.remember("Alpine Supplies AG", {"iban": "CH9300762011623852957", "mwst_rate": "8.1"})
    assert m.lookup("Alpine Supplies AG")["mwst_rate"] == "8.1"
    assert m.lookup("alpine supplies")["iban"].startswith("CH")
    assert m.lookup("Unknown Ltd") == {}


@pytest.mark.unit
def test_memory_fills_missing_field_from_a_known_vendor():
    memory = VendorMemory()
    memory.remember("Alpine Supplies AG", {"mwst_rate": "8.1"})
    pipe = DocumentPipeline(memory=memory)
    # this document names the vendor but omits the VAT rate
    out = pipe.process_document(
        "d1", "INVOICE number INV-5. From: Alpine Supplies AG. Total: CHF 100.00. "
              "Invoice date: 2026-01-01")
    assert out.result.value("mwst_rate") == "8.1"
    assert out.result.fields["mwst_rate"].confidence < 0.8  # recalled, not read


@pytest.mark.unit
def test_trained_classifier_predicts_document_type():
    clf = DocumentClassifier.train(
        ["INVOICE number 1 total 100", "RECEIPT merchant coop total 20",
         "AGREEMENT between A and B", "INVOICE number 2 total 300"],
        ["invoice", "receipt", "contract", "invoice"],
    )
    assert clf.predict("INVOICE number 9 total 50") == DocumentType.INVOICE
    assert 0.0 <= clf.confidence("INVOICE number 9") <= 1.0
