"""Unit tests for business-rule validation."""
import pytest

from doc_intelligence.types import DocumentType, ExtractionResult, Field
from doc_intelligence.validation.rules import _iban_valid, validate


def _result(doc_type, **field_values):
    fields = {k: Field(k, v, 0.9) for k, v in field_values.items()}
    return ExtractionResult("d1", doc_type, fields)


@pytest.mark.unit
def test_iban_checksum():
    assert _iban_valid("CH9300762011623852957")      # real Swiss IBAN
    assert not _iban_valid("CH0000000000000000000")


@pytest.mark.unit
def test_missing_required_field_flags_error():
    res = _result(DocumentType.INVOICE, invoice_number="INV-1", total_amount="100")
    errors = validate(res)  # invoice_date missing
    assert any(e.field == "invoice_date" for e in errors)


@pytest.mark.unit
def test_bad_date_format_flags_error():
    res = _result(DocumentType.INVOICE, invoice_number="INV-1", total_amount="100",
                  invoice_date="not-a-date")
    assert any(e.field == "invoice_date" and "format" in e.message for e in validate(res))


@pytest.mark.unit
def test_mwst_consistency():
    # total 108.10, rate 8.1% -> VAT ~= 8.10; a wrong 20.00 should fail
    good = _result(DocumentType.INVOICE, invoice_number="I1", total_amount="108.10",
                   invoice_date="2026-01-01", mwst_amount="8.10", mwst_rate="8.1")
    assert not any(e.field == "mwst_amount" for e in validate(good))
    bad = _result(DocumentType.INVOICE, invoice_number="I1", total_amount="108.10",
                  invoice_date="2026-01-01", mwst_amount="20.00", mwst_rate="8.1")
    assert any(e.field == "mwst_amount" for e in validate(bad))


@pytest.mark.unit
def test_validation_failure_triggers_review():
    res = _result(DocumentType.INVOICE, invoice_number="INV-1", total_amount="100")
    res.validation_errors = validate(res)
    assert res.needs_review
    assert not res.is_valid
    assert any("validation" in r for r in res.review_reasons)
