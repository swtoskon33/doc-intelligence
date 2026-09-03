"""Extraction must handle the formats the validator claims to support.

These are regression tests for a real bug: validation accepted Swiss and German
formats while the extraction patterns did not, so CHF 1'081.00 was extracted as "1"
and the document still passed validation.
"""
import pytest

from doc_intelligence.extraction.extractor import RuleExtractor
from doc_intelligence.ingest.documents import ingest_document
from doc_intelligence.validation.rules import validate_in_place


def _extract(text):
    return validate_in_place(RuleExtractor().extract(ingest_document("t", text)))


@pytest.mark.unit
def test_swiss_apostrophe_amount_is_not_truncated():
    r = _extract("INVOICE number I1. Total: CHF 1'081.00. Invoice date: 2026-01-01")
    assert r.value("total_amount") == "1'081.00"      # not "1"


@pytest.mark.unit
def test_german_amount_is_extracted_whole():
    r = _extract("Rechnung Nr I2. Betrag: EUR 1.081,00. Datum: 2026-01-01")
    assert r.value("total_amount") == "1.081,00"


@pytest.mark.unit
def test_swiss_dotted_date_is_extracted():
    r = _extract("INVOICE number I3. Total: 100.00. Invoice date: 15.09.2026")
    assert r.value("invoice_date") == "15.09.2026"


@pytest.mark.unit
def test_alphanumeric_iban_is_extracted():
    r = _extract("INVOICE number I4. Total: 100.00. Invoice date: 2026-01-01. "
                 "IBAN: GB29NWBK60161331926819")
    assert r.value("iban") == "GB29NWBK60161331926819"


@pytest.mark.unit
def test_iso_formats_still_work():
    r = _extract("INVOICE number I5. Total: CHF 1081.00. Invoice date: 2026-09-15")
    assert r.value("total_amount") == "1081.00"
    assert r.value("invoice_date") == "2026-09-15"
