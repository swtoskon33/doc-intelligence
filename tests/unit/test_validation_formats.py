"""Regression tests for the validation fixes: Swiss dates, amounts, IBAN structure."""
import pytest

from doc_intelligence.types import DocumentType, ExtractionResult, Field
from doc_intelligence.validation.rules import (
    AmountParseError,
    _iban_valid,
    _is_real_date,
    _to_float,
    validate,
)


def _res(**kw):
    return ExtractionResult("d1", DocumentType.INVOICE,
                            {k: Field(k, v, 0.9) for k, v in kw.items()})


@pytest.mark.unit
@pytest.mark.parametrize("value", ["15.09.2026", "2026-09-15", "15/09/2026"])
def test_accepts_real_dates_in_swiss_and_iso_formats(value):
    assert _is_real_date(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["9999/99/9999", "2026-13-01", "2026-02-30", "not-a-date"])
def test_rejects_impossible_dates(value):
    assert not _is_real_date(value)


@pytest.mark.unit
@pytest.mark.parametrize(("raw", "expected"), [
    ("1081.00", 1081.0),
    ("1'081.00", 1081.0),      # Swiss apostrophe separator
    ("1.081,00", 1081.0),      # German format
    ("1,081.00", 1081.0),      # English format
    ("108.10", 108.1),
])
def test_parses_european_and_english_amounts(raw, expected):
    assert _to_float(raw) == pytest.approx(expected)


@pytest.mark.unit
def test_unparseable_amount_raises_rather_than_returning_none():
    with pytest.raises(AmountParseError):
        _to_float("not-a-number")


@pytest.mark.unit
def test_unparseable_amount_becomes_a_validation_error():
    # the regression that matters: a bad amount must not silently skip the VAT check
    errors = validate(_res(invoice_number="I1", invoice_date="15.09.2026",
                           total_amount="not-a-number"))
    assert any("not parseable" in e.message for e in errors)


@pytest.mark.unit
def test_vat_without_total_is_flagged_not_skipped():
    errors = validate(_res(invoice_number="I1", invoice_date="15.09.2026",
                           mwst_amount="8.10", mwst_rate="8.1"))
    assert any(e.field == "total_amount" for e in errors)


@pytest.mark.unit
def test_iban_requires_structure_not_just_checksum():
    assert _iban_valid("CH9300762011623852957")
    assert not _iban_valid("CH93")                      # too short
    assert not _iban_valid("CHAB00762011623852957")     # check digits not numeric
    assert not _iban_valid("CH93 0076 2011 6238 5295 7 EXTRA LONG PADDING VALUE")
