"""Business-rule validation for extracted documents."""
from __future__ import annotations

import re

from doc_intelligence.schemas.registry import required_fields
from doc_intelligence.types import ExtractionResult, ValidationError

_DATE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value.replace(",", "").replace("'", ""))
    except (ValueError, AttributeError):
        return None


def _iban_valid(iban: str) -> bool:
    """IBAN mod-97 checksum (ISO 13616)."""
    s = iban.replace(" ", "").upper()
    if len(s) < 5 or not s[:2].isalpha():
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def validate(result: ExtractionResult) -> list[ValidationError]:
    """Run all business rules against an extraction result."""
    errors = []
    for name in required_fields(result.doc_type):
        if result.value(name) is None:
            errors.append(ValidationError(name, "required field missing"))
    for name in ("invoice_date", "due_date", "date", "effective_date"):
        v = result.value(name)
        if v is not None and not _DATE.match(v.strip()):
            errors.append(ValidationError(name, "date not in a recognised format"))
    iban = result.value("iban")
    if iban is not None and not _iban_valid(iban):
        errors.append(ValidationError("iban", "IBAN checksum failed"))
    total = _to_float(result.value("total_amount"))
    mwst = _to_float(result.value("mwst_amount"))
    rate = _to_float(result.value("mwst_rate"))
    if total is not None and mwst is not None and rate is not None and rate > 0:
        expected = total * rate / (100 + rate)
        if abs(expected - mwst) > max(0.05, expected * 0.02):
            errors.append(ValidationError("mwst_amount", "VAT amount inconsistent with total and rate"))
    return errors


def validate_in_place(result: ExtractionResult) -> ExtractionResult:
    """Attach validation errors to the result and return it."""
    result.validation_errors = validate(result)
    return result
