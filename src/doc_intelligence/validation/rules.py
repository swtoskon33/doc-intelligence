"""Business-rule validation for extracted documents."""
from __future__ import annotations

import re
from datetime import date

from doc_intelligence.schemas.registry import required_fields
from doc_intelligence.types import ExtractionResult, ValidationError

# Accepted separators include "." because Swiss and German invoices write 15.09.2026.
_DATE_SHAPE = re.compile(r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})$")


_CURRENCY = re.compile(r"\b(CHF|EUR|USD|GBP)\b|[$\u20ac\u00a3]", re.IGNORECASE)


class AmountParseError(ValueError):
    """Raised when an amount cannot be parsed, so it cannot be skipped silently."""


def _is_real_date(value: str) -> bool:
    """True if the value is a date that actually exists on the calendar.

    Accepts ISO (2026-09-15), Swiss and German (15.09.2026), and slash forms
    (15/09/2026), including two-digit years common on receipts (15.09.26 -> 2026).
    Separators must be consistent: 15.09/2026 is a formatting error, not a date.
    """
    text = value.strip()
    match = re.fullmatch(r"(\d{1,4})([-/.])(\d{1,2})\2(\d{1,4})", text)
    if not match:
        return False

    a, _, b, d = match.groups()

    def expand_year(y: str) -> int:
        n = int(y)
        # a two-digit year on a business document is this century
        return 2000 + n if len(y) <= 2 else n

    candidates = []
    if len(a) == 4:                       # year-first: 2026-09-15
        candidates.append((int(a), int(b), int(d)))
    else:                                 # day-first: 15.09.2026 or 15.09.26
        candidates.append((expand_year(d), int(b), int(a)))
        candidates.append((expand_year(d), int(a), int(b)))   # tolerate month-first

    for year, month, day in candidates:
        if not (1900 <= year <= 2200):
            continue
        try:
            date(year, month, day)
        except ValueError:
            continue
        return True
    return False


def _to_float(value):
    """Parse an amount written in European or Anglo notation.

    Handles 1'081.00, 1.081,00, 1,081.00 and 1081. Raises rather than returning None on
    failure: a silently skipped amount means a wrong invoice passes validation, which is
    the opposite of what this module is for.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise AmountParseError(f"amount is not text: {value!r}")

    # a value like "CHF 1081.00" is a number with a currency attached, not a parse failure
    cleaned = _CURRENCY.sub("", value)
    cleaned = cleaned.strip().replace("'", "").replace(" ", "").replace("\u00a0", "")
    if not cleaned:
        raise AmountParseError("amount is empty")

    # decide which separator is the decimal one: the last separator, if it is followed
    # by one or two digits, is decimal; anything else groups thousands.
    last_comma, last_dot = cleaned.rfind(","), cleaned.rfind(".")
    decimal_pos = max(last_comma, last_dot)
    if decimal_pos != -1 and len(cleaned) - decimal_pos - 1 in (1, 2):
        integer = re.sub(r"[.,]", "", cleaned[:decimal_pos])
        cleaned = integer + "." + cleaned[decimal_pos + 1:]
    else:
        cleaned = re.sub(r"[.,]", "", cleaned)

    try:
        return float(cleaned)
    except ValueError as exc:
        raise AmountParseError(f"cannot parse amount {value!r}") from exc


def _iban_problem(iban: str) -> str:
    """Name the specific IBAN check that failed, so review_reasons stay useful."""
    s = iban.replace(" ", "").upper()
    if not (15 <= len(s) <= 34):
        return f"IBAN length {len(s)} outside the valid range 15-34"
    if not s[:2].isalpha():
        return "IBAN does not start with a country code"
    if not s[2:4].isdigit():
        return "IBAN check digits are not numeric"
    if not s.isalnum():
        return "IBAN contains non-alphanumeric characters"
    return "IBAN checksum failed"


def _iban_valid(iban: str) -> bool:
    """IBAN check: structure, length, then the mod-97 checksum (ISO 13616).

    Checksum alone is not enough -- roughly one in 97 random strings passes it. An IBAN
    must also be 15-34 characters, start with two letters (country) followed by two
    digits (check digits), and contain nothing but alphanumerics.
    """
    s = iban.replace(" ", "").upper()
    if not (15 <= len(s) <= 34):
        return False
    if not (s[:2].isalpha() and s[2:4].isdigit()):
        return False
    if not s.isalnum():
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
        if v is not None and not _is_real_date(v.strip()):
            errors.append(ValidationError(name, "not a valid date (bad format or invalid calendar date)"))
    iban = result.value("iban")
    if iban is not None and not _iban_valid(iban):
        errors.append(ValidationError("iban", _iban_problem(iban)))
    def amount(name):
        """Parse a numeric field; an unparseable value is a validation error, not a skip."""
        raw = result.value(name)
        if raw is None:
            return None
        try:
            return _to_float(raw)
        except AmountParseError:
            errors.append(ValidationError(name, f"amount not parseable: {raw!r}"))
            return None

    total = amount("total_amount")
    mwst = amount("mwst_amount")
    rate = amount("mwst_rate")
    if total is not None and mwst is not None and rate is not None and rate > 0:
        expected = total * rate / (100 + rate)
        if abs(expected - mwst) > max(0.05, expected * 0.02):
            errors.append(ValidationError(
                "mwst_amount", "VAT amount inconsistent with total and rate"))
    elif mwst is not None and rate is not None and total is None:
        # we have VAT figures but no total to check them against
        errors.append(ValidationError(
            "total_amount", "VAT present but total missing, consistency unverifiable"))

    return errors


def validate_in_place(result: ExtractionResult) -> ExtractionResult:
    """Attach validation errors to the result and return it."""
    result.validation_errors = validate(result)
    return result
