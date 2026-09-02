"""Unit tests for domain types."""
import pytest

from doc_intelligence.types import DocumentType, ExtractionResult, Field


@pytest.mark.unit
def test_field_confidence_threshold():
    assert Field("x", "v", 0.8).is_confident
    assert not Field("x", "v", 0.5).is_confident
    assert not Field("x", None, 0.9).is_confident  # no value -> not confident


@pytest.mark.unit
def test_needs_review_when_any_field_low_confidence():
    good = Field("a", "1", 0.9)
    bad = Field("b", None, 0.0)
    assert ExtractionResult("d1", DocumentType.INVOICE, {"a": good, "b": bad}).needs_review
    assert not ExtractionResult("d2", DocumentType.INVOICE, {"a": good}).needs_review
