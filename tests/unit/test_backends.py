"""Tests for the multi-backend extractor interface."""
import pytest

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.extraction.extractor import RuleExtractor, get_extractor
from doc_intelligence.extraction.hf import HFExtractor, transformers_available
from doc_intelligence.extraction.llm import LLMExtractor
from doc_intelligence.ingest.documents import ingest_document


@pytest.mark.unit
def test_all_backends_implement_the_interface():
    for backend in ("rule", "llm", "hf"):
        ex = get_extractor(backend)
        assert isinstance(ex, Extractor)
        assert ex.name == backend


@pytest.mark.unit
def test_llm_without_key_returns_empty_fields_not_error():
    ex = LLMExtractor(api_key=None)
    result = ex.extract(ingest_document("inv1", "INVOICE number INV-1. Total: 100"))
    assert result.fields                      # schema fields are present
    assert all(f.value is None for f in result.fields.values())


@pytest.mark.unit
def test_hf_degrades_when_transformers_missing():
    ex = HFExtractor()
    if transformers_available():
        pytest.skip("transformers installed; offline degradation path not exercised")
    result = ex.extract(ingest_document("inv1", "INVOICE number INV-1"))
    assert all(f.value is None for f in result.fields.values())


@pytest.mark.unit
def test_rule_backend_is_the_default():
    assert isinstance(get_extractor(), RuleExtractor)
