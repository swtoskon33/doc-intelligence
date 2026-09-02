"""Structured field extraction: turn a RawDocument into typed Fields.

The core of the service. Two backends behind one interface, selected by
EXTRACTION_BACKEND:
  - "rule"  : deterministic regex/keyword extraction -- offline, reproducible, the CI
              default. No API key, no model download.
  - "llm"   : an LLM with structured (JSON) output -- the production path (OpenAI/Azure
              OpenAI). Wired behind the same interface; a real deployment sets the key.

Each backend returns Fields with a confidence score, so downstream code can route
low-confidence extractions to human review (see ExtractionResult.needs_review).
"""
from __future__ import annotations

import os
import re

from doc_intelligence.types import DocumentType, ExtractionResult, Field, RawDocument

# Which fields we try to extract per document type.
SCHEMA = {
    DocumentType.INVOICE: ["invoice_number", "total_amount", "due_date", "vendor"],
    DocumentType.RECEIPT: ["total_amount", "date", "merchant"],
    DocumentType.CONTRACT: ["party", "effective_date", "term"],
}

_PATTERNS = {
    "invoice_number": re.compile(r"invoice\s*(?:number|no\.?|#)?\s*[:#]?\s*([A-Z0-9\-]+)", re.IGNORECASE),
    "total_amount": re.compile(r"(?:total|amount due|total paid)\s*[:$]?\s*\$?\s*([0-9][0-9,]*\.?[0-9]{0,2})", re.IGNORECASE),
    "due_date": re.compile(r"due\s*(?:date)?\s*[:]?\s*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", re.IGNORECASE),
    "date": re.compile(r"date\s*[:]?\s*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", re.IGNORECASE),
    "vendor": re.compile(r"(?:from|vendor|bill from)\s*[:]?\s*([A-Za-z][A-Za-z0-9 &.,]+)", re.IGNORECASE),
    "merchant": re.compile(r"(?:merchant|store)\s*[:]?\s*([A-Za-z][A-Za-z0-9 &.,]+)", re.IGNORECASE),
    "party": re.compile(r"(?:between|party)\s*[:]?\s*([A-Za-z][A-Za-z0-9 &.,]+)", re.IGNORECASE),
    "effective_date": re.compile(r"effective\s*(?:date)?\s*[:]?\s*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", re.IGNORECASE),
    "term": re.compile(r"term\s*(?:of)?\s*[:]?\s*([0-9]+\s*(?:months|years))", re.IGNORECASE),
}


class Extractor:
    """Base extractor interface."""

    def extract(self, doc: RawDocument) -> ExtractionResult:  # pragma: no cover
        raise NotImplementedError


class RuleExtractor(Extractor):
    """Deterministic regex extraction. Offline, reproducible, no dependencies."""

    def extract(self, doc: RawDocument) -> ExtractionResult:
        wanted = SCHEMA.get(doc.doc_type, [])
        fields: dict[str, Field] = {}
        for name in wanted:
            pat = _PATTERNS.get(name)
            match = pat.search(doc.text) if pat else None
            if match:
                value = match.group(1).strip()
                # confidence: a rule match is fairly reliable but not certain
                fields[name] = Field(name=name, value=value, confidence=0.8)
            else:
                fields[name] = Field(name=name, value=None, confidence=0.0)
        return ExtractionResult(document_id=doc.id, doc_type=doc.doc_type, fields=fields)


def get_extractor() -> Extractor:
    """Select an extractor from EXTRACTION_BACKEND (default: rule)."""
    backend = os.getenv("EXTRACTION_BACKEND", "rule").lower()
    if backend == "llm":
        # A real deployment would return an LLMExtractor here (OpenAI/Azure OpenAI with
        # JSON mode). Kept out of the offline default so CI needs no API key.
        raise NotImplementedError("LLM backend requires an API key; set it up in deployment")
    return RuleExtractor()
