"""Structured field extraction: turn a RawDocument into typed Fields.

Field definitions live in the schema registry (YAML), so both extractors work from the
same source of truth. Two backends behind one interface, selected by EXTRACTION_BACKEND:
  - "rule" : regex extraction driven by the schema -- offline, reproducible, CI default.
  - "llm"  : an LLM with structured JSON output -- the production path (OpenAI / Azure
             OpenAI), wired behind the same interface.
"""
from __future__ import annotations

import os

from doc_intelligence.schemas.registry import load_schema
from doc_intelligence.types import ExtractionResult, Field, RawDocument


class Extractor:
    def extract(self, doc: RawDocument) -> ExtractionResult:  # pragma: no cover
        raise NotImplementedError


class RuleExtractor(Extractor):
    """Regex extraction driven by the per-type schema. Offline, deterministic."""

    def extract(self, doc: RawDocument) -> ExtractionResult:
        specs = load_schema(doc.doc_type)
        fields: dict[str, Field] = {}
        for spec in specs:
            match = spec.pattern.search(doc.text)
            if match:
                value = match.group(1).strip() if match.groups() else match.group(0).strip()
                fields[spec.name] = Field(spec.name, value, confidence=0.8)
            else:
                fields[spec.name] = Field(spec.name, None, confidence=0.0)
        return ExtractionResult(document_id=doc.id, doc_type=doc.doc_type, fields=fields)


def get_extractor() -> Extractor:
    """Select an extractor from EXTRACTION_BACKEND (default: rule)."""
    backend = os.getenv("EXTRACTION_BACKEND", "rule").lower()
    if backend == "llm":
        raise NotImplementedError("LLM backend requires an API key; set it up in deployment")
    return RuleExtractor()
