"""Extractor registry: select a backend and expose them for benchmarking.

Backends:
  rule - regex driven by the YAML schema. Offline, deterministic, the CI default.
  llm  - LLM with structured JSON output (OpenAI / Azure OpenAI). Production path.
  hf   - transformer question answering over the document text.
  layoutlmv3 - fine-tuned LayoutLMv3 token classification over words and boxes
               (layout-aware; needs a trained checkpoint and word bounding boxes).

All three implement the same Extractor interface, so scripts/benchmark_extractors.py
can score them side by side on one golden set.
"""
from __future__ import annotations

import os

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.schemas.registry import load_schema
from doc_intelligence.types import ExtractionResult, Field, RawDocument


class RuleExtractor(Extractor):
    """Regex extraction driven by the per-type schema. Offline, deterministic."""

    name = "rule"

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


def get_extractor(backend: str | None = None) -> Extractor:
    """Return an extractor by name (default from EXTRACTION_BACKEND, else rule)."""
    name = (backend or os.getenv("EXTRACTION_BACKEND", "rule")).lower()
    if name == "llm":
        from doc_intelligence.extraction.llm import LLMExtractor

        return LLMExtractor()
    if name == "hf":
        from doc_intelligence.extraction.hf import HFExtractor

        return HFExtractor()
    if name == "layoutlmv3":
        from doc_intelligence.extraction.layoutlm import LayoutLMv3Extractor

        return LayoutLMv3Extractor()
    return RuleExtractor()
