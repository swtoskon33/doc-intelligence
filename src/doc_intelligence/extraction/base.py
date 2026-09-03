"""Extractor interface shared by every backend.

All extractors take a RawDocument and return an ExtractionResult with per-field
confidence, so they can be benchmarked head to head on the same golden set.
"""
from __future__ import annotations

from doc_intelligence.types import ExtractionResult, RawDocument


class Extractor:
    """Base interface. `name` identifies the backend in benchmark reports."""

    name: str = "base"

    def extract(self, doc: RawDocument) -> ExtractionResult:  # pragma: no cover
        raise NotImplementedError
