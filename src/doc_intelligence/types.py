"""Shared domain types for the document-processing pipeline.

A Document flows through the pipeline: ingest produces a RawDocument, extraction turns
it into an ExtractionResult (a set of typed Fields with confidence), and evaluation
compares extracted fields against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


@dataclass
class RawDocument:
    """A document as ingested: an id, its type, and the raw text extracted from it."""

    id: str
    doc_type: DocumentType
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Field:
    """One extracted field: a name, its value, and how confident the extractor is."""

    name: str
    value: str | None
    confidence: float

    @property
    def is_confident(self) -> bool:
        return self.value is not None and self.confidence >= 0.7


@dataclass
class ExtractionResult:
    """The structured output for one document: its fields, keyed by field name."""

    document_id: str
    doc_type: DocumentType
    fields: dict[str, Field]

    def value(self, name):
        f = self.fields.get(name)
        return f.value if f else None

    @property
    def needs_review(self) -> bool:
        """Route to a human if any present field is low-confidence."""
        return any(not f.is_confident for f in self.fields.values())
